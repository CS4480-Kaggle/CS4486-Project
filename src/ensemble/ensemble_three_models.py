from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


EPS = 1e-6
STATIC_STEP = 0.01
STACK_C = 0.1

RUNS = {
    "a": "a2_14_mean3ch_wide_24_48_96_seed42",
    "b": (
        "r1_resnet18_highres_pretrained_image_angle_"
        "lr0p0001_wd0p0001_do0p3_adamw_augflip_"
        "ttanone_10ep_seed42_e2_highres_stem"
    ),
    "c": (
        "lightgbm_c_c5b_c4_plus_joint_pca30_"
        "multiseed_42_52_62"
    )
}


def validate_probabilities(values, name):
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")

    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name} contains values outside [0, 1].")

    return values


def read_oof(path):
    frame = pd.read_csv(path)
    required = {
        "id",
        "fold",
        "target",
        "prediction",
        "model_name",
        "run_id"
    }

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if frame["id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate IDs.")

    validate_probabilities(frame["prediction"], path.name)
    return frame


def read_test(path):
    frame = pd.read_csv(path)
    required = {
        "id",
        "prediction",
        "model_name",
        "run_id",
    }

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if frame["id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate IDs.")

    validate_probabilities(frame["prediction"], path.name)
    return frame


def align_frame(reference_ids, frame, name):
    if set(reference_ids) != set(frame["id"]):
        raise ValueError(f"{name} IDs do not match the reference IDs.")

    aligned = frame.set_index("id").loc[reference_ids].reset_index()
    return aligned


def load_oof_matrix(prediction_dir):
    frames = {
        key: read_oof(prediction_dir / f"oof_{run_id}.csv")
        for key, run_id in RUNS.items()
    }

    reference = frames["a"].copy()
    reference_ids = reference["id"].tolist()

    aligned = {
        "a": reference,
        "b": align_frame(reference_ids, frames["b"], "Model B OOF"),
        "c": align_frame(reference_ids, frames["c"], "Model C OOF")
    }

    for key in ("b", "c"):
        if not np.array_equal(reference["fold"].to_numpy(), aligned[key]["fold"].to_numpy()):
            raise ValueError(f"Model {key.upper()} fold assignments differ.")

        if not np.array_equal(reference["target"].to_numpy(), aligned[key]["target"].to_numpy()):
            raise ValueError(f"Model {key.upper()} targets differ.")

    matrix = np.column_stack([
        aligned["a"]["prediction"].to_numpy(dtype=np.float64),
        aligned["b"]["prediction"].to_numpy(dtype=np.float64),
        aligned["c"]["prediction"].to_numpy(dtype=np.float64)
    ])

    return reference, matrix


def load_test_matrix(prediction_dir):
    frames = {
        key: read_test(prediction_dir / f"test_{run_id}.csv")
        for key, run_id in RUNS.items()
    }

    reference = frames["a"].copy()
    reference_ids = reference["id"].tolist()

    aligned = {
        "a": reference,
        "b": align_frame(reference_ids, frames["b"], "Model B test"),
        "c": align_frame(reference_ids, frames["c"], "Model C test")
    }

    matrix = np.column_stack([
        aligned["a"]["prediction"].to_numpy(dtype=np.float64),
        aligned["b"]["prediction"].to_numpy(dtype=np.float64),
        aligned["c"]["prediction"].to_numpy(dtype=np.float64)
    ])

    return reference, matrix


def search_static_weights(targets, predictions, step=STATIC_STEP):
    units = int(round(1.0 / step))

    if not np.isclose(units * step, 1.0):
        raise ValueError("The weight step must divide 1.0 exactly.")

    best_loss = float("inf")
    best_weights = None

    for a_units in range(units + 1):
        for b_units in range(units - a_units + 1):
            c_units = units - a_units - b_units

            weights = np.array([a_units, b_units, c_units],dtype=np.float64) / units

            probabilities = predictions @ weights
            loss = log_loss(targets, probabilities, labels=[0, 1])

            if loss < best_loss:
                best_loss = float(loss)
                best_weights = weights

    return best_weights, best_loss


def crossfit_static(targets, predictions, folds):
    oof_predictions = np.zeros(len(targets), dtype=np.float64)
    records = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        validation_mask = folds == fold

        weights, selection_loss = search_static_weights(targets[train_mask], predictions[train_mask])

        fold_predictions = predictions[validation_mask] @ weights
        oof_predictions[validation_mask] = fold_predictions

        records.append({
            "fold": int(fold),
            "weight_a": float(weights[0]),
            "weight_b": float(weights[1]),
            "weight_c": float(weights[2]),
            "selection_log_loss": selection_loss,
            "fold_log_loss": log_loss(
                targets[validation_mask],
                fold_predictions,
                labels=[0, 1]
            )
        })

    return oof_predictions, pd.DataFrame(records)


def to_logit_features(probabilities):
    probabilities = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(probabilities) - np.log1p(-probabilities)


def crossfit_logistic(targets, predictions, folds):
    features = to_logit_features(predictions)
    oof_predictions = np.zeros(len(targets), dtype=np.float64)
    records = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        validation_mask = folds == fold

        model = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=2000)
        model.fit(features[train_mask], targets[train_mask])

        fold_predictions = model.predict_proba(features[validation_mask])[:, 1]

        oof_predictions[validation_mask] = fold_predictions

        records.append({
            "fold": int(fold),
            "coefficient_a": float(model.coef_[0, 0]),
            "coefficient_b": float(model.coef_[0, 1]),
            "coefficient_c": float(model.coef_[0, 2]),
            "intercept": float(model.intercept_[0]),
            "fold_log_loss": log_loss(
                targets[validation_mask],
                fold_predictions,
                labels=[0, 1]
            )
        })

    return oof_predictions, pd.DataFrame(records)


def save_predictions(
    prediction_dir,
    metric_dir,
    reference_oof,
    reference_test,
    run_id,
    model_name,
    oof_predictions,
    test_predictions,
    summary,
    fold_records=None
):
    validate_probabilities(oof_predictions, f"{run_id} OOF")
    validate_probabilities(test_predictions, f"{run_id} test")

    oof_output = pd.DataFrame({
        "id": reference_oof["id"],
        "fold": reference_oof["fold"],
        "target": reference_oof["target"],
        "prediction": oof_predictions,
        "model_name": model_name,
        "run_id": run_id
    })

    test_output = pd.DataFrame({
        "id": reference_test["id"],
        "prediction": test_predictions,
        "model_name": model_name,
        "run_id": run_id
    })

    submission = test_output[["id", "prediction"]].rename(columns={"prediction": "is_iceberg"})

    oof_output.to_csv(prediction_dir / f"oof_{run_id}.csv", index=False)
    test_output.to_csv(prediction_dir / f"test_{run_id}.csv", index=False)
    submission.to_csv(prediction_dir / f"submission_{run_id}.csv", index=False)
    pd.DataFrame([summary]).to_csv(metric_dir / f"summary_{run_id}.csv", index=False)

    if fold_records is not None:
        fold_records.to_csv(metric_dir / f"folds_{run_id}.csv", index=False)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    prediction_dir = repo_root / "artifacts" / "predictions"
    metric_dir = repo_root / "artifacts" / "metrics"

    reference_oof, oof_matrix = load_oof_matrix(prediction_dir)
    reference_test, test_matrix = load_test_matrix(prediction_dir)

    targets = reference_oof["target"].to_numpy(dtype=np.int64)
    folds = reference_oof["fold"].to_numpy(dtype=np.int64)

    summary_rows = []

    equal_run_id = "ensemble_abc_equal_seed42"
    equal_weights = np.array([1.0 / 3.0] * 3)
    equal_oof = oof_matrix @ equal_weights
    equal_test = test_matrix @ equal_weights
    equal_loss = log_loss(targets, equal_oof, labels=[0, 1])

    equal_summary = {
        "run_id": equal_run_id,
        "method": "equal_probability_average",
        "run_a": RUNS["a"],
        "run_b": RUNS["b"],
        "run_c": RUNS["c"],
        "weight_a": equal_weights[0],
        "weight_b": equal_weights[1],
        "weight_c": equal_weights[2],
        "overall_oof_log_loss": equal_loss,
    }
    save_predictions(
        prediction_dir,
        metric_dir,
        reference_oof,
        reference_test,
        equal_run_id,
        "ensemble_equal",
        equal_oof,
        equal_test,
        equal_summary
    )
    summary_rows.append(equal_summary)

    static_run_id = "ensemble_abc_static_crossfit_seed42"
    static_oof, static_fold_records = crossfit_static(targets, oof_matrix, folds)
    final_weights, full_selection_loss = search_static_weights(targets, oof_matrix)
    static_test = test_matrix @ final_weights
    static_loss = log_loss(targets, static_oof, labels=[0, 1])

    static_summary = {
        "run_id": static_run_id,
        "method": "crossfit_static_probability_blend",
        "run_a": RUNS["a"],
        "run_b": RUNS["b"],
        "run_c": RUNS["c"],
        "final_weight_a": final_weights[0],
        "final_weight_b": final_weights[1],
        "final_weight_c": final_weights[2],
        "full_oof_selection_log_loss": full_selection_loss,
        "overall_oof_log_loss": static_loss
    }
    save_predictions(
        prediction_dir,
        metric_dir,
        reference_oof,
        reference_test,
        static_run_id,
        "ensemble_static",
        static_oof,
        static_test,
        static_summary,
        static_fold_records
    )
    summary_rows.append(static_summary)

    stack_run_id = "ensemble_abc_logit_stack_c0p1_seed42"
    stack_oof, stack_fold_records = crossfit_logistic(
        targets,
        oof_matrix,
        folds
    )
    stack_loss = log_loss(
        targets,
        stack_oof,
        labels=[0, 1]
    )

    final_stack = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=2000)
    final_stack.fit(to_logit_features(oof_matrix), targets)
    stack_test = final_stack.predict_proba(to_logit_features(test_matrix))[:, 1]

    stack_summary = {
        "run_id": stack_run_id,
        "method": "crossfit_logit_logistic_stacking",
        "run_a": RUNS["a"],
        "run_b": RUNS["b"],
        "run_c": RUNS["c"],
        "regularization_c": STACK_C,
        "final_coefficient_a": final_stack.coef_[0, 0],
        "final_coefficient_b": final_stack.coef_[0, 1],
        "final_coefficient_c": final_stack.coef_[0, 2],
        "final_intercept": final_stack.intercept_[0],
        "overall_oof_log_loss": stack_loss
    }
    save_predictions(
        prediction_dir,
        metric_dir,
        reference_oof,
        reference_test,
        stack_run_id,
        "ensemble_logistic_stack",
        stack_oof,
        stack_test,
        stack_summary,
        stack_fold_records
    )
    summary_rows.append(stack_summary)

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(metric_dir / "comparison_ensemble_abc_seed42.csv",index=False)

    print(comparison[["run_id", "method", "overall_oof_log_loss"]].sort_values("overall_oof_log_loss"))
    print()
    print("Static final weights:", final_weights.tolist())
    print("Final stack coefficients:", final_stack.coef_[0].tolist())
    print("Final stack intercept:", float(final_stack.intercept_[0]))


if __name__ == "__main__":
    main()
