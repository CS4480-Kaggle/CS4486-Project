"""
C3e_lightgbm_pool10_multiseed_ensemble.py

Model C — C3c pool10 multi-seed probability ensemble.

Purpose
-------
C3d showed that C3c (C1e + 10x10 pooled spatial features) beats C1e
consistently for seeds 42, 52, and 62.

This script:
- reuses saved C3c seed predictions when available;
- otherwise trains the missing seeds;
- averages probabilities from seeds [42, 52, 62];
- evaluates OOF / seen / unseen / test-like LogLoss;
- writes a Kaggle-ready submission.

No best-seed selection and no leaderboard-tuned weights.
"""

from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

C1E_SCRIPT = SCRIPT_DIR / "10_lightgbm_c1e_richer_image_gate.py"
C3C_SCRIPT = SCRIPT_DIR / "C3c_lightgbm_pool10_only_v2.py"

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 52, 62]
N_FOLDS = 5
EPS = 1e-7
MODEL_NAME = "lightgbm_c"
ENSEMBLE_RUN_ID = "lightgbm_c_c3e_pool10_multiseed_42_52_62"

ENSEMBLE_OOF_PATH = PRED_DIR / f"oof_{ENSEMBLE_RUN_ID}.csv"
ENSEMBLE_TEST_PATH = PRED_DIR / f"test_{ENSEMBLE_RUN_ID}.csv"
SUBMISSION_PATH = PRED_DIR / "submission_lightgbm_c3e_pool10_multiseed.csv"

PER_SEED_METRICS_PATH = METRIC_DIR / f"per_seed_{ENSEMBLE_RUN_ID}.csv"
SUMMARY_PATH = METRIC_DIR / f"summary_{ENSEMBLE_RUN_ID}.csv"
TEST_DISPERSION_PATH = METRIC_DIR / f"test_seed_dispersion_{ENSEMBLE_RUN_ID}.csv"
OOF_CORRELATION_PATH = METRIC_DIR / f"oof_seed_correlation_{ENSEMBLE_RUN_ID}.csv"
CONFIG_PATH = METRIC_DIR / f"config_{ENSEMBLE_RUN_ID}.json"


def load_module(path, module_name):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def safe_log_loss(y_true, pred):
    return float(
        log_loss(
            np.asarray(y_true, dtype=np.int64),
            clip_prob(pred),
            labels=[0, 1],
        )
    )


def seed_run_id(seed):
    return f"lightgbm_c_c3c_pool10_only_richer_gate_seed{seed}"


def seed_oof_path(seed):
    return PRED_DIR / f"oof_{seed_run_id(seed)}.csv"


def seed_test_path(seed):
    return PRED_DIR / f"test_{seed_run_id(seed)}.csv"


def set_seed(c1e, seed):
    c1e.SEED = int(seed)
    c1e.LGBM_PARAMS = dict(c1e.LGBM_PARAMS)
    c1e.LGBM_PARAMS["random_state"] = int(seed)


def load_saved_seed_predictions(seed, train_ids, test_ids, y, folds):
    op, tp = seed_oof_path(seed), seed_test_path(seed)
    if not (op.exists() and tp.exists()):
        return None
    from prediction_validation import read_csv, validate_oof, validate_test
    run_id = seed_run_id(seed)
    oof_df = validate_oof(read_csv(op), train_ids, y, folds, run_id)
    test_df = validate_test(read_csv(tp), test_ids, run_id)
    return {
        "oof": clip_prob(oof_df["prediction"].to_numpy(dtype=np.float64)),
        "test": clip_prob(test_df["prediction"].to_numpy(dtype=np.float64)),
        "source": "reused",
    }


def train_seed(
    c1e,
    seed,
    X_train,
    X_test,
    train_ids,
    test_ids,
    y,
    folds,
    train_angle,
    test_angle,
):
    print("\n" + "=" * 78)
    print(f"TRAIN C3c | SEED {seed}")
    print("=" * 78)

    set_seed(c1e, seed)

    oof = np.full(len(train_ids), np.nan, dtype=np.float64)
    test_matrix = np.full((N_FOLDS, len(test_ids)), np.nan, dtype=np.float64)

    for fold in range(N_FOLDS):
        result = c1e.run_fold(
            fold,
            X_train,
            X_test,
            y,
            folds,
            train_angle,
            test_angle,
        )

        va_idx = result["outer_valid_idx"]
        oof[va_idx] = result["gated_valid_prob"]
        test_matrix[fold] = result["gated_test_prob"]

        print(
            f"Fold {fold} | gated={result['gated_loss']:.6f} | "
            f"seen={result['seen_loss']:.6f} | "
            f"unseen={result['unseen_observed_loss']:.6f} | "
            f"test-like={result['test_like']:.6f}"
        )

    if np.isnan(oof).any() or np.isnan(test_matrix).any():
        raise ValueError(f"Seed {seed}: incomplete predictions")

    oof = clip_prob(oof)
    test = clip_prob(test_matrix.mean(axis=0))

    run_id = seed_run_id(seed)

    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": oof,
        "model_name": MODEL_NAME,
        "run_id": run_id,
    }).to_csv(seed_oof_path(seed), index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction": test,
        "model_name": MODEL_NAME,
        "run_id": run_id,
    }).to_csv(seed_test_path(seed), index=False)

    return {
        "oof": oof,
        "test": test,
        "source": "trained",
    }


def build_seen_diagnostics(c1e, y, folds, train_angle, test_angle):
    oof_seen = np.zeros(len(y), dtype=bool)
    test_seen_matrix = np.zeros(
        (N_FOLDS, len(test_angle)),
        dtype=bool,
    )

    for fold in range(N_FOLDS):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        mapping, global_prior = c1e.fit_group_mapping(
            train_angle[tr_idx],
            y[tr_idx],
        )

        valid_group = c1e.encode_group_features(
            train_angle[va_idx],
            mapping,
            global_prior,
        )

        test_group = c1e.encode_group_features(
            test_angle,
            mapping,
            global_prior,
        )

        oof_seen[va_idx] = (
            valid_group["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
        )

        test_seen_matrix[fold] = (
            test_group["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
        )

    return oof_seen, float(test_seen_matrix.mean())


def main():
    print("=" * 78)
    print("MODEL C — C3e POOL10 MULTI-SEED ENSEMBLE")
    print("=" * 78)
    print("Seeds:", SEEDS)
    print("Ensemble: equal-weight probability average")
    print("No best-seed selection.")

    c1e = load_module(C1E_SCRIPT, "c1e_for_c3e")
    c3c = load_module(C3C_SCRIPT, "c3c_for_c3e")

    train_df, test_df, train_ids, test_ids, y, folds = c1e.load_data()

    train_angle = pd.to_numeric(
        train_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    test_angle = pd.to_numeric(
        test_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    print("\nBuilding C1e 62 image features...")
    base_train = c1e.build_richer_image_features(train_df)
    base_test = c1e.build_richer_image_features(test_df)

    print("Building C3c pool10 362 image features...")
    train_arrays = c3c.extract_image_arrays(train_df)
    test_arrays = c3c.extract_image_arrays(test_df)

    X_train = c3c.build_pool10_features(base_train, train_arrays)
    X_test = c3c.build_pool10_features(base_test, test_arrays)

    print("Image feature count:", X_train.shape[1])

    oof_seen, mean_test_seen_rate = build_seen_diagnostics(
        c1e,
        y,
        folds,
        train_angle,
        test_angle,
    )

    observed = np.isfinite(train_angle)
    unseen_observed = (~oof_seen) & observed

    seed_results = {}

    for seed in SEEDS:
        saved = load_saved_seed_predictions(
            seed,
            train_ids,
            test_ids,
            y,
            folds,
        )

        if saved is not None:
            print(f"\nSeed {seed}: reusing saved predictions.")
            seed_results[seed] = saved
        else:
            seed_results[seed] = train_seed(
                c1e,
                seed,
                X_train,
                X_test,
                train_ids,
                test_ids,
                y,
                folds,
                train_angle,
                test_angle,
            )

    metric_rows = []
    oof_matrix = []
    test_matrix = []

    for seed in SEEDS:
        oof = seed_results[seed]["oof"]
        test = seed_results[seed]["test"]

        oof_matrix.append(oof)
        test_matrix.append(test)

        overall = safe_log_loss(y, oof)
        seen_loss = safe_log_loss(y[oof_seen], oof[oof_seen])
        unseen_loss = safe_log_loss(
            y[unseen_observed],
            oof[unseen_observed],
        )
        test_like = (
            mean_test_seen_rate * seen_loss
            + (1.0 - mean_test_seen_rate) * unseen_loss
        )

        metric_rows.append({
            "seed": seed,
            "source": seed_results[seed]["source"],
            "overall_oof_log_loss": overall,
            "seen_group_oof_log_loss": seen_loss,
            "unseen_observed_oof_log_loss": unseen_loss,
            "test_like_observed_reweighted_log_loss": test_like,
        })

    oof_matrix = np.stack(oof_matrix, axis=0)
    test_matrix = np.stack(test_matrix, axis=0)

    ensemble_oof = clip_prob(oof_matrix.mean(axis=0))
    ensemble_test = clip_prob(test_matrix.mean(axis=0))

    ensemble_overall = safe_log_loss(y, ensemble_oof)
    ensemble_seen = safe_log_loss(y[oof_seen], ensemble_oof[oof_seen])
    ensemble_unseen = safe_log_loss(
        y[unseen_observed],
        ensemble_oof[unseen_observed],
    )
    ensemble_test_like = (
        mean_test_seen_rate * ensemble_seen
        + (1.0 - mean_test_seen_rate) * ensemble_unseen
    )

    per_seed_df = pd.DataFrame(metric_rows)

    mean_single_seed_oof = float(
        per_seed_df["overall_oof_log_loss"].mean()
    )
    mean_single_seed_unseen = float(
        per_seed_df["unseen_observed_oof_log_loss"].mean()
    )
    mean_single_seed_test_like = float(
        per_seed_df["test_like_observed_reweighted_log_loss"].mean()
    )

    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": ensemble_oof,
        "model_name": MODEL_NAME,
        "run_id": ENSEMBLE_RUN_ID,
    }).to_csv(ENSEMBLE_OOF_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction": ensemble_test,
        "model_name": MODEL_NAME,
        "run_id": ENSEMBLE_RUN_ID,
    }).to_csv(ENSEMBLE_TEST_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "is_iceberg": ensemble_test,
    }).to_csv(SUBMISSION_PATH, index=False)

    per_seed_df.to_csv(PER_SEED_METRICS_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction_mean": ensemble_test,
        "prediction_std_across_seeds": test_matrix.std(axis=0, ddof=0),
        "prediction_min": test_matrix.min(axis=0),
        "prediction_max": test_matrix.max(axis=0),
    }).to_csv(TEST_DISPERSION_PATH, index=False)

    correlation_df = pd.DataFrame(
        np.corrcoef(oof_matrix),
        index=[f"seed{s}" for s in SEEDS],
        columns=[f"seed{s}" for s in SEEDS],
    )
    correlation_df.to_csv(OOF_CORRELATION_PATH)

    summary_df = pd.DataFrame([{
        "run_id": ENSEMBLE_RUN_ID,
        "seeds": json.dumps(SEEDS),
        "n_seeds": len(SEEDS),
        "ensemble_method": "equal_weight_probability_mean",
        "overall_oof_log_loss": ensemble_overall,
        "seen_group_oof_log_loss": ensemble_seen,
        "unseen_observed_oof_log_loss": ensemble_unseen,
        "test_like_observed_reweighted_log_loss": ensemble_test_like,
        "mean_test_seen_rate": mean_test_seen_rate,
        "mean_single_seed_oof": mean_single_seed_oof,
        "mean_single_seed_unseen": mean_single_seed_unseen,
        "mean_single_seed_test_like": mean_single_seed_test_like,
        "ensemble_minus_mean_seed_oof":
            ensemble_overall - mean_single_seed_oof,
        "ensemble_minus_mean_seed_unseen":
            ensemble_unseen - mean_single_seed_unseen,
        "ensemble_minus_mean_seed_test_like":
            ensemble_test_like - mean_single_seed_test_like,
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "run_id": ENSEMBLE_RUN_ID,
                "architecture": "C3c pool10-only",
                "seeds": SEEDS,
                "weights": {str(s): 1 / len(SEEDS) for s in SEEDS},
                "ensemble": "equal-weight probability mean",
                "feature_count": int(X_train.shape[1]),
                "note": (
                    "Seeds fixed in advance from prior robustness test; "
                    "no best-seed selection."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("C3e MULTI-SEED ENSEMBLE RESULT")
    print("=" * 78)

    print("\nPer-seed:")
    print(per_seed_df.to_string(index=False))

    print("\nEnsemble:")
    print(f"OOF       : {ensemble_overall:.6f}")
    print(f"Seen      : {ensemble_seen:.6f}")
    print(f"Unseen    : {ensemble_unseen:.6f}")
    print(f"Test-like : {ensemble_test_like:.6f}")

    print("\nCompared with mean single seed:")
    print(
        f"OOF delta       : "
        f"{ensemble_overall - mean_single_seed_oof:+.6f}"
    )
    print(
        f"Unseen delta    : "
        f"{ensemble_unseen - mean_single_seed_unseen:+.6f}"
    )
    print(
        f"Test-like delta : "
        f"{ensemble_test_like - mean_single_seed_test_like:+.6f}"
    )

    print("\nOOF seed correlation:")
    print(correlation_df.to_string())

    print("\nSaved:")
    for p in [
        ENSEMBLE_OOF_PATH,
        ENSEMBLE_TEST_PATH,
        SUBMISSION_PATH,
        PER_SEED_METRICS_PATH,
        SUMMARY_PATH,
        TEST_DISPERSION_PATH,
        OOF_CORRELATION_PATH,
        CONFIG_PATH,
    ]:
        print(" ", p)

    print("\nNo final oof_c.csv/test_c.csv was overwritten.")


if __name__ == "__main__":
    main()
