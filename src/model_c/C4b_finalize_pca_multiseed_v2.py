"""
C4b_finalize_pca_multiseed.py

Final Model-C robustness + ensemble script.

It validates:
    C4 = C3c pool10 + fold-safe PCA90
against:
    C3c pool10

Seeds:
    42, 52, 62

Seed 42 C4 predictions are reused from C4a.
Seeds 52/62 C4 are trained here.
C3c seed42/52/62 predictions are reused from the previous C3e work.

After paired robustness, this script also creates the equal-weight
C4 3-seed ensemble and a Kaggle submission.

This is the terminal Model-C experiment.
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
C4A_SCRIPT = SCRIPT_DIR / "C4a_lightgbm_pool10_foldsafe_pca.py"

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 52, 62]
TRAIN_SEEDS = [52, 62]
N_FOLDS = 5
EPS = 1e-7

C4_SEED42_RUN_ID = "lightgbm_c_c4a_pool10_foldsafe_pca90_seed42"
FINAL_RUN_ID = "lightgbm_c_c4b_pool10_pca90_multiseed_42_52_62"

FINAL_OOF_PATH = PRED_DIR / f"oof_{FINAL_RUN_ID}.csv"
FINAL_TEST_PATH = PRED_DIR / f"test_{FINAL_RUN_ID}.csv"
FINAL_SUBMISSION_PATH = PRED_DIR / "submission_lightgbm_c4b_pool10_pca90_multiseed.csv"

RESULTS_PATH = METRIC_DIR / "seed_robustness_c3c_vs_c4_pca90.csv"
PAIRED_PATH = METRIC_DIR / "seed_robustness_c3c_vs_c4_pca90_paired.csv"
SUMMARY_PATH = METRIC_DIR / f"summary_{FINAL_RUN_ID}.csv"
FOLD_PATH = METRIC_DIR / "seed_robustness_c4_pca90_by_fold.csv"
CONFIG_PATH = METRIC_DIR / f"config_{FINAL_RUN_ID}.json"


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
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


def set_seed(c1e, seed):
    c1e.SEED = int(seed)
    c1e.LGBM_PARAMS = dict(c1e.LGBM_PARAMS)
    c1e.LGBM_PARAMS["random_state"] = int(seed)


def c3c_run_id(seed):
    return f"lightgbm_c_c3c_pool10_only_richer_gate_seed{seed}"


def c4_run_id(seed):
    if seed == 42:
        return C4_SEED42_RUN_ID
    return f"lightgbm_c_c4a_pool10_foldsafe_pca90_seed{seed}"


def oof_path(run_id):
    return PRED_DIR / f"oof_{run_id}.csv"


def test_path(run_id):
    return PRED_DIR / f"test_{run_id}.csv"


def load_prediction_pair(run_id, train_ids, test_ids, y, folds):
    # Validate source values/metadata BEFORE clipping for numerical log loss.
    from prediction_validation import read_csv, validate_oof, validate_test
    oof_df = validate_oof(read_csv(oof_path(run_id)), train_ids, y, folds, run_id)
    test_df = validate_test(read_csv(test_path(run_id)), test_ids, run_id)
    return {
        "oof": clip_prob(oof_df["prediction"].to_numpy(dtype=np.float64)),
        "test": clip_prob(test_df["prediction"].to_numpy(dtype=np.float64)),
    }


def build_seen_mask(c1e, y, folds, train_angle, test_angle):
    oof_seen = np.zeros(len(y), dtype=bool)
    test_seen = np.zeros((N_FOLDS, len(test_angle)), dtype=bool)

    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]

        mapping, global_prior = c1e.fit_group_mapping(train_angle[tr], y[tr])

        vg = c1e.encode_group_features(train_angle[va], mapping, global_prior)

        tg = c1e.encode_group_features(test_angle, mapping, global_prior)

        oof_seen[va] = (vg["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5)

        test_seen[fold] = (tg["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5)

    return oof_seen, float(test_seen.mean())


def metrics(y, pred, seen, observed, mean_test_seen_rate):
    unseen = (~seen) & observed

    overall = safe_log_loss(y, pred)
    seen_loss = safe_log_loss(y[seen], pred[seen])
    unseen_loss = safe_log_loss(y[unseen], pred[unseen])

    test_like = (
        mean_test_seen_rate * seen_loss
        + (1.0 - mean_test_seen_rate) * unseen_loss
    )

    return {
        "overall_oof_log_loss": overall,
        "seen_group_oof_log_loss": seen_loss,
        "unseen_observed_oof_log_loss": unseen_loss,
        "test_like_observed_reweighted_log_loss": test_like,
    }


def train_missing_c4_seeds(
    c1e,
    c4a,
    train_df,
    test_df,
    train_ids,
    test_ids,
    y,
    folds,
    train_angle,
    test_angle,
):
    train_arrays = c4a.extract_image_arrays(train_df)
    test_arrays = c4a.extract_image_arrays(test_df)

    base_train = c4a.build_c3c_base_features(
        c1e,
        train_df,
        train_arrays,
    )
    base_test = c4a.build_c3c_base_features(
        c1e,
        test_df,
        test_arrays,
    )

    seed_oof = {
        seed: np.full(len(train_ids), np.nan)
        for seed in TRAIN_SEEDS
    }

    seed_test = {
        seed: np.full((N_FOLDS, len(test_ids)), np.nan)
        for seed in TRAIN_SEEDS
    }

    fold_rows = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 80)
        print(f"BUILD FOLD-SAFE PCA | FOLD {fold}")
        print("=" * 80)

        outer_train_idx = np.where(folds != fold)[0]

        X_train_fold, X_test_fold, diagnostics = (
            c4a.build_fold_specific_c4a_features(
                base_train=base_train,
                base_test=base_test,
                train_arrays=train_arrays,
                test_arrays=test_arrays,
                outer_train_idx=outer_train_idx,
                fold=fold,
            )
        )

        for d in diagnostics:
            print(
                f"{d['representation']:9s} | "
                f"30-PC explained={d['explained_variance_sum']:.4f}"
            )

        for seed in TRAIN_SEEDS:
            print(f"\nC4 | seed={seed} | fold={fold}")
            set_seed(c1e, seed)

            result = c1e.run_fold(
                fold,
                X_train_fold,
                X_test_fold,
                y,
                folds,
                train_angle,
                test_angle
            )

            va = result["outer_valid_idx"]
            seed_oof[seed][va] = result["gated_valid_prob"]
            seed_test[seed][fold] = result["gated_test_prob"]

            fold_rows.append({
                "seed": seed,
                "fold": fold,
                "gated_log_loss": result["gated_loss"],
                "seen_gated_log_loss": result["seen_loss"],
                "unseen_observed_gated_log_loss":
                    result["unseen_observed_loss"],
                "test_like_observed_reweighted_log_loss":
                    result["test_like"]
            })

            print(
                f"gated={result['gated_loss']:.6f} | "
                f"seen={result['seen_loss']:.6f} | "
                f"unseen={result['unseen_observed_loss']:.6f} | "
                f"test-like={result['test_like']:.6f}"
            )

    for seed in TRAIN_SEEDS:
        if np.isnan(seed_oof[seed]).any():
            raise ValueError(f"seed {seed}: incomplete OOF")
        if np.isnan(seed_test[seed]).any():
            raise ValueError(f"seed {seed}: incomplete test")

        poof = clip_prob(seed_oof[seed])
        ptest = clip_prob(seed_test[seed].mean(axis=0))
        run_id = c4_run_id(seed)

        pd.DataFrame({
            "id": train_ids,
            "fold": folds,
            "target": y,
            "prediction": poof,
            "model_name": "lightgbm_c",
            "run_id": run_id
        }).to_csv(oof_path(run_id), index=False)

        pd.DataFrame({
            "id": test_ids,
            "prediction": ptest,
            "model_name": "lightgbm_c",
            "run_id": run_id
        }).to_csv(test_path(run_id), index=False)

    pd.DataFrame(fold_rows).to_csv(FOLD_PATH, index=False)


def main():
    print("=" * 80)
    print("MODEL C — FINAL C4 PCA ROBUSTNESS + MULTI-SEED FREEZE")
    print("=" * 80)
    print("Seeds:", SEEDS)
    print("This is the final Model-C experiment.")

    c1e = load_module(C1E_SCRIPT, "c1e_for_c4b")
    c4a = load_module(C4A_SCRIPT, "c4a_for_c4b")

    train_df, test_df, train_ids, test_ids, y, folds = c1e.load_data()

    train_angle = pd.to_numeric(train_df["inc_angle"], errors="coerce").to_numpy(dtype=np.float64)
    test_angle = pd.to_numeric(test_df["inc_angle"], errors="coerce").to_numpy(dtype=np.float64)

    observed = np.isfinite(train_angle)

    seen, mean_test_seen_rate = build_seen_mask(
        c1e,
        y,
        folds,
        train_angle,
        test_angle
    )

    print("\nChecking C4 seed42 predictions...")
    load_prediction_pair(
        C4_SEED42_RUN_ID,
        train_ids,
        test_ids,
        y,
        folds
    )

    print("Checking C3c seed42/52/62 predictions...")
    for seed in SEEDS:
        load_prediction_pair(
            c3c_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds
        )

    missing = [
        seed
        for seed in TRAIN_SEEDS
        if not (
            oof_path(c4_run_id(seed)).exists()
            and test_path(c4_run_id(seed)).exists()
        )
    ]

    if missing:
        print("\nTraining missing C4 seeds:", missing)
        train_missing_c4_seeds(
            c1e,
            c4a,
            train_df,
            test_df,
            train_ids,
            test_ids,
            y,
            folds,
            train_angle,
            test_angle
        )
    else:
        print("\nC4 seed52/62 already exist; reusing.")

    rows = []
    paired_rows = []
    c4_oof_list = []
    c4_test_list = []

    for seed in SEEDS:
        c3 = load_prediction_pair(
            c3c_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds
        )
        c4 = load_prediction_pair(
            c4_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds
        )

        c4_oof_list.append(c4["oof"])
        c4_test_list.append(c4["test"])

        m3 = metrics(
            y,
            c3["oof"],
            seen,
            observed,
            mean_test_seen_rate
        )
        m4 = metrics(
            y,
            c4["oof"],
            seen,
            observed,
            mean_test_seen_rate
        )

        rows.append({
            "seed": seed,
            "architecture": "C3c_pool10",
            **m3
        })

        rows.append({
            "seed": seed,
            "architecture": "C4_pool10_pca90",
            **m4
        })

        paired_rows.append({
            "seed": seed,
            "delta_oof_c4_minus_c3c":
                m4["overall_oof_log_loss"]
                - m3["overall_oof_log_loss"],
            "delta_seen_c4_minus_c3c":
                m4["seen_group_oof_log_loss"]
                - m3["seen_group_oof_log_loss"],
            "delta_unseen_c4_minus_c3c":
                m4["unseen_observed_oof_log_loss"]
                - m3["unseen_observed_oof_log_loss"],
            "delta_test_like_c4_minus_c3c":
                m4["test_like_observed_reweighted_log_loss"]
                - m3["test_like_observed_reweighted_log_loss"]
        })

    results_df = pd.DataFrame(rows)
    paired_df = pd.DataFrame(paired_rows)

    results_df.to_csv(RESULTS_PATH, index=False)
    paired_df.to_csv(PAIRED_PATH, index=False)

    c4_oof_matrix = np.stack(c4_oof_list, axis=0)
    c4_test_matrix = np.stack(c4_test_list, axis=0)

    final_oof = clip_prob(c4_oof_matrix.mean(axis=0))
    final_test = clip_prob(c4_test_matrix.mean(axis=0))

    final_metrics = metrics(
        y,
        final_oof,
        seen,
        observed,
        mean_test_seen_rate
    )

    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": final_oof,
        "model_name": "lightgbm_c",
        "run_id": FINAL_RUN_ID
    }).to_csv(FINAL_OOF_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction": final_test,
        "model_name": "lightgbm_c",
        "run_id": FINAL_RUN_ID
    }).to_csv(FINAL_TEST_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "is_iceberg": final_test
    }).to_csv(FINAL_SUBMISSION_PATH, index=False)

    better_oof = int((paired_df["delta_oof_c4_minus_c3c"] < 0).sum())
    better_unseen = int((paired_df["delta_unseen_c4_minus_c3c"] < 0).sum())
    better_test_like = int((paired_df["delta_test_like_c4_minus_c3c"] < 0).sum())

    mean_delta_oof = float(paired_df["delta_oof_c4_minus_c3c"].mean())
    mean_delta_unseen = float(paired_df["delta_unseen_c4_minus_c3c"].mean())
    mean_delta_test_like = float(paired_df["delta_test_like_c4_minus_c3c"].mean())

    robust = (
        better_oof >= 2
        and better_unseen >= 2
        and better_test_like >= 2
        and mean_delta_oof < 0
        and mean_delta_unseen < 0
        and mean_delta_test_like < 0
    )

    if robust:
        decision = "FREEZE_C4B_AS_FINAL_MODEL_C_AND_MOVE_ON"
    else:
        decision = "FREEZE_C3E_AS_FINAL_MODEL_C_AND_MOVE_ON"

    pd.DataFrame([{
        "run_id": FINAL_RUN_ID,
        "final_oof_log_loss":
            final_metrics["overall_oof_log_loss"],
        "final_seen_log_loss":
            final_metrics["seen_group_oof_log_loss"],
        "final_unseen_log_loss":
            final_metrics["unseen_observed_oof_log_loss"],
        "final_test_like_log_loss":
            final_metrics["test_like_observed_reweighted_log_loss"],
        "c4_better_oof_count": better_oof,
        "c4_better_unseen_count": better_unseen,
        "c4_better_test_like_count": better_test_like,
        "mean_delta_oof_c4_minus_c3c": mean_delta_oof,
        "mean_delta_unseen_c4_minus_c3c": mean_delta_unseen,
        "mean_delta_test_like_c4_minus_c3c":
            mean_delta_test_like,
        "decision": decision
    }]).to_csv(SUMMARY_PATH, index=False)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "architecture":
                    "C3c pool10 + fold-safe PCA90",
                "seeds": SEEDS,
                "ensemble":
                    "equal-weight probability mean",
                "decision": decision,
                "note":
                    "Terminal Model-C experiment; move on afterwards."
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("PAIRED ROBUSTNESS: C4 - C3c")
    print("Negative = C4 better")
    print("=" * 80)
    print(paired_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("C4 MULTI-SEED FINAL CANDIDATE")
    print("=" * 80)
    print(f"OOF: {final_metrics['overall_oof_log_loss']:.6f}")
    print(f"Seen: {final_metrics['seen_group_oof_log_loss']:.6f}")
    print(f"Unseen: {final_metrics['unseen_observed_oof_log_loss']:.6f}")
    print(f"Test-like: {final_metrics['test_like_observed_reweighted_log_loss']:.6f}")

    print("\nRobustness counts:")
    print(f"OOF better       : {better_oof}/3")
    print(f"Unseen better    : {better_unseen}/3")
    print(f"Test-like better : {better_test_like}/3")

    print("\nMean paired deltas:")
    print(f"OOF       : {mean_delta_oof:+.6f}")
    print(f"Unseen    : {mean_delta_unseen:+.6f}")
    print(f"Test-like : {mean_delta_test_like:+.6f}")

    print("\n" + "=" * 80)
    print("FINAL MODEL-C FREEZE DECISION")
    print("=" * 80)
    print(decision)

    print("\nSubmission:")
    print(" ", FINAL_SUBMISSION_PATH)

    print(
        "\nAfter this result, stop Model-C experimentation "
        "and move to the next project component."
    )


if __name__ == "__main__":
    main()
