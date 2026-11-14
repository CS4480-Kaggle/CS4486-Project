"""
C5b_lightgbm_joint_pca_multiseed.py

3-seed robustness + ensemble for C5.

C5 architecture
---------------
    C3c base                         362 features
  + C4 separate fold-safe PCA         90 features
      - band1 PCA30
      - band2 PCA30
      - band_diff PCA30
  + C5 joint fold-safe PCA            30 features
      - concat(flatten(band1), flatten(band2)) PCA30
  =                                  482 image features

Seeds
-----
42, 52, 62

Seed 42 C5 predictions are reused from C5a.
Seeds 52/62 are trained only if their prediction files are missing.
C4 seed42/52/62 predictions are reused as the paired reference.

Outputs
-------
All predictions/submission are written under:
    artifacts/predictions/

Metrics/configs are written under:
    artifacts/metrics/
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


# ============================================================
# 0. PATHS / CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

C1E_SCRIPT = SCRIPT_DIR / "10_lightgbm_c1e_richer_image_gate.py"
C5A_SCRIPT = SCRIPT_DIR / "C5a_lightgbm_c4_plus_joint_pca30.py"

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 52, 62]
TRAIN_SEEDS = [52, 62]
N_FOLDS = 5
EPS = 1e-7
EXPECTED_C5_FEATURES = 482

C5_SEED42_RUN_ID = "lightgbm_c_c5a_c4_plus_joint_pca30_seed42"
FINAL_RUN_ID = "lightgbm_c_c5b_c4_plus_joint_pca30_multiseed_42_52_62"

FINAL_OOF_PATH = PRED_DIR / f"oof_{FINAL_RUN_ID}.csv"
FINAL_TEST_PATH = PRED_DIR / f"test_{FINAL_RUN_ID}.csv"
FINAL_SUBMISSION_PATH = PRED_DIR / "submission_lightgbm_c5b_joint_pca30_multiseed.csv"

RESULTS_PATH = METRIC_DIR / "seed_robustness_c4_vs_c5_joint_pca30.csv"
PAIRED_PATH = METRIC_DIR / "seed_robustness_c4_vs_c5_joint_pca30_paired.csv"
SUMMARY_PATH = METRIC_DIR / f"summary_{FINAL_RUN_ID}.csv"
FOLD_PATH = METRIC_DIR / "seed_robustness_c5_joint_pca30_by_fold.csv"
CONFIG_PATH = METRIC_DIR / f"config_{FINAL_RUN_ID}.json"


# ============================================================
# 1. GENERIC HELPERS
# ============================================================

def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def safe_log_loss(y_true, pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(y_true) == 0:
        return np.nan
    return float(log_loss(y_true, clip_prob(pred), labels=[0, 1]))


def set_seed(c1e, seed):
    c1e.SEED = int(seed)
    c1e.LGBM_PARAMS = dict(c1e.LGBM_PARAMS)
    c1e.LGBM_PARAMS["random_state"] = int(seed)


def c4_run_id(seed):
    return f"lightgbm_c_c4a_pool10_foldsafe_pca90_seed{seed}"


def c5_run_id(seed):
    if seed == 42:
        return C5_SEED42_RUN_ID
    return f"lightgbm_c_c5a_c4_plus_joint_pca30_seed{seed}"


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


# ============================================================
# 2. SEEN / UNSEEN EVALUATION MASK
# ============================================================

def build_seen_mask(c1e, y, folds, train_angle, test_angle):
    """
    Rebuild the exact fold-safe seen/unseen definition used by C1e/C4/C5.

    Important: this is evaluation/gating structure, not the old Group1
    high-purity postprocess.
    """
    oof_seen = np.zeros(len(y), dtype=bool)
    test_seen = np.zeros((N_FOLDS, len(test_angle)), dtype=bool)

    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]

        mapping, global_prior = c1e.fit_group_mapping(
            train_angle[tr], y[tr]
        )

        vg = c1e.encode_group_features(
            train_angle[va], mapping, global_prior
        )
        tg = c1e.encode_group_features(
            test_angle, mapping, global_prior
        )

        oof_seen[va] = (
            vg["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
        )
        test_seen[fold] = (
            tg["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
        )

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


# ============================================================
# 3. TRAIN MISSING C5 SEEDS (52 / 62)
# ============================================================

def train_missing_c5_seeds(
    c1e,
    c5a,
    train_df,
    test_df,
    train_ids,
    test_ids,
    y,
    folds,
    train_angle,
    test_angle,
    seeds_to_train,
):
    if not seeds_to_train:
        return

    print("\nExtracting image arrays once...")
    train_arrays = c5a.load_module(
        c5a.C4A_SCRIPT, "c4a_inside_c5b"
    ).extract_image_arrays(train_df)
    test_arrays = c5a.load_module(
        c5a.C4A_SCRIPT, "c4a_inside_c5b_test"
    ).extract_image_arrays(test_df)

    # Use a single C4a module instance for all feature construction.
    c4a = load_module(c5a.C4A_SCRIPT, "c4a_for_c5b")

    print("Building fixed C3c 362-feature base once...")
    base_train = c4a.build_c3c_base_features(
        c1e, train_df, train_arrays
    )
    base_test = c4a.build_c3c_base_features(
        c1e, test_df, test_arrays
    )

    seed_oof = {
        seed: np.full(len(train_ids), np.nan, dtype=np.float64)
        for seed in seeds_to_train
    }
    seed_test = {
        seed: np.full(
            (N_FOLDS, len(test_ids)), np.nan, dtype=np.float64
        )
        for seed in seeds_to_train
    }

    fold_rows = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 80)
        print(f"C5b BUILD FEATURES | OUTER FOLD {fold}")
        print("=" * 80)

        outer_train_idx = np.where(folds != fold)[0]

        X_train_fold, X_test_fold, diagnostics = (
            c5a.build_fold_specific_c5_features(
                c4a=c4a,
                base_train=base_train,
                base_test=base_test,
                train_arrays=train_arrays,
                test_arrays=test_arrays,
                outer_train_idx=outer_train_idx,
                fold=fold,
            )
        )

        if X_train_fold.shape[1] != EXPECTED_C5_FEATURES:
            raise ValueError(
                f"Fold {fold}: expected {EXPECTED_C5_FEATURES} features, "
                f"got {X_train_fold.shape[1]}"
            )

        for d in diagnostics:
            print(
                f"{d['representation']:18s} | "
                f"{d['n_components']:2d}-PC explained="
                f"{d['explained_variance_sum']:.4f}"
            )

        for seed in seeds_to_train:
            print(f"\nC5 | seed={seed} | fold={fold}")
            set_seed(c1e, seed)

            result = c1e.run_fold(
                fold,
                X_train_fold,
                X_test_fold,
                y,
                folds,
                train_angle,
                test_angle,
            )

            va = result["outer_valid_idx"]
            seed_oof[seed][va] = result["gated_valid_prob"]
            seed_test[seed][fold] = result["gated_test_prob"]

            fold_rows.append({
                "seed": seed,
                "fold": fold,
                "n_image_features": EXPECTED_C5_FEATURES,
                "gated_log_loss": result["gated_loss"],
                "seen_gated_log_loss": result["seen_loss"],
                "unseen_observed_gated_log_loss":
                    result["unseen_observed_loss"],
                "test_seen_rate": result["test_seen_rate"],
                "test_like_observed_reweighted_log_loss":
                    result["test_like"],
            })

            print(
                f"gated={result['gated_loss']:.6f} | "
                f"seen={result['seen_loss']:.6f} | "
                f"unseen={result['unseen_observed_loss']:.6f} | "
                f"test-like={result['test_like']:.6f}"
            )

    for seed in seeds_to_train:
        if np.isnan(seed_oof[seed]).any():
            raise ValueError(f"seed {seed}: incomplete C5 OOF")
        if np.isnan(seed_test[seed]).any():
            raise ValueError(f"seed {seed}: incomplete C5 test matrix")

        poof = clip_prob(seed_oof[seed])
        ptest = clip_prob(seed_test[seed].mean(axis=0))
        run_id = c5_run_id(seed)

        pd.DataFrame({
            "id": train_ids,
            "fold": folds,
            "target": y,
            "prediction": poof,
            "model_name": "lightgbm_c",
            "run_id": run_id,
        }).to_csv(oof_path(run_id), index=False)

        pd.DataFrame({
            "id": test_ids,
            "prediction": ptest,
            "model_name": "lightgbm_c",
            "run_id": run_id,
        }).to_csv(test_path(run_id), index=False)

    new_fold_df = pd.DataFrame(fold_rows)
    if FOLD_PATH.exists():
        old = pd.read_csv(FOLD_PATH)
        old = old[~old["seed"].isin(seeds_to_train)]
        new_fold_df = pd.concat([old, new_fold_df], ignore_index=True)

    new_fold_df.sort_values(["seed", "fold"]).to_csv(
        FOLD_PATH, index=False
    )


# ============================================================
# 4. MAIN ROBUSTNESS + ENSEMBLE
# ============================================================

def main():
    print("=" * 80)
    print("MODEL C — C5b JOINT-PCA30 3-SEED ROBUSTNESS + ENSEMBLE")
    print("=" * 80)
    print("Seeds                 :", SEEDS)
    print("C5 image features     :", EXPECTED_C5_FEATURES)
    print("Reference             : C4 pool10 + separate PCA90")
    print("New block             : joint band1/band2 PCA30")
    print("Prediction directory  :", PRED_DIR)

    c1e = load_module(C1E_SCRIPT, "c1e_for_c5b")
    c5a = load_module(C5A_SCRIPT, "c5a_for_c5b")

    train_df, test_df, train_ids, test_ids, y, folds = c1e.load_data()

    train_angle = pd.to_numeric(
        train_df["inc_angle"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    test_angle = pd.to_numeric(
        test_df["inc_angle"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    observed = np.isfinite(train_angle)
    seen, mean_test_seen_rate = build_seen_mask(
        c1e, y, folds, train_angle, test_angle
    )

    # --------------------------------------------------------
    # Verify C4 references for all seeds.
    # --------------------------------------------------------
    print("\nChecking C4 seed42/52/62 references...")
    for seed in SEEDS:
        load_prediction_pair(
            c4_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds,
        )

    # --------------------------------------------------------
    # Reuse C5 seed42; train only missing seed52/62.
    # --------------------------------------------------------
    print("Checking C5 seed42 reference from C5a...")
    load_prediction_pair(
        C5_SEED42_RUN_ID,
        train_ids,
        test_ids,
        y,
        folds,
    )

    missing = [
        seed
        for seed in TRAIN_SEEDS
        if not (
            oof_path(c5_run_id(seed)).exists()
            and test_path(c5_run_id(seed)).exists()
        )
    ]

    if missing:
        print("\nTraining missing C5 seeds:", missing)
        train_missing_c5_seeds(
            c1e=c1e,
            c5a=c5a,
            train_df=train_df,
            test_df=test_df,
            train_ids=train_ids,
            test_ids=test_ids,
            y=y,
            folds=folds,
            train_angle=train_angle,
            test_angle=test_angle,
            seeds_to_train=missing,
        )
    else:
        print("\nC5 seed52/62 predictions already exist; reusing.")

    # --------------------------------------------------------
    # Paired C4 vs C5 metrics per seed.
    # --------------------------------------------------------
    rows = []
    paired_rows = []
    c5_oof_list = []
    c5_test_list = []

    for seed in SEEDS:
        c4 = load_prediction_pair(
            c4_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds,
        )
        c5 = load_prediction_pair(
            c5_run_id(seed),
            train_ids,
            test_ids,
            y,
            folds,
        )

        c5_oof_list.append(c5["oof"])
        c5_test_list.append(c5["test"])

        m4 = metrics(
            y, c4["oof"], seen, observed, mean_test_seen_rate
        )
        m5 = metrics(
            y, c5["oof"], seen, observed, mean_test_seen_rate
        )

        rows.append({
            "seed": seed,
            "architecture": "C4_pool10_pca90",
            **m4,
        })
        rows.append({
            "seed": seed,
            "architecture": "C5_c4_plus_joint_pca30",
            **m5,
        })

        paired_rows.append({
            "seed": seed,
            "delta_oof_c5_minus_c4":
                m5["overall_oof_log_loss"]
                - m4["overall_oof_log_loss"],
            "delta_seen_c5_minus_c4":
                m5["seen_group_oof_log_loss"]
                - m4["seen_group_oof_log_loss"],
            "delta_unseen_c5_minus_c4":
                m5["unseen_observed_oof_log_loss"]
                - m4["unseen_observed_oof_log_loss"],
            "delta_test_like_c5_minus_c4":
                m5["test_like_observed_reweighted_log_loss"]
                - m4["test_like_observed_reweighted_log_loss"],
        })

    results_df = pd.DataFrame(rows)
    paired_df = pd.DataFrame(paired_rows)

    results_df.to_csv(RESULTS_PATH, index=False)
    paired_df.to_csv(PAIRED_PATH, index=False)

    # --------------------------------------------------------
    # Equal-weight 3-seed C5 ensemble.
    # --------------------------------------------------------
    c5_oof_matrix = np.stack(c5_oof_list, axis=0)
    c5_test_matrix = np.stack(c5_test_list, axis=0)

    final_oof = clip_prob(c5_oof_matrix.mean(axis=0))
    final_test = clip_prob(c5_test_matrix.mean(axis=0))

    final_metrics = metrics(
        y, final_oof, seen, observed, mean_test_seen_rate
    )

    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": final_oof,
        "model_name": "lightgbm_c",
        "run_id": FINAL_RUN_ID,
    }).to_csv(FINAL_OOF_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction": final_test,
        "model_name": "lightgbm_c",
        "run_id": FINAL_RUN_ID,
    }).to_csv(FINAL_TEST_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "is_iceberg": final_test,
    }).to_csv(FINAL_SUBMISSION_PATH, index=False)

    # --------------------------------------------------------
    # Robustness summary.
    # --------------------------------------------------------
    better_oof = int((paired_df["delta_oof_c5_minus_c4"] < 0).sum())
    better_unseen = int(
        (paired_df["delta_unseen_c5_minus_c4"] < 0).sum()
    )
    better_test_like = int(
        (paired_df["delta_test_like_c5_minus_c4"] < 0).sum()
    )

    mean_delta_oof = float(
        paired_df["delta_oof_c5_minus_c4"].mean()
    )
    mean_delta_seen = float(
        paired_df["delta_seen_c5_minus_c4"].mean()
    )
    mean_delta_unseen = float(
        paired_df["delta_unseen_c5_minus_c4"].mean()
    )
    mean_delta_test_like = float(
        paired_df["delta_test_like_c5_minus_c4"].mean()
    )

    # Same spirit as previous robustness rule: at least 2/3 seed wins
    # in the important metrics and negative mean paired deltas.
    robust = (
        better_oof >= 2
        and better_unseen >= 2
        and better_test_like >= 2
        and mean_delta_oof < 0
        and mean_delta_unseen < 0
        and mean_delta_test_like < 0
    )

    decision = (
        "PROMOTE_C5B_OVER_C4B"
        if robust
        else "KEEP_C4B_REJECT_C5B"
    )

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
        "c5_better_oof_count": better_oof,
        "c5_better_unseen_count": better_unseen,
        "c5_better_test_like_count": better_test_like,
        "mean_delta_oof_c5_minus_c4": mean_delta_oof,
        "mean_delta_seen_c5_minus_c4": mean_delta_seen,
        "mean_delta_unseen_c5_minus_c4": mean_delta_unseen,
        "mean_delta_test_like_c5_minus_c4": mean_delta_test_like,
        "decision": decision,
    }]).to_csv(SUMMARY_PATH, index=False)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "architecture": (
                    "C4 pool10 + separate fold-safe PCA90 + "
                    "joint band1/band2 fold-safe PCA30"
                ),
                "seeds": SEEDS,
                "ensemble": "equal-weight probability mean",
                "n_image_features": EXPECTED_C5_FEATURES,
                "reference_architecture": "C4 pool10 + PCA90",
                "prediction_directory": str(PRED_DIR),
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console report.
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("PAIRED ROBUSTNESS: C5 - C4")
    print("Negative = C5 better")
    print("=" * 80)
    print(paired_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("C5 MULTI-SEED CANDIDATE")
    print("=" * 80)
    print(
        f"OOF       : "
        f"{final_metrics['overall_oof_log_loss']:.6f}"
    )
    print(
        f"Seen      : "
        f"{final_metrics['seen_group_oof_log_loss']:.6f}"
    )
    print(
        f"Unseen    : "
        f"{final_metrics['unseen_observed_oof_log_loss']:.6f}"
    )
    print(
        f"Test-like : "
        f"{final_metrics['test_like_observed_reweighted_log_loss']:.6f}"
    )

    print("\nRobustness counts:")
    print(f"OOF better       : {better_oof}/3")
    print(f"Unseen better    : {better_unseen}/3")
    print(f"Test-like better : {better_test_like}/3")

    print("\nMean paired deltas (C5 - C4):")
    print(f"OOF       : {mean_delta_oof:+.6f}")
    print(f"Seen      : {mean_delta_seen:+.6f}")
    print(f"Unseen    : {mean_delta_unseen:+.6f}")
    print(f"Test-like : {mean_delta_test_like:+.6f}")

    print("\n" + "=" * 80)
    print("C5B DECISION")
    print("=" * 80)
    print(decision)

    print("\nOutputs:")
    print("  OOF       :", FINAL_OOF_PATH)
    print("  Test      :", FINAL_TEST_PATH)
    print("  Submission:", FINAL_SUBMISSION_PATH)
    print("  Summary   :", SUMMARY_PATH)


if __name__ == "__main__":
    main()
