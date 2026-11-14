"""
C5a_lightgbm_c4_plus_joint_pca30.py

Targeted follow-up to C4b.

Hypothesis
----------
C4 showed a large, robust gain from fold-safe PCA applied separately to:
    band1, band2, band1-band2.

Those three PCA blocks capture within-representation global variation, but they
cannot directly learn arbitrary cross-band linear covariance. C5a therefore
keeps the validated C4 representation and adds ONE joint PCA block fitted on:
    concat(flatten(band1), flatten(band2))

Architecture
------------
    C3c base                  362 features
  + C4 separate PCA           90 features (30 x 3)
  + joint band1+band2 PCA     30 features
  =                          482 image features

Leakage rule
------------
Every PCA is fit only on the current outer-training fold. Validation and test
rows are transform-only.

Experiment policy
-----------------
Run seed42 only first. Compare directly with C4a seed42 predictions using the
same folds and metrics. Only if C5a gives a coherent improvement should it be
promoted to a 3-seed robustness run.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss


# ============================================================
# 0. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

C1E_SCRIPT = SCRIPT_DIR / "10_lightgbm_c1e_richer_image_gate.py"
C4A_SCRIPT = SCRIPT_DIR / "C4a_lightgbm_pool10_foldsafe_pca.py"

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
EPS = 1e-7
JOINT_PCA_COMPONENTS = 30
JOINT_PCA_RANDOM_BASE = 27182

RUN_ID = "lightgbm_c_c5a_c4_plus_joint_pca30_seed42"
C4_RUN_ID = "lightgbm_c_c4a_pool10_foldsafe_pca90_seed42"

OUT_OOF_PATH = PRED_DIR / f"oof_{RUN_ID}.csv"
OUT_TEST_PATH = PRED_DIR / f"test_{RUN_ID}.csv"
OUT_SUMMARY_PATH = METRIC_DIR / f"summary_{RUN_ID}.csv"
OUT_FOLDS_PATH = METRIC_DIR / f"folds_{RUN_ID}.csv"
OUT_PCA_PATH = METRIC_DIR / f"pca_diagnostics_{RUN_ID}.csv"
OUT_COMPARE_PATH = METRIC_DIR / "comparison_c4a_vs_c5a_joint_pca30.csv"
OUT_CONFIG_PATH = METRIC_DIR / f"config_{RUN_ID}.json"

C4_OOF_PATH = PRED_DIR / f"oof_{C4_RUN_ID}.csv"
C4_TEST_PATH = PRED_DIR / f"test_{C4_RUN_ID}.csv"

EXPECTED_C4_FEATURES = 452
EXPECTED_C5_FEATURES = 482


# ============================================================
# 1. HELPERS
# ============================================================

def load_module(path, module_name):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
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


def load_reference_oof(path, train_ids, y, folds):
    from prediction_validation import read_csv, validate_oof
    df = validate_oof(read_csv(path), train_ids, y, folds, C4_RUN_ID)
    return clip_prob(df["prediction"].to_numpy(dtype=np.float64))


def build_seen_mask(c1e, y, folds, train_angle, test_angle):
    oof_seen = np.zeros(len(y), dtype=bool)
    test_seen = np.zeros((N_FOLDS, len(test_angle)), dtype=bool)

    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]

        mapping, global_prior = c1e.fit_group_mapping(train_angle[tr], y[tr])

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
# 2. JOINT PCA
# ============================================================

def fit_transform_joint_pca(
    train_arrays,
    test_arrays,
    outer_train_idx,
    fold,
):
    """
    Joint PCA on concatenated raw band1 + band2 pixels.

    Shape before PCA:
        train: [N, 11250]
        test : [M, 11250]

    Fit is strictly outer-train only.
    """

    b1_train = train_arrays["band1"].reshape(len(train_arrays["band1"]), -1)
    b2_train = train_arrays["band2"].reshape(len(train_arrays["band2"]), -1)
    b1_test = test_arrays["band1"].reshape(len(test_arrays["band1"]), -1)
    b2_test = test_arrays["band2"].reshape(len(test_arrays["band2"]), -1)

    train_joint = np.concatenate([b1_train, b2_train], axis=1).astype(np.float32)
    test_joint = np.concatenate([b1_test, b2_test], axis=1).astype(np.float32)

    pca = PCA(
        n_components=JOINT_PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=JOINT_PCA_RANDOM_BASE + fold,
    )

    pca.fit(train_joint[outer_train_idx])

    train_pc = pca.transform(train_joint).astype(np.float32)
    test_pc = pca.transform(test_joint).astype(np.float32)

    columns = [
        f"joint_b1_b2_pca_{i + 1:02d}"
        for i in range(JOINT_PCA_COMPONENTS)
    ]

    train_df = pd.DataFrame(train_pc, columns=columns)
    test_df = pd.DataFrame(test_pc, columns=columns)

    explained = pca.explained_variance_ratio_
    diag = {
        "fold": fold,
        "representation": "joint_band1_band2",
        "n_components": JOINT_PCA_COMPONENTS,
        "explained_variance_sum": float(explained.sum()),
        "explained_variance_pc1": float(explained[0]),
        "explained_variance_pc10_cumulative": float(explained[:10].sum()),
        "explained_variance_pc30_cumulative": float(explained[:30].sum()),
    }

    return train_df, test_df, diag


def build_fold_specific_c5_features(
    c4a,
    base_train,
    base_test,
    train_arrays,
    test_arrays,
    outer_train_idx,
    fold,
):
    # First build the exact validated C4 representation.
    X_train, X_test, diagnostics = c4a.build_fold_specific_c4a_features(
        base_train=base_train,
        base_test=base_test,
        train_arrays=train_arrays,
        test_arrays=test_arrays,
        outer_train_idx=outer_train_idx,
        fold=fold,
    )

    if X_train.shape[1] != EXPECTED_C4_FEATURES:
        raise ValueError(
            f"Fold {fold}: expected {EXPECTED_C4_FEATURES} C4 features, "
            f"got {X_train.shape[1]}"
        )

    joint_train, joint_test, joint_diag = fit_transform_joint_pca(
        train_arrays=train_arrays,
        test_arrays=test_arrays,
        outer_train_idx=outer_train_idx,
        fold=fold,
    )

    X_train = pd.concat(
        [X_train.reset_index(drop=True), joint_train.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat(
        [X_test.reset_index(drop=True), joint_test.reset_index(drop=True)],
        axis=1,
    )

    diagnostics = list(diagnostics) + [joint_diag]

    if X_train.columns.duplicated().any():
        raise ValueError(f"Fold {fold}: duplicate feature names")
    if X_train.shape[1] != EXPECTED_C5_FEATURES:
        raise ValueError(
            f"Fold {fold}: expected {EXPECTED_C5_FEATURES} features, "
            f"got {X_train.shape[1]}"
        )
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"Fold {fold}: train/test columns differ")
    if (
        not np.isfinite(X_train.to_numpy()).all()
        or not np.isfinite(X_test.to_numpy()).all()
    ):
        raise ValueError(f"Fold {fold}: non-finite features")

    return X_train.astype(np.float32), X_test.astype(np.float32), diagnostics


# ============================================================
# 3. MAIN
# ============================================================

def main():
    print("=" * 80)
    print("MODEL C — C5a: C4 + FOLD-SAFE JOINT BAND1/BAND2 PCA30")
    print("=" * 80)
    print("Seed                  :", SEED)
    print("C4 base features      :", EXPECTED_C4_FEATURES)
    print("Joint PCA components  :", JOINT_PCA_COMPONENTS)
    print("Total image features  :", EXPECTED_C5_FEATURES)
    print("PCA fitting           : OUTER-TRAIN ONLY")

    c1e = load_module(C1E_SCRIPT, "c1e_for_c5a")
    c4a = load_module(C4A_SCRIPT, "c4a_for_c5a")

    c1e.SEED = SEED
    c1e.LGBM_PARAMS = dict(c1e.LGBM_PARAMS)
    c1e.LGBM_PARAMS["random_state"] = SEED

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

    print("\nLoading C4 seed42 reference...")
    c4_oof = load_reference_oof(C4_OOF_PATH, train_ids, y, folds)
    c4_metrics = metrics(
        y, c4_oof, seen, observed, mean_test_seen_rate
    )

    print("Extracting image arrays...")
    train_arrays = c4a.extract_image_arrays(train_df)
    test_arrays = c4a.extract_image_arrays(test_df)

    print("Building fixed C3c 362-feature base...")
    base_train = c4a.build_c3c_base_features(
        c1e, train_df, train_arrays
    )
    base_test = c4a.build_c3c_base_features(
        c1e, test_df, test_arrays
    )

    oof = np.full(len(train_ids), np.nan, dtype=np.float64)
    test_matrix = np.full((N_FOLDS, len(test_ids)), np.nan, dtype=np.float64)

    fold_rows = []
    pca_rows = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 80)
        print(f"C5a | OUTER FOLD {fold}")
        print("=" * 80)

        outer_train_idx = np.where(folds != fold)[0]

        X_train_fold, X_test_fold, diagnostics = build_fold_specific_c5_features(
            c4a=c4a,
            base_train=base_train,
            base_test=base_test,
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            outer_train_idx=outer_train_idx,
            fold=fold,
        )

        pca_rows.extend(diagnostics)

        for d in diagnostics:
            print(
                f"{d['representation']:18s} | "
                f"{d['n_components']:2d}-PC explained="
                f"{d['explained_variance_sum']:.4f}"
            )

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
        oof[va] = result["gated_valid_prob"]
        test_matrix[fold] = result["gated_test_prob"]

        fold_rows.append({
            "run_id": RUN_ID,
            "fold": fold,
            "n_image_features": EXPECTED_C5_FEATURES,
            "gated_log_loss": result["gated_loss"],
            "seen_gated_log_loss": result["seen_loss"],
            "unseen_observed_gated_log_loss": result["unseen_observed_loss"],
            "test_seen_rate": result["test_seen_rate"],
            "test_like_observed_reweighted_log_loss": result["test_like"],
        })

        print(
            f"gated={result['gated_loss']:.6f} | "
            f"seen={result['seen_loss']:.6f} | "
            f"unseen={result['unseen_observed_loss']:.6f} | "
            f"test-like={result['test_like']:.6f}"
        )

    if np.isnan(oof).any():
        raise ValueError("Incomplete C5a OOF")
    if np.isnan(test_matrix).any():
        raise ValueError("Incomplete C5a test predictions")

    oof = clip_prob(oof)
    test_pred = clip_prob(test_matrix.mean(axis=0))

    c5_metrics = metrics(
        y, oof, seen, observed, mean_test_seen_rate
    )

    # Save predictions
    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": oof,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    }).to_csv(OUT_OOF_PATH, index=False)

    pd.DataFrame({
        "id": test_ids,
        "prediction": test_pred,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    }).to_csv(OUT_TEST_PATH, index=False)

    pd.DataFrame(fold_rows).to_csv(OUT_FOLDS_PATH, index=False)
    pd.DataFrame(pca_rows).to_csv(OUT_PCA_PATH, index=False)

    delta = {
        key: c5_metrics[key] - c4_metrics[key]
        for key in c4_metrics
    }

    compare_df = pd.DataFrame([
        {"experiment": "C4a_seed42", **c4_metrics},
        {"experiment": "C5a_joint_pca30_seed42", **c5_metrics},
    ])
    compare_df.to_csv(OUT_COMPARE_PATH, index=False)

    promote = (
        delta["overall_oof_log_loss"] <= -0.002
        and delta["unseen_observed_oof_log_loss"] < 0
        and delta["test_like_observed_reweighted_log_loss"] < 0
    )

    decision = (
        "PROMOTE_C5A_TO_3SEED_ROBUSTNESS"
        if promote
        else "REJECT_C5A_KEEP_C4B"
    )

    pd.DataFrame([{
        "run_id": RUN_ID,
        **c5_metrics,
        "delta_oof_vs_c4": delta["overall_oof_log_loss"],
        "delta_seen_vs_c4": delta["seen_group_oof_log_loss"],
        "delta_unseen_vs_c4": delta["unseen_observed_oof_log_loss"],
        "delta_test_like_vs_c4": delta["test_like_observed_reweighted_log_loss"],
        "decision": decision,
    }]).to_csv(OUT_SUMMARY_PATH, index=False)

    OUT_CONFIG_PATH.write_text(
        json.dumps(
            {
                "architecture": (
                    "C4 (pool10 + separate PCA90) + "
                    "fold-safe joint band1/band2 PCA30"
                ),
                "seed": SEED,
                "joint_pca_components": JOINT_PCA_COMPONENTS,
                "expected_image_features": EXPECTED_C5_FEATURES,
                "reference": C4_RUN_ID,
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("C4a seed42 vs C5a seed42")
    print("Negative delta = C5a better")
    print("=" * 80)

    print(f"C4 OOF       : {c4_metrics['overall_oof_log_loss']:.6f}")
    print(f"C5 OOF       : {c5_metrics['overall_oof_log_loss']:.6f}")
    print(f"delta OOF    : {delta['overall_oof_log_loss']:+.6f}")
    print()
    print(f"C4 Seen      : {c4_metrics['seen_group_oof_log_loss']:.6f}")
    print(f"C5 Seen      : {c5_metrics['seen_group_oof_log_loss']:.6f}")
    print(f"delta Seen   : {delta['seen_group_oof_log_loss']:+.6f}")
    print()
    print(f"C4 Unseen    : {c4_metrics['unseen_observed_oof_log_loss']:.6f}")
    print(f"C5 Unseen    : {c5_metrics['unseen_observed_oof_log_loss']:.6f}")
    print(f"delta Unseen : {delta['unseen_observed_oof_log_loss']:+.6f}")
    print()
    print(f"C4 Test-like : {c4_metrics['test_like_observed_reweighted_log_loss']:.6f}")
    print(f"C5 Test-like : {c5_metrics['test_like_observed_reweighted_log_loss']:.6f}")
    print(f"delta T-like : {delta['test_like_observed_reweighted_log_loss']:+.6f}")

    print("\nDecision:")
    print(decision)

    print("\nOutputs:")
    print(" ", OUT_OOF_PATH)
    print(" ", OUT_TEST_PATH)
    print(" ", OUT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
