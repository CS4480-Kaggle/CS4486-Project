"""
C4a_lightgbm_pool10_foldsafe_pca.py

FINAL OPTIONAL Model-C architecture experiment.

Question
--------
Can fold-safe PCA add useful global image-shape information on top of the
validated C3c pool10 representation?

Baseline architecture (C3c, seed 42):
    62 richer handcrafted image features
  + 300 pool10 spatial mean features
  = 362 image features

C4a adds:
    30 PCA components from band_1
  + 30 PCA components from band_2
  + 30 PCA components from band_1 - band_2
  = 90 fold-specific PCA features

Total:
    362 + 90 = 452 image features

CRITICAL LEAKAGE RULE
---------------------
PCA is fit separately INSIDE each outer fold:

    PCA.fit(outer_train only)
    PCA.transform(outer_train)
    PCA.transform(outer_valid)
    PCA.transform(test)

Validation and test images are NEVER used to fit PCA.

Everything else is frozen from C1e/C3c:
- same folds
- same 62 handcrafted features
- same 10x10 pooling
- same angle preprocessing
- same strict cross-fit angle prior
- same seen/unseen gate
- same LightGBM parameters
- same seed 42 architecture comparison

STOP RULE
---------
This is intended to be the last new Model-C feature-family experiment.

Promote PCA to a final robustness check ONLY if, vs C3c seed42:
1) overall OOF improves meaningfully (target >= about 0.002),
2) unseen-observed improves,
3) test-like improves,
4) there is no obvious fold collapse.

Otherwise reject PCA and freeze C3e as final Model C.
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

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"

PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "lightgbm_c_c4a_pool10_foldsafe_pca90_seed42"

OUT_OOF_PATH = PRED_DIR / f"oof_{RUN_ID}.csv"
OUT_TEST_PATH = PRED_DIR / f"test_{RUN_ID}.csv"
OUT_SUMMARY_PATH = METRIC_DIR / f"summary_{RUN_ID}.csv"
OUT_FOLDS_PATH = METRIC_DIR / f"folds_{RUN_ID}.csv"
OUT_PCA_PATH = METRIC_DIR / f"pca_diagnostics_{RUN_ID}.csv"
OUT_COMPARISON_PATH = METRIC_DIR / "comparison_c3c_c3e_c4a_pca.csv"
OUT_CONFIG_PATH = METRIC_DIR / f"config_{RUN_ID}.json"

C3C_RUN_ID = "lightgbm_c_c3c_pool10_only_richer_gate_seed42"
C3C_SUMMARY_PATH = METRIC_DIR / f"summary_{C3C_RUN_ID}.csv"

C3E_RUN_ID = "lightgbm_c_c3e_pool10_multiseed_42_52_62"
C3E_SUMMARY_PATH = METRIC_DIR / f"summary_{C3E_RUN_ID}.csv"


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
N_FOLDS = 5
GRID_SIZE = 10
PCA_COMPONENTS = 30
PCA_RANDOM_BASE = 31415
EPS = 1e-7

BASE_C3C_FEATURE_COUNT = 362
EXPECTED_C4A_FEATURE_COUNT = 452


# ============================================================
# 2. HELPERS
# ============================================================

def load_module(path, module_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module from {path}"
        )

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


# ============================================================
# 3. C3c BASE IMAGE REPRESENTATION
# ============================================================

def extract_image_arrays(df):
    """
    Return:
        band1, band2, band_diff
    each shaped [N, 75, 75].
    """

    if not (df["band_1"].map(len) == 5625).all():
        raise ValueError("Invalid band_1 length")

    if not (df["band_2"].map(len) == 5625).all():
        raise ValueError("Invalid band_2 length")

    b1 = np.stack(df["band_1"].to_numpy()).astype(np.float32).reshape(-1, 75, 75)


    b2 = np.stack(df["band_2"].to_numpy()).astype(np.float32).reshape(-1, 75, 75)

    if (
        not np.isfinite(b1).all()
        or not np.isfinite(b2).all()
    ):
        raise ValueError("Non-finite radar values")

    return {
        "band1": b1,
        "band2": b2,
        "band_diff": (b1 - b2).astype(np.float32)
    }


def adaptive_pool_mean(images, grid_size, prefix):
    """
    10x10 adaptive mean pooling for a 75x75 image.

    This transformation is deterministic and needs no fitting.
    """

    images = np.asarray(images, dtype=np.float32)

    if images.ndim != 3 or images.shape[1:] != (75, 75):
        raise ValueError(
            f"Expected [N,75,75], got {images.shape}"
        )

    row_bins = np.array_split(np.arange(75), grid_size)
    col_bins = np.array_split(np.arange(75), grid_size)

    data = {}

    for r, r_idx in enumerate(row_bins):
        for c, c_idx in enumerate(col_bins):
            patch = images[:, r_idx[:, None], c_idx]

            name = (
                f"{prefix}_pool{grid_size}_"
                f"r{r:02d}_c{c:02d}"
            )

            data[name] = patch.mean(axis=(1, 2))

    return pd.DataFrame(data).astype(np.float32)


def build_c3c_base_features(c1e, df, image_arrays):
    """
    Build the already-validated C3c representation:
        62 handcrafted + 300 pool10 = 362.
    """

    X = c1e.build_richer_image_features(df).reset_index(drop=True).copy()
    

    for prefix, images in image_arrays.items():
        pooled = adaptive_pool_mean(
            images,
            GRID_SIZE,
            prefix
        )

        X = pd.concat(
            [
                X,
                pooled.reset_index(drop=True),
            ],
            axis=1
        )

    if X.columns.duplicated().any():
        raise ValueError("Duplicate C3c feature names")

    if X.shape[1] != BASE_C3C_FEATURE_COUNT:
        raise ValueError(f"Expected {BASE_C3C_FEATURE_COUNT} C3c features, got {X.shape[1]}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Invalid C3c features")

    return X.astype(np.float32)


# ============================================================
# 4. FOLD-SAFE PCA
# ============================================================

def fit_transform_pca_one_representation(
    train_images,
    test_images,
    outer_train_idx,
    *,
    fold,
    prefix,
):
    """
    Fit PCA ONLY on outer-training images.

    PCA is applied to raw flattened SAR values.
    sklearn PCA centers each pixel feature automatically.
    We intentionally do NOT standardize each pixel separately.
    """

    train_flat = (
        train_images
        .reshape(len(train_images), -1)
        .astype(np.float32)
    )

    test_flat = (
        test_images
        .reshape(len(test_images), -1)
        .astype(np.float32)
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=PCA_RANDOM_BASE + fold,
    )

    # The critical fold-safe line:
    pca.fit(
        train_flat[outer_train_idx]
    )

    # Transform all train rows using this OUTER-FOLD PCA.
    # Only this fold's outer-valid predictions will be retained.
    train_pc = pca.transform(
        train_flat
    ).astype(np.float32)

    test_pc = pca.transform(
        test_flat
    ).astype(np.float32)

    columns = [
        f"{prefix}_pca_{i + 1:02d}"
        for i in range(PCA_COMPONENTS)
    ]

    train_df = pd.DataFrame(
        train_pc,
        columns=columns,
    )

    test_df = pd.DataFrame(
        test_pc,
        columns=columns,
    )

    explained = pca.explained_variance_ratio_

    diag = {
        "fold": fold,
        "representation": prefix,
        "n_components": PCA_COMPONENTS,
        "explained_variance_sum": float(
            explained.sum()
        ),
        "explained_variance_pc1": float(
            explained[0]
        ),
        "explained_variance_pc10_cumulative": float(
            explained[:10].sum()
        ),
        "explained_variance_pc30_cumulative": float(
            explained[:30].sum()
        ),
    }

    return train_df, test_df, diag


def build_fold_specific_c4a_features(
    base_train,
    base_test,
    train_arrays,
    test_arrays,
    outer_train_idx,
    fold,
):
    """
    For one outer fold:
        C3c 362 fixed features
        + fold-safe PCA 90 features
        = 452.
    """

    X_train = (
        base_train
        .reset_index(drop=True)
        .copy()
    )

    X_test = (
        base_test
        .reset_index(drop=True)
        .copy()
    )

    diagnostics = []

    for prefix in [
        "band1",
        "band2",
        "band_diff",
    ]:
        pca_train, pca_test, diag = (
            fit_transform_pca_one_representation(
                train_images=train_arrays[prefix],
                test_images=test_arrays[prefix],
                outer_train_idx=outer_train_idx,
                fold=fold,
                prefix=prefix,
            )
        )

        X_train = pd.concat(
            [
                X_train,
                pca_train,
            ],
            axis=1,
        )

        X_test = pd.concat(
            [
                X_test,
                pca_test,
            ],
            axis=1,
        )

        diagnostics.append(diag)

    if X_train.columns.duplicated().any():
        raise ValueError(
            f"Fold {fold}: duplicate feature names"
        )

    if X_train.shape[1] != EXPECTED_C4A_FEATURE_COUNT:
        raise ValueError(
            f"Fold {fold}: expected "
            f"{EXPECTED_C4A_FEATURE_COUNT} features, "
            f"got {X_train.shape[1]}"
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            f"Fold {fold}: train/test feature columns differ"
        )

    if (
        not np.isfinite(X_train.to_numpy()).all()
        or not np.isfinite(X_test.to_numpy()).all()
    ):
        raise ValueError(
            f"Fold {fold}: non-finite features"
        )

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        diagnostics,
    )


# ============================================================
# 5. LOAD REFERENCE SUMMARY
# ============================================================

def summary_row_from_file(
    path,
    experiment_name,
    model_kind,
):
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"Expected one summary row in {path}"
        )

    row = df.iloc[0]

    return {
        "experiment": experiment_name,
        "model_kind": model_kind,
        "overall_oof_log_loss": float(
            row["overall_oof_log_loss"]
        ),
        "seen_group_oof_log_loss": float(
            row["seen_group_oof_log_loss"]
        ),
        "unseen_observed_oof_log_loss": float(
            row["unseen_observed_oof_log_loss"]
        ),
        "test_like_observed_reweighted_log_loss": float(
            row["test_like_observed_reweighted_log_loss"]
        ),
    }


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 80)
    print("MODEL C — FINAL OPTIONAL C4a FOLD-SAFE PCA EXPERIMENT")
    print("=" * 80)

    print("Baseline architecture : C3c pool10, seed42")
    print("PCA                   : 30 + 30 + 30 components")
    print("Total image features  : 452")
    print("PCA fitting            : OUTER-TRAIN ONLY")
    print("This is the final planned new feature-family experiment.")

    c1e = load_module(
        C1E_SCRIPT,
        "c1e_for_c4a",
    )

    # Lock same architecture seed as C3c seed42.
    c1e.SEED = SEED
    c1e.LGBM_PARAMS = dict(
        c1e.LGBM_PARAMS
    )
    c1e.LGBM_PARAMS[
        "random_state"
    ] = SEED

    (
        train_df,
        test_df,
        train_ids,
        test_ids,
        y,
        folds,
    ) = c1e.load_data()

    train_angle = pd.to_numeric(
        train_df["inc_angle"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    test_angle = pd.to_numeric(
        test_df["inc_angle"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    print("\nExtracting raw image arrays...")
    train_arrays = extract_image_arrays(
        train_df
    )
    test_arrays = extract_image_arrays(
        test_df
    )

    print("Building fixed C3c 362-feature base...")
    base_train = build_c3c_base_features(
        c1e,
        train_df,
        train_arrays,
    )
    base_test = build_c3c_base_features(
        c1e,
        test_df,
        test_arrays,
    )

    print(
        "Base feature count:",
        base_train.shape[1],
    )

    oof_gated = np.full(
        len(train_ids),
        np.nan,
        dtype=np.float64,
    )

    oof_structural = np.full(
        len(train_ids),
        np.nan,
        dtype=np.float64,
    )

    oof_prior = np.full(
        len(train_ids),
        np.nan,
        dtype=np.float64,
    )

    oof_seen = np.zeros(
        len(train_ids),
        dtype=bool,
    )

    test_matrix = np.full(
        (
            N_FOLDS,
            len(test_ids),
        ),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []
    pca_rows = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 80)
        print(f"OUTER FOLD {fold}")
        print("=" * 80)

        outer_train_idx = np.where(
            folds != fold
        )[0]

        print(
            "Fitting PCA on outer-train rows only:",
            len(outer_train_idx),
        )

        (
            X_train_fold,
            X_test_fold,
            diagnostics,
        ) = build_fold_specific_c4a_features(
            base_train=base_train,
            base_test=base_test,
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            outer_train_idx=outer_train_idx,
            fold=fold,
        )

        pca_rows.extend(
            diagnostics
        )

        for d in diagnostics:
            print(
                f"PCA {d['representation']:9s} | "
                f"30-PC explained variance = "
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

        va_idx = result[
            "outer_valid_idx"
        ]

        oof_gated[
            va_idx
        ] = result[
            "gated_valid_prob"
        ]

        oof_structural[
            va_idx
        ] = result[
            "structural_valid_prob"
        ]

        oof_prior[
            va_idx
        ] = result[
            "prior_valid_prob"
        ]

        oof_seen[
            va_idx
        ] = result[
            "valid_seen"
        ]

        test_matrix[
            fold
        ] = result[
            "gated_test_prob"
        ]

        fold_rows.append({
            "run_id": RUN_ID,
            "fold": fold,
            "n_image_features":
                EXPECTED_C4A_FEATURE_COUNT,
            "structural_log_loss":
                result["structural_loss"],
            "prior_log_loss":
                result["prior_loss"],
            "gated_log_loss":
                result["gated_loss"],
            "seen_gated_log_loss":
                result["seen_loss"],
            "unseen_observed_gated_log_loss":
                result["unseen_observed_loss"],
            "test_seen_rate":
                result["test_seen_rate"],
            "test_like_observed_reweighted_log_loss":
                result["test_like"],
        })

        print(
            f"Gated     : "
            f"{result['gated_loss']:.6f}"
        )
        print(
            f"Seen      : "
            f"{result['seen_loss']:.6f}"
        )
        print(
            f"Unseen    : "
            f"{result['unseen_observed_loss']:.6f}"
        )
        print(
            f"Test-like : "
            f"{result['test_like']:.6f}"
        )

    # --------------------------------------------------------
    # Validate completed predictions
    # --------------------------------------------------------

    for name, arr in [
        ("oof_gated", oof_gated),
        ("oof_structural", oof_structural),
        ("oof_prior", oof_prior),
        ("test_matrix", test_matrix),
    ]:
        if np.isnan(arr).any():
            raise ValueError(
                f"Incomplete predictions: {name}"
            )

    oof_gated = clip_prob(
        oof_gated
    )

    oof_structural = clip_prob(
        oof_structural
    )

    oof_prior = clip_prob(
        oof_prior
    )

    test_pred = clip_prob(
        test_matrix.mean(axis=0)
    )

    observed = np.isfinite(
        train_angle
    )

    seen = oof_seen

    unseen_observed = (
        (~seen)
        & observed
    )

    overall = safe_log_loss(
        y,
        oof_gated,
    )

    structural = safe_log_loss(
        y,
        oof_structural,
    )

    prior = safe_log_loss(
        y,
        oof_prior,
    )

    seen_loss = safe_log_loss(
        y[seen],
        oof_gated[seen],
    )

    unseen_loss = safe_log_loss(
        y[unseen_observed],
        oof_gated[unseen_observed],
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    mean_test_seen_rate = float(
        fold_df[
            "test_seen_rate"
        ].mean()
    )

    test_like = (
        mean_test_seen_rate
        * seen_loss
        + (
            1.0
            - mean_test_seen_rate
        )
        * unseen_loss
    )

    # --------------------------------------------------------
    # Save C4a predictions
    # --------------------------------------------------------

    pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": oof_gated,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    }).to_csv(
        OUT_OOF_PATH,
        index=False,
    )

    pd.DataFrame({
        "id": test_ids,
        "prediction": test_pred,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    }).to_csv(
        OUT_TEST_PATH,
        index=False,
    )

    fold_df.to_csv(
        OUT_FOLDS_PATH,
        index=False,
    )

    pd.DataFrame(
        pca_rows
    ).to_csv(
        OUT_PCA_PATH,
        index=False,
    )

    summary_df = pd.DataFrame([{
        "run_id":
            RUN_ID,
        "experiment":
            "C4a_pool10_foldsafe_pca90",
        "seed":
            SEED,
        "n_image_features":
            EXPECTED_C4A_FEATURE_COUNT,
        "pca_components_per_representation":
            PCA_COMPONENTS,
        "overall_oof_log_loss":
            overall,
        "structural_oof_log_loss":
            structural,
        "prior_oof_log_loss":
            prior,
        "seen_group_oof_log_loss":
            seen_loss,
        "unseen_observed_oof_log_loss":
            unseen_loss,
        "test_like_observed_reweighted_log_loss":
            test_like,
        "mean_fold_log_loss":
            float(
                fold_df[
                    "gated_log_loss"
                ].mean()
            ),
        "std_fold_log_loss":
            float(
                fold_df[
                    "gated_log_loss"
                ].std(ddof=0)
            ),
        "mean_test_seen_rate":
            mean_test_seen_rate,
    }])

    summary_df.to_csv(
        OUT_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Compare C3c / C3e / C4a
    # --------------------------------------------------------

    comparison_rows = []

    c3c_row = summary_row_from_file(
        C3C_SUMMARY_PATH,
        "C3c_pool10_seed42",
        "single_seed",
    )

    c3e_row = summary_row_from_file(
        C3E_SUMMARY_PATH,
        "C3e_pool10_multiseed",
        "3_seed_ensemble",
    )

    if c3c_row is not None:
        comparison_rows.append(
            c3c_row
        )

    if c3e_row is not None:
        comparison_rows.append(
            c3e_row
        )

    comparison_rows.append({
        "experiment":
            "C4a_pool10_foldsafe_pca90_seed42",
        "model_kind":
            "single_seed",
        "overall_oof_log_loss":
            overall,
        "seen_group_oof_log_loss":
            seen_loss,
        "unseen_observed_oof_log_loss":
            unseen_loss,
        "test_like_observed_reweighted_log_loss":
            test_like,
    })

    comparison = pd.DataFrame(
        comparison_rows
    )

    if c3c_row is not None:
        ref_oof = (
            c3c_row[
                "overall_oof_log_loss"
            ]
        )

        ref_unseen = (
            c3c_row[
                "unseen_observed_oof_log_loss"
            ]
        )

        ref_test_like = (
            c3c_row[
                "test_like_observed_reweighted_log_loss"
            ]
        )

        comparison[
            "delta_oof_vs_c3c_seed42"
        ] = (
            comparison[
                "overall_oof_log_loss"
            ]
            - ref_oof
        )

        comparison[
            "delta_unseen_vs_c3c_seed42"
        ] = (
            comparison[
                "unseen_observed_oof_log_loss"
            ]
            - ref_unseen
        )

        comparison[
            "delta_test_like_vs_c3c_seed42"
        ] = (
            comparison[
                "test_like_observed_reweighted_log_loss"
            ]
            - ref_test_like
        )

    comparison.to_csv(
        OUT_COMPARISON_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Explicit stopping decision
    # --------------------------------------------------------

    decision = "NO_C3C_REFERENCE"

    if c3c_row is not None:
        delta_oof = (
            overall
            - c3c_row[
                "overall_oof_log_loss"
            ]
        )

        delta_unseen = (
            unseen_loss
            - c3c_row[
                "unseen_observed_oof_log_loss"
            ]
        )

        delta_test_like = (
            test_like
            - c3c_row[
                "test_like_observed_reweighted_log_loss"
            ]
        )

        meaningful_oof = (
            delta_oof <= -0.002
        )

        unseen_better = (
            delta_unseen < 0
        )

        test_like_better = (
            delta_test_like < 0
        )

        if (
            meaningful_oof
            and unseen_better
            and test_like_better
        ):
            decision = (
                "PCA_PROMISING_RUN_ONE_FINAL_SEED_ROBUSTNESS"
            )
        else:
            decision = (
                "REJECT_PCA_FREEZE_C3E_AND_MOVE_ON"
            )

    config = {
        "run_id":
            RUN_ID,
        "purpose":
            "final optional Model-C feature-family experiment",
        "base":
            "C3c pool10-only",
        "seed":
            SEED,
        "base_features":
            BASE_C3C_FEATURE_COUNT,
        "pca_components_per_representation":
            PCA_COMPONENTS,
        "pca_representations": [
            "band1",
            "band2",
            "band1-band2",
        ],
        "total_image_features":
            EXPECTED_C4A_FEATURE_COUNT,
        "pca_fit_rule":
            "fit on outer_train only for each outer fold",
        "pca_solver":
            "randomized",
        "pca_random_base":
            PCA_RANDOM_BASE,
        "promotion_rule": {
            "minimum_oof_improvement_vs_c3c_seed42":
                0.002,
            "require_unseen_improvement":
                True,
            "require_test_like_improvement":
                True,
        },
        "decision":
            decision,
    }

    OUT_CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("C4a FOLD-SAFE PCA RESULT")
    print("=" * 80)

    print(
        f"OOF       : {overall:.6f}"
    )
    print(
        f"Seen      : {seen_loss:.6f}"
    )
    print(
        f"Unseen    : {unseen_loss:.6f}"
    )
    print(
        f"Test-like : {test_like:.6f}"
    )

    print("\nComparison:")
    print(
        comparison
        .sort_values(
            "overall_oof_log_loss"
        )
        .to_string(
            index=False
        )
    )

    print("\nPer-fold:")
    print(
        fold_df[[
            "fold",
            "gated_log_loss",
            "seen_gated_log_loss",
            "unseen_observed_gated_log_loss",
            "test_like_observed_reweighted_log_loss",
        ]]
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("MODEL-C STOP DECISION")
    print("=" * 80)
    print(decision)

    if decision == (
        "PCA_PROMISING_RUN_ONE_FINAL_SEED_ROBUSTNESS"
    ):
        print(
            "\nC4a passed the pre-defined bar."
        )
        print(
            "Do ONE final seed42/52/62 robustness check, "
            "then freeze Model C."
        )

    elif decision == (
        "REJECT_PCA_FREEZE_C3E_AND_MOVE_ON"
    ):
        print(
            "\nPCA did not clear the pre-defined bar."
        )
        print(
            "Reject C4a, freeze C3e as final Model C, "
            "and move to the next project component."
        )

    print("\nSaved:")
    for path in [
        OUT_OOF_PATH,
        OUT_TEST_PATH,
        OUT_SUMMARY_PATH,
        OUT_FOLDS_PATH,
        OUT_PCA_PATH,
        OUT_COMPARISON_PATH,
        OUT_CONFIG_PATH,
    ]:
        print(" ", path)

    print(
        "\nNo final oof_c.csv/test_c.csv was overwritten."
    )


if __name__ == "__main__":
    main()
