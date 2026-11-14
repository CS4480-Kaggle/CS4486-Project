"""
C3c_lightgbm_pool10_only.py

Model C — LightGBM pool10-only spatial feature ablation.

Purpose
-------
Test whether the improvement seen in C3b (5x5 + 10x10 pooling)
is mainly caused by the 10x10 spatial grid.

This experiment keeps ALL C1e modeling logic unchanged:
- same shared folds
- same angle preprocessing
- same strict cross-fit angle prior
- same seen/unseen gate
- same LightGBM hyperparameters
- same 62 handcrafted image features

ONLY CHANGE:
+ 10x10 pooled means for band_1, band_2, band_1-band_2

Feature count:
62 + 100*3 = 362 image features
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

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
GRID_SIZE = 10
EPS = 1e-7

C1E_RUN_ID = "lightgbm_c_c1e_richer_image_seen_unseen_gate_seed42"
C1E_SUMMARY_PATH = METRIC_DIR / f"summary_{C1E_RUN_ID}.csv"

C3B_RUN_ID = "lightgbm_c_c3b_pool5_10_richer_gate_seed42"
C3B_SUMMARY_PATH = METRIC_DIR / f"summary_{C3B_RUN_ID}.csv"

RUN_ID = "lightgbm_c_c3c_pool10_only_richer_gate_seed42"

OUT_OOF_PATH = PRED_DIR / f"oof_{RUN_ID}.csv"
OUT_TEST_PATH = PRED_DIR / f"test_{RUN_ID}.csv"
OUT_SUMMARY_PATH = METRIC_DIR / f"summary_{RUN_ID}.csv"
OUT_FOLDS_PATH = METRIC_DIR / f"folds_{RUN_ID}.csv"
OUT_COMPARISON_PATH = METRIC_DIR / "comparison_lightgbm_c1e_c3b_c3c_pool10.csv"
OUT_CONFIG_PATH = METRIC_DIR / f"config_{RUN_ID}.json"


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
    return float(
        log_loss(
            np.asarray(y_true, dtype=np.int64),
            clip_prob(pred),
            labels=[0, 1],
        )
    )


def extract_image_arrays(df):
    if not (df["band_1"].map(len) == 5625).all():
        raise ValueError("Invalid band_1 length")
    if not (df["band_2"].map(len) == 5625).all():
        raise ValueError("Invalid band_2 length")

    b1 = np.stack(df["band_1"].to_numpy()).astype(np.float32).reshape(-1, 75, 75)
    b2 = np.stack(df["band_2"].to_numpy()).astype(np.float32).reshape(-1, 75, 75)

    if not np.isfinite(b1).all() or not np.isfinite(b2).all():
        raise ValueError("Non-finite radar values")

    return {
        "band1": b1,
        "band2": b2,
        "band_diff": (b1 - b2).astype(np.float32),
    }


def adaptive_pool_mean(images, grid_size, prefix):
    images = np.asarray(images, dtype=np.float32)

    if images.ndim != 3 or images.shape[1:] != (75, 75):
        raise ValueError(f"Expected [N,75,75], got {images.shape}")

    row_bins = np.array_split(np.arange(75), grid_size)
    col_bins = np.array_split(np.arange(75), grid_size)

    data = {}

    for r, r_idx in enumerate(row_bins):
        for c, c_idx in enumerate(col_bins):
            patch = images[:, r_idx[:, None], c_idx]
            feature_name = f"{prefix}_pool{grid_size}_r{r:02d}_c{c:02d}"
            data[feature_name] = patch.mean(axis=(1, 2))

    out = pd.DataFrame(data).astype(np.float32)

    if out.shape[1] != grid_size ** 2:
        raise ValueError("Unexpected number of pooled features")

    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite pooled features")

    return out


def build_pool10_features(base_features, image_arrays):
    X = base_features.reset_index(drop=True).copy()

    for prefix, images in image_arrays.items():
        pooled = adaptive_pool_mean(
            images=images,
            grid_size=GRID_SIZE,
            prefix=prefix,
        )
        X = pd.concat([X, pooled.reset_index(drop=True)], axis=1)

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature names")

    if X.shape[1] != 362:
        raise ValueError(f"Expected 362 image features, got {X.shape[1]}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Invalid combined image features")

    return X.astype(np.float32)


def main():
    print("=" * 78)
    print("MODEL C — C3c POOL10-ONLY SPATIAL ABLATION")
    print("=" * 78)
    print("Reference: C1e")
    print("Only change: +10x10 pooled spatial means")
    print("Expected image feature count: 362")
    print("Angle logic / gate / folds / LightGBM params unchanged.")

    c1e = load_module(C1E_SCRIPT, "c1e_pool10_module")

    c1e.SEED = SEED
    c1e.LGBM_PARAMS = dict(c1e.LGBM_PARAMS)
    c1e.LGBM_PARAMS["random_state"] = SEED

    train_df, test_df, train_ids, test_ids, y, folds = c1e.load_data()

    train_angle = pd.to_numeric(
        train_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    test_angle = pd.to_numeric(
        test_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    print("\nBuilding original 62 C1e image features...")
    base_train = c1e.build_richer_image_features(train_df)
    base_test = c1e.build_richer_image_features(test_df)

    print("Extracting 75x75 image arrays...")
    train_arrays = extract_image_arrays(train_df)
    test_arrays = extract_image_arrays(test_df)

    print("Building pool10-only feature table...")
    X_train = build_pool10_features(base_train, train_arrays)
    X_test = build_pool10_features(base_test, test_arrays)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns differ")

    print("\nTrain rows          :", len(train_df))
    print("Test rows           :", len(test_df))
    print("Image feature count :", X_train.shape[1])

    oof_gated = np.full(len(train_ids), np.nan, dtype=np.float64)
    oof_structural = np.full(len(train_ids), np.nan, dtype=np.float64)
    oof_prior = np.full(len(train_ids), np.nan, dtype=np.float64)
    oof_seen = np.zeros(len(train_ids), dtype=bool)
    test_matrix = np.full((N_FOLDS, len(test_ids)), np.nan, dtype=np.float64)

    fold_rows = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 78)
        print(f"FOLD {fold}")
        print("=" * 78)

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

        oof_gated[va_idx] = result["gated_valid_prob"]
        oof_structural[va_idx] = result["structural_valid_prob"]
        oof_prior[va_idx] = result["prior_valid_prob"]
        oof_seen[va_idx] = result["valid_seen"]
        test_matrix[fold] = result["gated_test_prob"]

        fold_rows.append({
            "run_id": RUN_ID,
            "fold": fold,
            "n_image_features": X_train.shape[1],
            "structural_log_loss": result["structural_loss"],
            "prior_log_loss": result["prior_loss"],
            "gated_log_loss": result["gated_loss"],
            "seen_gated_log_loss": result["seen_loss"],
            "unseen_observed_gated_log_loss": result["unseen_observed_loss"],
            "valid_seen_rate_observed": result["valid_seen_rate_observed"],
            "test_seen_rate": result["test_seen_rate"],
            "test_like_observed_reweighted_log_loss": result["test_like"],
        })

        print(f"Structural : {result['structural_loss']:.6f}")
        print(f"Prior      : {result['prior_loss']:.6f}")
        print(f"Gated      : {result['gated_loss']:.6f}")
        print(f"Seen       : {result['seen_loss']:.6f}")
        print(f"Unseen     : {result['unseen_observed_loss']:.6f}")
        print(f"Test-like  : {result['test_like']:.6f}")

    if (
        np.isnan(oof_gated).any()
        or np.isnan(oof_structural).any()
        or np.isnan(oof_prior).any()
        or np.isnan(test_matrix).any()
    ):
        raise ValueError("Incomplete predictions")

    oof_gated = clip_prob(oof_gated)
    oof_structural = clip_prob(oof_structural)
    oof_prior = clip_prob(oof_prior)
    test_pred = clip_prob(test_matrix.mean(axis=0))

    observed = np.isfinite(train_angle)
    seen = oof_seen
    unseen_observed = (~seen) & observed

    overall = safe_log_loss(y, oof_gated)
    structural = safe_log_loss(y, oof_structural)
    prior = safe_log_loss(y, oof_prior)
    seen_loss = safe_log_loss(y[seen], oof_gated[seen])
    unseen_loss = safe_log_loss(y[unseen_observed], oof_gated[unseen_observed])

    fold_df = pd.DataFrame(fold_rows)
    mean_test_seen_rate = float(fold_df["test_seen_rate"].mean())

    test_like = (
        mean_test_seen_rate * seen_loss
        + (1.0 - mean_test_seen_rate) * unseen_loss
    )

    mean_fold = float(fold_df["gated_log_loss"].mean())
    std_fold = float(fold_df["gated_log_loss"].std(ddof=0))

    oof_df = pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": oof_gated,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    })

    test_df_out = pd.DataFrame({
        "id": test_ids,
        "prediction": test_pred,
        "model_name": "lightgbm_c",
        "run_id": RUN_ID,
    })

    summary_df = pd.DataFrame([{
        "run_id": RUN_ID,
        "experiment": "C3c_pool10_only",
        "n_image_features": int(X_train.shape[1]),
        "grid_size": GRID_SIZE,
        "overall_oof_log_loss": overall,
        "structural_oof_log_loss": structural,
        "prior_oof_log_loss": prior,
        "seen_group_oof_log_loss": seen_loss,
        "unseen_observed_oof_log_loss": unseen_loss,
        "test_like_observed_reweighted_log_loss": test_like,
        "mean_fold_log_loss": mean_fold,
        "std_fold_log_loss": std_fold,
        "mean_test_seen_rate": mean_test_seen_rate,
    }])

    oof_df.to_csv(OUT_OOF_PATH, index=False)
    test_df_out.to_csv(OUT_TEST_PATH, index=False)
    fold_df.to_csv(OUT_FOLDS_PATH, index=False)
    summary_df.to_csv(OUT_SUMMARY_PATH, index=False)

    comparison_rows = []

    if C1E_SUMMARY_PATH.exists():
        ref = pd.read_csv(C1E_SUMMARY_PATH).iloc[0]
        comparison_rows.append({
            "experiment": "C1e_reference",
            "n_image_features": int(ref.get("image_feature_count", 62)),
            "overall_oof_log_loss": float(ref["overall_oof_log_loss"]),
            "seen_group_oof_log_loss": float(ref["seen_group_oof_log_loss"]),
            "unseen_observed_oof_log_loss": float(
                ref["unseen_observed_oof_log_loss"]
            ),
            "test_like_observed_reweighted_log_loss": float(
                ref["test_like_observed_reweighted_log_loss"]
            ),
        })

    if C3B_SUMMARY_PATH.exists():
        c3b = pd.read_csv(C3B_SUMMARY_PATH).iloc[0]
        comparison_rows.append({
            "experiment": "C3b_pool5_10",
            "n_image_features": int(c3b["n_image_features"]),
            "overall_oof_log_loss": float(c3b["overall_oof_log_loss"]),
            "seen_group_oof_log_loss": float(c3b["seen_group_oof_log_loss"]),
            "unseen_observed_oof_log_loss": float(
                c3b["unseen_observed_oof_log_loss"]
            ),
            "test_like_observed_reweighted_log_loss": float(
                c3b["test_like_observed_reweighted_log_loss"]
            ),
        })

    comparison_rows.append({
        "experiment": "C3c_pool10_only",
        "n_image_features": int(X_train.shape[1]),
        "overall_oof_log_loss": overall,
        "seen_group_oof_log_loss": seen_loss,
        "unseen_observed_oof_log_loss": unseen_loss,
        "test_like_observed_reweighted_log_loss": test_like,
    })

    comparison = pd.DataFrame(comparison_rows)

    ref_rows = comparison[
        comparison["experiment"] == "C1e_reference"
    ]

    if len(ref_rows) == 1:
        ref = ref_rows.iloc[0]
        comparison["delta_oof_vs_c1e"] = (
            comparison["overall_oof_log_loss"]
            - float(ref["overall_oof_log_loss"])
        )
        comparison["delta_unseen_vs_c1e"] = (
            comparison["unseen_observed_oof_log_loss"]
            - float(ref["unseen_observed_oof_log_loss"])
        )
        comparison["delta_test_like_vs_c1e"] = (
            comparison["test_like_observed_reweighted_log_loss"]
            - float(ref["test_like_observed_reweighted_log_loss"])
        )

    comparison.to_csv(OUT_COMPARISON_PATH, index=False)

    config = {
        "run_id": RUN_ID,
        "reference": C1E_RUN_ID,
        "only_change": "add 10x10 pooled spatial means; no 5x5 pooling",
        "base_image_features": 62,
        "pooled_features": 300,
        "total_image_features": 362,
        "grid_size": 10,
        "representations": ["band1", "band2", "band1-band2"],
        "angle_logic": "unchanged from C1e",
        "lightgbm_params": "unchanged from C1e",
    }

    OUT_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("C3c POOL10-ONLY RESULT")
    print("=" * 78)
    print(f"Overall OOF       : {overall:.6f}")
    print(f"Seen OOF          : {seen_loss:.6f}")
    print(f"Unseen OOF        : {unseen_loss:.6f}")
    print(f"Test-like         : {test_like:.6f}")

    print("\nComparison:")
    cols = [
        "experiment",
        "n_image_features",
        "overall_oof_log_loss",
        "seen_group_oof_log_loss",
        "unseen_observed_oof_log_loss",
        "test_like_observed_reweighted_log_loss",
    ]
    for col in [
        "delta_oof_vs_c1e",
        "delta_unseen_vs_c1e",
        "delta_test_like_vs_c1e",
    ]:
        if col in comparison.columns:
            cols.append(col)

    print(
        comparison[cols]
        .sort_values("overall_oof_log_loss")
        .to_string(index=False)
    )

    print("\nPer-fold:")
    print(
        fold_df[[
            "fold",
            "gated_log_loss",
            "seen_gated_log_loss",
            "unseen_observed_gated_log_loss",
            "test_like_observed_reweighted_log_loss",
        ]].to_string(index=False)
    )

    print("\nNo final oof_c.csv/test_c.csv was overwritten.")


if __name__ == "__main__":
    main()
