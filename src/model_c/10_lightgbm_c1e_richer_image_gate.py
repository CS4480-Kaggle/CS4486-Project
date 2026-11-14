from pathlib import Path
import json
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# ============================================================
# 0. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

TRAIN_PATH = REPO_ROOT / "data" / "raw" / "train.json"
TEST_PATH = REPO_ROOT / "data" / "raw" / "test.json"
FOLD_PATH = REPO_ROOT / "artifacts" / "folds.csv"

PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
PREP_DIR = REPO_ROOT / "artifacts" / "preprocessing"

for d in [PRED_DIR, METRIC_DIR, PREP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Previous gated baseline for comparison if present.
OLD_GATE_RUN_ID = "lightgbm_c_c1d_seen_unseen_gate_seed42"
OLD_GATE_SUMMARY_PATH = METRIC_DIR / f"summary_{OLD_GATE_RUN_ID}.csv"

# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
N_FOLDS = 5
MODEL_NAME = "lightgbm_c"
EXPERIMENT = "c1e_richer_image_seen_unseen_gate"
RUN_ID = f"{MODEL_NAME}_{EXPERIMENT}_seed{SEED}"

EPS = 1e-7
EARLY_STOPPING_ROUNDS = 100

ANGLE_ROUND_DECIMALS = 4
PATTERN_SMOOTHING_ALPHA = 5.0

# Keep model hyperparameters unchanged.
LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.90,
    "subsample_freq": 1,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.10,
    "reg_lambda": 1.00,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

IMAGE_H = 75
IMAGE_W = 75
CENTER_LO = 20
CENTER_HI = 55

STRUCTURAL_NAME = "richer_structural_no_prior"
PRIOR_NAME = "richer_full_crossfit_no_missing"

# ============================================================
# 2. GENERIC HELPERS
# ============================================================

def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def safe_log_loss(y_true, pred):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    if len(y_true) == 0:
        return np.nan
    return float(log_loss(y_true, clip_prob(pred), labels=[0, 1]))


def fit_model(model, X_train, y_train, X_valid, y_valid):
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    try:
        model.fit(
            X_train,
            y_train,
            eval_X=X_valid,
            eval_y=y_valid,
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )
    return model


# ============================================================
# 3. LOAD / ALIGN
# ============================================================

def load_data():
    for path in [TRAIN_PATH, TEST_PATH, FOLD_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_json(TRAIN_PATH)
    test_df = pd.read_json(TEST_PATH)
    fold_df = pd.read_csv(FOLD_PATH)

    train_ids = train_df["id"].astype(str).to_numpy()
    test_ids = test_df["id"].astype(str).to_numpy()
    y = train_df["is_iceberg"].to_numpy(dtype=np.int64)

    fold_df = fold_df.copy()
    fold_df["id"] = fold_df["id"].astype(str)

    if fold_df["id"].duplicated().any():
        raise ValueError("Duplicate IDs in folds.csv")
    if set(fold_df["id"]) != set(train_ids):
        raise ValueError("folds.csv IDs do not match train IDs")

    aligned = fold_df.set_index("id").loc[train_ids]
    folds = aligned["fold"].to_numpy(dtype=np.int64)
    fold_y = aligned["target"].to_numpy(dtype=np.int64)

    if not np.array_equal(y, fold_y):
        raise ValueError("Target mismatch between train.json and folds.csv")
    if sorted(np.unique(folds).tolist()) != list(range(N_FOLDS)):
        raise ValueError("Expected folds 0..4")

    return train_df, test_df, train_ids, test_ids, y, folds


# ============================================================
# 4. RICHER IMAGE FEATURES (same 62-feature family as your new 9)
# ============================================================

def safe_corr_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)

    num = np.sum(ac * bc, axis=1)
    den = np.sqrt(
        np.sum(ac ** 2, axis=1)
        * np.sum(bc ** 2, axis=1)
    )

    out = np.zeros(len(a), dtype=np.float64)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def add_distribution_stats(features, prefix, x2d):
    q10, q25, q50, q75, q90 = np.percentile(
        x2d, [10, 25, 50, 75, 90], axis=1
    )

    features[f"{prefix}_mean"] = x2d.mean(axis=1)
    features[f"{prefix}_std"] = x2d.std(axis=1)
    features[f"{prefix}_min"] = x2d.min(axis=1)
    features[f"{prefix}_max"] = x2d.max(axis=1)
    features[f"{prefix}_p10"] = q10
    features[f"{prefix}_p25"] = q25
    features[f"{prefix}_median"] = q50
    features[f"{prefix}_p75"] = q75
    features[f"{prefix}_p90"] = q90


def add_center_edge_stats(features, prefix, x3d, edge_mask):
    center = x3d[
        :, CENTER_LO:CENTER_HI, CENTER_LO:CENTER_HI
    ].reshape(len(x3d), -1)

    edge = x3d[:, edge_mask].reshape(len(x3d), -1)

    features[f"{prefix}_center_mean"] = center.mean(axis=1)
    features[f"{prefix}_center_std"] = center.std(axis=1)
    features[f"{prefix}_edge_mean"] = edge.mean(axis=1)
    features[f"{prefix}_edge_std"] = edge.std(axis=1)


def add_quadrant_means(features, prefix, x3d):
    r = IMAGE_H // 2
    c = IMAGE_W // 2

    q1 = x3d[:, :r, :c]
    q2 = x3d[:, :r, c:]
    q3 = x3d[:, r:, :c]
    q4 = x3d[:, r:, c:]

    features[f"{prefix}_q1_mean"] = q1.mean(axis=(1, 2))
    features[f"{prefix}_q2_mean"] = q2.mean(axis=(1, 2))
    features[f"{prefix}_q3_mean"] = q3.mean(axis=(1, 2))
    features[f"{prefix}_q4_mean"] = q4.mean(axis=(1, 2))


def add_gradient_features(features, prefix, x3d):
    gx = np.abs(np.diff(x3d, axis=2))
    gy = np.abs(np.diff(x3d, axis=1))

    features[f"{prefix}_grad_x_mean_abs"] = gx.mean(axis=(1, 2))
    features[f"{prefix}_grad_y_mean_abs"] = gy.mean(axis=(1, 2))


def build_richer_image_features(df):
    if not (df["band_1"].map(len) == IMAGE_H * IMAGE_W).all():
        raise ValueError("Invalid band_1 length")
    if not (df["band_2"].map(len) == IMAGE_H * IMAGE_W).all():
        raise ValueError("Invalid band_2 length")

    b1 = np.stack(df["band_1"].to_numpy()).astype(np.float32)
    b2 = np.stack(df["band_2"].to_numpy()).astype(np.float32)

    if not np.isfinite(b1).all() or not np.isfinite(b2).all():
        raise ValueError("Non-finite radar values")

    diff = b1 - b2
    abs_diff = np.abs(diff)

    b1_img = b1.reshape(-1, IMAGE_H, IMAGE_W)
    b2_img = b2.reshape(-1, IMAGE_H, IMAGE_W)
    diff_img = diff.reshape(-1, IMAGE_H, IMAGE_W)

    features = {}

    add_distribution_stats(features, "band1", b1)
    add_distribution_stats(features, "band2", b2)
    add_distribution_stats(features, "band_diff", diff)

    abs_q50, abs_q90 = np.percentile(
        abs_diff, [50, 90], axis=1
    )
    features["band_absdiff_mean"] = abs_diff.mean(axis=1)
    features["band_absdiff_std"] = abs_diff.std(axis=1)
    features["band_absdiff_median"] = abs_q50
    features["band_absdiff_p90"] = abs_q90

    features["band1_band2_corr"] = safe_corr_rows(b1, b2)

    edge_mask = np.ones((IMAGE_H, IMAGE_W), dtype=bool)
    edge_mask[
        CENTER_LO:CENTER_HI,
        CENTER_LO:CENTER_HI
    ] = False

    add_center_edge_stats(features, "band1", b1_img, edge_mask)
    add_center_edge_stats(features, "band2", b2_img, edge_mask)
    add_center_edge_stats(features, "band_diff", diff_img, edge_mask)

    add_quadrant_means(features, "band1", b1_img)
    add_quadrant_means(features, "band2", b2_img)
    add_quadrant_means(features, "band_diff", diff_img)

    add_gradient_features(features, "band1", b1_img)
    add_gradient_features(features, "band2", b2_img)
    add_gradient_features(features, "band_diff", diff_img)

    X = pd.DataFrame(features).astype(np.float32)

    if X.shape[1] != 62:
        raise ValueError(f"Expected 62 image features, got {X.shape[1]}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Invalid richer image features")

    return X


# ============================================================
# 5. FOLD-SAFE ANGLE NORMALIZATION
# ============================================================

def fit_angle_preprocessor(angle_train):
    angle_train = np.asarray(angle_train, dtype=np.float64)
    observed = angle_train[np.isfinite(angle_train)]

    if len(observed) == 0:
        raise ValueError("No observed angle in training subset")

    median = float(np.median(observed))
    filled = np.where(np.isfinite(angle_train), angle_train, median)
    mean = float(filled.mean())
    std = float(filled.std(ddof=0))

    if not np.isfinite(std) or std < 1e-12:
        raise ValueError("Invalid angle std")

    return {
        "angle_median": median,
        "angle_mean": mean,
        "angle_std": std,
    }


def transform_angle(angle, stats):
    angle = np.asarray(angle, dtype=np.float64)
    missing = ~np.isfinite(angle)
    filled = np.where(missing, stats["angle_median"], angle)
    normalized = (filled - stats["angle_mean"]) / stats["angle_std"]

    return {
        "inc_angle_norm": normalized.astype(np.float32),
        "angle_missing": missing.astype(np.float32),
    }


# ============================================================
# 6. ANGLE GROUP FEATURES
# ============================================================

def make_angle_keys(angle):
    angle = np.asarray(angle, dtype=np.float64)
    keys = np.empty(len(angle), dtype=object)

    finite = np.isfinite(angle)
    keys[~finite] = None

    rounded = np.round(angle[finite], ANGLE_ROUND_DECIMALS)
    keys[finite] = [
        f"{value:.{ANGLE_ROUND_DECIMALS}f}"
        for value in rounded
    ]

    return keys


def fit_group_mapping(angle_values, y_values):
    keys = make_angle_keys(angle_values)
    y_values = np.asarray(y_values, dtype=np.int64)

    global_prior = float(y_values.mean())
    mapping = {}

    for key, target in zip(keys, y_values):
        if key is None:
            continue

        if key not in mapping:
            mapping[key] = [0, 0.0]

        mapping[key][0] += 1
        mapping[key][1] += float(target)

    return mapping, global_prior


def encode_group_features(angle_query, mapping, global_prior):
    keys = make_angle_keys(angle_query)

    seen = np.zeros(len(keys), dtype=np.float32)
    log_count = np.zeros(len(keys), dtype=np.float32)
    prior = np.full(len(keys), global_prior, dtype=np.float32)

    for i, key in enumerate(keys):
        if key is None:
            continue

        count, target_sum = mapping.get(key, [0, 0.0])

        if count <= 0:
            continue

        seen[i] = 1.0
        log_count[i] = np.log1p(float(count))
        prior[i] = (
            target_sum
            + PATTERN_SMOOTHING_ALPHA * global_prior
        ) / (
            count
            + PATTERN_SMOOTHING_ALPHA
        )

    return pd.DataFrame({
        "angle_group_seen": seen,
        "angle_group_log_count": log_count,
        "angle_group_prior": prior,
    }).astype(np.float32)


# ============================================================
# 7. STRICT INNER CROSSFIT FOR OUTER-TRAIN ROWS
# ============================================================

def crossfit_training_pattern(
    outer_train_idx,
    folds,
    train_angle,
    y,
):
    columns = [
        "angle_group_seen",
        "angle_group_log_count",
        "angle_group_prior",
    ]

    result = pd.DataFrame(
        index=outer_train_idx,
        columns=columns,
        dtype=np.float32,
    )

    inner_folds = sorted(
        np.unique(folds[outer_train_idx]).tolist()
    )

    for inner_valid_fold in inner_folds:
        inner_valid_idx = outer_train_idx[
            folds[outer_train_idx] == inner_valid_fold
        ]
        inner_train_idx = outer_train_idx[
            folds[outer_train_idx] != inner_valid_fold
        ]

        mapping, global_prior = fit_group_mapping(
            train_angle[inner_train_idx],
            y[inner_train_idx],
        )

        encoded = encode_group_features(
            train_angle[inner_valid_idx],
            mapping,
            global_prior,
        )

        result.loc[
            inner_valid_idx,
            columns,
        ] = encoded[columns].to_numpy()

    if result.isna().any().any():
        raise ValueError("Incomplete inner-crossfit group features")

    return (
        result
        .loc[outer_train_idx]
        .reset_index(drop=True)
        .astype(np.float32)
    )


# ============================================================
# 8. FEATURE ASSEMBLY
# ============================================================

def assemble_features(
    image_features,
    angle_features,
    pattern_features,
    *,
    include_prior,
    include_missing,
):
    X = image_features.reset_index(drop=True).copy()

    X["inc_angle_norm"] = angle_features["inc_angle_norm"]

    if include_missing:
        X["angle_missing"] = angle_features["angle_missing"]

    X["angle_group_seen"] = (
        pattern_features["angle_group_seen"].to_numpy(dtype=np.float32)
    )
    X["angle_group_log_count"] = (
        pattern_features["angle_group_log_count"].to_numpy(dtype=np.float32)
    )

    if include_prior:
        X["angle_group_prior"] = (
            pattern_features["angle_group_prior"].to_numpy(dtype=np.float32)
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Assembled feature table contains invalid values")

    return X


# ============================================================
# 9. TRAIN ONE BRANCH FOR ONE FOLD
# ============================================================

def train_branch(
    fold,
    branch_name,
    include_prior,
    include_missing,
    X_train_image,
    X_valid_image,
    X_test_image,
    train_angle_f,
    valid_angle_f,
    test_angle_f,
    train_pattern,
    valid_pattern,
    test_pattern,
    y_train,
    y_valid,
):
    X_train = assemble_features(
        X_train_image,
        train_angle_f,
        train_pattern,
        include_prior=include_prior,
        include_missing=include_missing,
    )

    X_valid = assemble_features(
        X_valid_image,
        valid_angle_f,
        valid_pattern,
        include_prior=include_prior,
        include_missing=include_missing,
    )

    X_test = assemble_features(
        X_test_image,
        test_angle_f,
        test_pattern,
        include_prior=include_prior,
        include_missing=include_missing,
    )

    params = dict(LGBM_PARAMS)
    params["random_state"] = SEED + fold

    model = lgb.LGBMClassifier(**params)
    model = fit_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    valid_prob = clip_prob(
        model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]
    )

    test_prob = clip_prob(
        model.predict_proba(
            X_test,
            num_iteration=model.best_iteration_,
        )[:, 1]
    )

    gains = model.booster_.feature_importance(
        importance_type="gain"
    )

    importance = pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "gain": gains.astype(np.float64),
        "branch": branch_name,
        "fold": fold,
    })

    return {
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "best_iteration": int(model.best_iteration_),
        "n_features": int(X_train.shape[1]),
        "importance": importance,
    }


# ============================================================
# 10. ONE OUTER FOLD
# ============================================================

def run_fold(
    fold,
    image_X,
    image_test_X,
    y,
    folds,
    train_angle,
    test_angle,
):
    outer_train_idx = np.where(folds != fold)[0]
    outer_valid_idx = np.where(folds == fold)[0]

    y_train = y[outer_train_idx]
    y_valid = y[outer_valid_idx]

    angle_stats = fit_angle_preprocessor(
        train_angle[outer_train_idx]
    )

    train_angle_f = transform_angle(
        train_angle[outer_train_idx],
        angle_stats,
    )
    valid_angle_f = transform_angle(
        train_angle[outer_valid_idx],
        angle_stats,
    )
    test_angle_f = transform_angle(
        test_angle,
        angle_stats,
    )

    train_pattern = crossfit_training_pattern(
        outer_train_idx,
        folds,
        train_angle,
        y,
    )

    mapping, global_prior = fit_group_mapping(
        train_angle[outer_train_idx],
        y[outer_train_idx],
    )

    valid_pattern = encode_group_features(
        train_angle[outer_valid_idx],
        mapping,
        global_prior,
    )

    test_pattern = encode_group_features(
        test_angle,
        mapping,
        global_prior,
    )

    # Unseen fallback branch: richer image + structural angle features.
    structural = train_branch(
        fold=fold,
        branch_name=STRUCTURAL_NAME,
        include_prior=False,
        include_missing=True,
        X_train_image=image_X.iloc[outer_train_idx],
        X_valid_image=image_X.iloc[outer_valid_idx],
        X_test_image=image_test_X,
        train_angle_f=train_angle_f,
        valid_angle_f=valid_angle_f,
        test_angle_f=test_angle_f,
        train_pattern=train_pattern,
        valid_pattern=valid_pattern,
        test_pattern=test_pattern,
        y_train=y_train,
        y_valid=y_valid,
    )

    # Seen branch: richer image + safe target prior, no missing flag.
    prior = train_branch(
        fold=fold,
        branch_name=PRIOR_NAME,
        include_prior=True,
        include_missing=False,
        X_train_image=image_X.iloc[outer_train_idx],
        X_valid_image=image_X.iloc[outer_valid_idx],
        X_test_image=image_test_X,
        train_angle_f=train_angle_f,
        valid_angle_f=valid_angle_f,
        test_angle_f=test_angle_f,
        train_pattern=train_pattern,
        valid_pattern=valid_pattern,
        test_pattern=test_pattern,
        y_train=y_train,
        y_valid=y_valid,
    )

    valid_seen = (
        valid_pattern["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
    )
    test_seen = (
        test_pattern["angle_group_seen"].to_numpy(dtype=np.float32) > 0.5
    )

    gated_valid = np.where(
        valid_seen,
        prior["valid_prob"],
        structural["valid_prob"],
    )

    gated_test = np.where(
        test_seen,
        prior["test_prob"],
        structural["test_prob"],
    )

    observed_valid = np.isfinite(
        train_angle[outer_valid_idx]
    )
    unseen_observed = (~valid_seen) & observed_valid

    structural_loss = safe_log_loss(
        y_valid,
        structural["valid_prob"],
    )
    prior_loss = safe_log_loss(
        y_valid,
        prior["valid_prob"],
    )
    gated_loss = safe_log_loss(
        y_valid,
        gated_valid,
    )

    seen_loss = safe_log_loss(
        y_valid[valid_seen],
        gated_valid[valid_seen],
    )
    unseen_observed_loss = safe_log_loss(
        y_valid[unseen_observed],
        gated_valid[unseen_observed],
    )

    test_seen_rate = float(test_seen.mean())

    if np.isfinite(seen_loss) and np.isfinite(unseen_observed_loss):
        test_like = (
            test_seen_rate * seen_loss
            + (1.0 - test_seen_rate) * unseen_observed_loss
        )
    else:
        test_like = np.nan

    return {
        "outer_valid_idx": outer_valid_idx,
        "angle_stats": angle_stats,
        "global_prior": global_prior,
        "valid_seen": valid_seen,
        "test_seen": test_seen,
        "structural_valid_prob": structural["valid_prob"],
        "structural_test_prob": structural["test_prob"],
        "prior_valid_prob": prior["valid_prob"],
        "prior_test_prob": prior["test_prob"],
        "gated_valid_prob": clip_prob(gated_valid),
        "gated_test_prob": clip_prob(gated_test),
        "structural_best_iteration": structural["best_iteration"],
        "prior_best_iteration": prior["best_iteration"],
        "structural_n_features": structural["n_features"],
        "prior_n_features": prior["n_features"],
        "structural_loss": structural_loss,
        "prior_loss": prior_loss,
        "gated_loss": gated_loss,
        "seen_loss": seen_loss,
        "unseen_observed_loss": unseen_observed_loss,
        "valid_seen_rate_observed": float(
            valid_seen[observed_valid].mean()
        ) if observed_valid.any() else np.nan,
        "test_seen_rate": test_seen_rate,
        "test_like": test_like,
        "importance": pd.concat(
            [structural["importance"], prior["importance"]],
            ignore_index=True,
        ),
    }


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("=" * 78)
    print("MODEL C — LIGHTGBM C1e RICHER-IMAGE SEEN/UNSEEN GATED MODEL")
    print("=" * 78)
    print("Seen exact angle group -> richer image + safe angle prior")
    print("Unseen/missing         -> richer image + structural angle fallback")
    print("Image features         -> 62 richer target-free features")
    print("LightGBM params        -> unchanged")

    (
        train_df,
        test_df,
        train_ids,
        test_ids,
        y,
        folds,
    ) = load_data()

    print("\nBuilding richer TRAIN image features...")
    image_X = build_richer_image_features(train_df)

    print("Building richer TEST image features...")
    image_test_X = build_richer_image_features(test_df)

    train_angle = pd.to_numeric(
        train_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    test_angle = pd.to_numeric(
        test_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    print("\nTrain rows          :", len(train_df))
    print("Test rows           :", len(test_df))
    print("Image feature count :", image_X.shape[1])
    print("Train missing angle :", int(np.isnan(train_angle).sum()))
    print("Test missing angle  :", int(np.isnan(test_angle).sum()))

    n_train = len(train_ids)
    n_test = len(test_ids)

    oof_gated = np.full(n_train, np.nan, dtype=np.float64)
    oof_structural = np.full(n_train, np.nan, dtype=np.float64)
    oof_prior = np.full(n_train, np.nan, dtype=np.float64)
    oof_seen = np.zeros(n_train, dtype=bool)

    test_gated_matrix = np.full(
        (N_FOLDS, n_test),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []
    prep_rows = []
    importance_parts = []
    oof_diag_parts = []
    test_diag_parts = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 78)
        print(f"FOLD {fold}")
        print("=" * 78)

        result = run_fold(
            fold,
            image_X,
            image_test_X,
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

        test_gated_matrix[fold] = result["gated_test_prob"]

        fold_rows.append({
            "run_id": RUN_ID,
            "fold": fold,
            "structural_best_iteration":
                result["structural_best_iteration"],
            "prior_best_iteration":
                result["prior_best_iteration"],
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
            "valid_seen_rate_observed":
                result["valid_seen_rate_observed"],
            "test_seen_rate":
                result["test_seen_rate"],
            "test_like_observed_reweighted_log_loss":
                result["test_like"],
        })

        prep_rows.append({
            "run_id": RUN_ID,
            "fold": fold,
            **result["angle_stats"],
            "group_global_prior":
                result["global_prior"],
        })

        importance_parts.append(
            result["importance"]
        )

        valid_source = np.where(
            result["valid_seen"],
            PRIOR_NAME,
            STRUCTURAL_NAME,
        )

        oof_diag_parts.append(pd.DataFrame({
            "id": train_ids[va_idx],
            "fold": fold,
            "target": y[va_idx],
            "angle_raw": train_angle[va_idx],
            "angle_observed":
                np.isfinite(train_angle[va_idx]).astype(np.int8),
            "angle_group_seen":
                result["valid_seen"].astype(np.int8),
            "selected_source": valid_source,
            "prediction_structural":
                result["structural_valid_prob"],
            "prediction_prior":
                result["prior_valid_prob"],
            "prediction_gated":
                result["gated_valid_prob"],
        }))

        test_source = np.where(
            result["test_seen"],
            PRIOR_NAME,
            STRUCTURAL_NAME,
        )

        test_diag_parts.append(pd.DataFrame({
            "id": test_ids,
            "fold": fold,
            "angle_raw": test_angle,
            "angle_group_seen":
                result["test_seen"].astype(np.int8),
            "selected_source": test_source,
            "prediction_structural":
                result["structural_test_prob"],
            "prediction_prior":
                result["prior_test_prob"],
            "prediction_gated":
                result["gated_test_prob"],
        }))

        print(f"Structural LogLoss : {result['structural_loss']:.6f}")
        print(f"Prior LogLoss      : {result['prior_loss']:.6f}")
        print(f"Gated LogLoss      : {result['gated_loss']:.6f}")
        print(f"Seen gated loss    : {result['seen_loss']:.6f}")
        print(f"Unseen obs loss    : {result['unseen_observed_loss']:.6f}")
        print(f"Valid seen(obs)    : {result['valid_seen_rate_observed']:.4f}")
        print(f"Test seen          : {result['test_seen_rate']:.4f}")
        print(f"Test-like diagnostic: {result['test_like']:.6f}")

    for name, arr in [
        ("oof_gated", oof_gated),
        ("oof_structural", oof_structural),
        ("oof_prior", oof_prior),
        ("test_gated_matrix", test_gated_matrix),
    ]:
        if np.isnan(arr).any():
            raise ValueError(f"Incomplete predictions: {name}")

    oof_gated = clip_prob(oof_gated)
    oof_structural = clip_prob(oof_structural)
    oof_prior = clip_prob(oof_prior)
    test_pred = clip_prob(test_gated_matrix.mean(axis=0))

    observed = np.isfinite(train_angle)
    seen = oof_seen
    unseen_observed = (~seen) & observed

    overall_gated = safe_log_loss(y, oof_gated)
    overall_structural = safe_log_loss(y, oof_structural)
    overall_prior = safe_log_loss(y, oof_prior)

    observed_gated = safe_log_loss(
        y[observed],
        oof_gated[observed],
    )
    seen_gated = safe_log_loss(
        y[seen],
        oof_gated[seen],
    )
    unseen_observed_gated = safe_log_loss(
        y[unseen_observed],
        oof_gated[unseen_observed],
    )

    fold_df = pd.DataFrame(fold_rows)
    mean_test_seen_rate = float(fold_df["test_seen_rate"].mean())

    test_like_gated = (
        mean_test_seen_rate * seen_gated
        + (1.0 - mean_test_seen_rate)
        * unseen_observed_gated
    )

    mean_fold = float(fold_df["gated_log_loss"].mean())
    std_fold = float(fold_df["gated_log_loss"].std(ddof=0))
    mean_fold_test_like = float(
        fold_df[
            "test_like_observed_reweighted_log_loss"
        ].mean()
    )

    # Standard predictions.
    oof_df = pd.DataFrame({
        "id": train_ids,
        "fold": folds,
        "target": y,
        "prediction": oof_gated,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
    })

    test_df_out = pd.DataFrame({
        "id": test_ids,
        "prediction": test_pred,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
    })

    # Summary.
    summary = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "experiment": EXPERIMENT,
        "image_feature_count": int(image_X.shape[1]),
        "overall_oof_log_loss": overall_gated,
        "observed_angle_oof_log_loss": observed_gated,
        "seen_group_oof_log_loss": seen_gated,
        "unseen_observed_oof_log_loss":
            unseen_observed_gated,
        "test_like_observed_reweighted_log_loss":
            test_like_gated,
        "mean_fold_test_like_observed_reweighted_log_loss":
            mean_fold_test_like,
        "mean_fold_log_loss": mean_fold,
        "std_fold_log_loss": std_fold,
        "recomputed_richer_structural_oof_log_loss":
            overall_structural,
        "recomputed_richer_prior_oof_log_loss":
            overall_prior,
        "oof_seen_rate_all": float(seen.mean()),
        "oof_seen_rate_observed":
            float(seen[observed].mean()),
        "mean_test_seen_rate": mean_test_seen_rate,
        "angle_round_decimals": ANGLE_ROUND_DECIMALS,
        "pattern_smoothing_alpha":
            PATTERN_SMOOTHING_ALPHA,
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
    }

    # Compare to old gated model if its summary exists.
    comparison_rows = []

    if OLD_GATE_SUMMARY_PATH.exists():
        old = pd.read_csv(OLD_GATE_SUMMARY_PATH).iloc[0]

        comparison_rows.append({
            "experiment": "C1d_basic_image_gate",
            "run_id": OLD_GATE_RUN_ID,
            "overall_oof_log_loss":
                float(old["overall_oof_log_loss"]),
            "unseen_observed_oof_log_loss":
                float(old["unseen_observed_oof_log_loss"]),
            "test_like_observed_reweighted_log_loss":
                float(old["test_like_observed_reweighted_log_loss"]),
        })

        summary["delta_oof_vs_c1d"] = (
            overall_gated
            - float(old["overall_oof_log_loss"])
        )
        summary["delta_unseen_vs_c1d"] = (
            unseen_observed_gated
            - float(old["unseen_observed_oof_log_loss"])
        )
        summary["delta_test_like_vs_c1d"] = (
            test_like_gated
            - float(old["test_like_observed_reweighted_log_loss"])
        )

    comparison_rows.append({
        "experiment": "C1e_richer_image_gate",
        "run_id": RUN_ID,
        "overall_oof_log_loss": overall_gated,
        "unseen_observed_oof_log_loss":
            unseen_observed_gated,
        "test_like_observed_reweighted_log_loss":
            test_like_gated,
    })

    comparison_df = pd.DataFrame(comparison_rows)

    # Feature importance.
    importance_raw = pd.concat(
        importance_parts,
        ignore_index=True,
    )

    importance = (
        importance_raw
        .groupby(["branch", "feature"], as_index=False)
        .agg(
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
        )
    )

    importance["gain_fraction"] = (
        importance
        .groupby("branch")["mean_gain"]
        .transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
    )

    importance = importance.sort_values(
        ["branch", "mean_gain"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Diagnostics.
    oof_diag_df = (
        pd.concat(oof_diag_parts, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_diag_df = pd.concat(
        test_diag_parts,
        ignore_index=True,
    )
    prep_df = pd.DataFrame(prep_rows)
    summary_df = pd.DataFrame([summary])

    # Save.
    oof_path = PRED_DIR / f"oof_{RUN_ID}.csv"
    test_path = PRED_DIR / f"test_{RUN_ID}.csv"
    summary_path = METRIC_DIR / f"summary_{RUN_ID}.csv"
    folds_path = METRIC_DIR / f"folds_{RUN_ID}.csv"
    comparison_path = (
        METRIC_DIR / "comparison_lightgbm_c1d_c1e_richer_gate.csv"
    )
    importance_path = (
        METRIC_DIR / f"feature_importance_{RUN_ID}.csv"
    )
    oof_diag_path = (
        METRIC_DIR / f"gate_oof_diagnostics_{RUN_ID}.csv"
    )
    test_diag_path = (
        METRIC_DIR / f"gate_test_fold_diagnostics_{RUN_ID}.csv"
    )
    prep_path = PREP_DIR / f"angle_stats_{RUN_ID}.csv"
    config_path = METRIC_DIR / f"config_{RUN_ID}.json"

    oof_df.to_csv(oof_path, index=False)
    test_df_out.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    importance.to_csv(importance_path, index=False)
    oof_diag_df.to_csv(oof_diag_path, index=False)
    test_diag_df.to_csv(test_diag_path, index=False)
    prep_df.to_csv(prep_path, index=False)

    config = {
        "run_id": RUN_ID,
        "image_feature_count": int(image_X.shape[1]),
        "gate": {
            "seen": PRIOR_NAME,
            "unseen_or_missing": STRUCTURAL_NAME,
        },
        "lgbm_params": LGBM_PARAMS,
        "angle_round_decimals": ANGLE_ROUND_DECIMALS,
        "pattern_smoothing_alpha": PATTERN_SMOOTHING_ALPHA,
        "note": (
            "Same fold-safe gating logic as C1d, but with the 62 richer "
            "image features from C0b."
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("C1e RICHER-IMAGE GATED RESULT")
    print("=" * 78)
    print(f"Richer structural OOF    : {overall_structural:.6f}")
    print(f"Richer prior OOF         : {overall_prior:.6f}")
    print(f"GATED overall OOF        : {overall_gated:.6f}")
    print(f"GATED observed-angle OOF : {observed_gated:.6f}")
    print(f"GATED seen-group OOF     : {seen_gated:.6f}")
    print(f"GATED unseen-observed OOF: {unseen_observed_gated:.6f}")
    print(f"Mean test seen rate      : {mean_test_seen_rate:.4f}")
    print(f"GATED test-like          : {test_like_gated:.6f}")
    print(f"Mean fold test-like      : {mean_fold_test_like:.6f}")

    if OLD_GATE_SUMMARY_PATH.exists():
        old = pd.read_csv(OLD_GATE_SUMMARY_PATH).iloc[0]
        print("\nComparison vs previous C1d:")
        print(
            f"OOF delta       : "
            f"{overall_gated - float(old['overall_oof_log_loss']):+.6f}"
        )
        print(
            f"Unseen delta    : "
            f"{unseen_observed_gated - float(old['unseen_observed_oof_log_loss']):+.6f}"
        )
        print(
            f"Test-like delta : "
            f"{test_like_gated - float(old['test_like_observed_reweighted_log_loss']):+.6f}"
        )

    print("\nPer-fold summary:")
    print(
        fold_df[[
            "fold",
            "structural_log_loss",
            "prior_log_loss",
            "gated_log_loss",
            "seen_gated_log_loss",
            "unseen_observed_gated_log_loss",
            "valid_seen_rate_observed",
            "test_seen_rate",
            "test_like_observed_reweighted_log_loss",
        ]].to_string(index=False)
    )

    print("\nTop 15 structural features:")
    print(
        importance[
            importance["branch"] == STRUCTURAL_NAME
        ][["feature", "mean_gain", "gain_fraction"]]
        .head(15)
        .to_string(index=False)
    )

    print("\nTop 15 prior features:")
    print(
        importance[
            importance["branch"] == PRIOR_NAME
        ][["feature", "mean_gain", "gain_fraction"]]
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    for p in [
        oof_path,
        test_path,
        summary_path,
        folds_path,
        comparison_path,
        importance_path,
        oof_diag_path,
        test_diag_path,
        prep_path,
        config_path,
    ]:
        print(" ", p)

    print("\nNo final oof_c.csv/test_c.csv was created.")
    print("Inspect this result before trying reliability/purity refinement.")


if __name__ == "__main__":
    main()
