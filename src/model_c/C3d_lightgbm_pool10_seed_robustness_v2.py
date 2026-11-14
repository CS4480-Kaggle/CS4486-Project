"""
C3d_lightgbm_pool10_seed_robustness_v2.py

Seed robustness check for C3c pool10-only versus the stable C1e reference.

Compare:
    C1e reference
vs
    C3c = C1e + 10x10 pooled spatial means

Seeds:
    42, 52, 62

Seed 42:
    reuse existing completed summaries

Seeds 52 and 62:
    retrain both variants under the SAME:
      - shared folds
      - angle preprocessing
      - strict cross-fit prior
      - seen/unseen gate
      - LightGBM hyperparameters

Only the random seed changes.

This is robustness checking, NOT best-seed selection.
"""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

C1E_SCRIPT = SCRIPT_DIR / "10_lightgbm_c1e_richer_image_gate.py"
C3C_SCRIPT = SCRIPT_DIR / "C3c_lightgbm_pool10_only_v2.py"

METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 52, 62]
NEW_SEEDS = [52, 62]
N_FOLDS = 5
EPS = 1e-7

C1E_RUN_ID_42 = "lightgbm_c_c1e_richer_image_seen_unseen_gate_seed42"
C3C_RUN_ID_42 = "lightgbm_c_c3c_pool10_only_richer_gate_seed42"

C1E_SUMMARY_42 = METRIC_DIR / f"summary_{C1E_RUN_ID_42}.csv"
C3C_SUMMARY_42 = METRIC_DIR / f"summary_{C3C_RUN_ID_42}.csv"

OUT_RESULTS = METRIC_DIR / "seed_robustness_c1e_vs_c3c_pool10.csv"
OUT_PAIRED = METRIC_DIR / "seed_robustness_c1e_vs_c3c_pool10_paired.csv"
OUT_SUMMARY = METRIC_DIR / "seed_robustness_c1e_vs_c3c_pool10_summary.csv"
OUT_FOLDS = METRIC_DIR / "seed_robustness_c1e_vs_c3c_pool10_by_fold.csv"


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


def read_existing_summary(path, experiment, seed):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {path}")

    row = df.iloc[0]

    return {
        "seed": seed,
        "experiment": experiment,
        "overall_oof_log_loss": float(row["overall_oof_log_loss"]),
        "seen_group_oof_log_loss": float(row["seen_group_oof_log_loss"]),
        "unseen_observed_oof_log_loss": float(
            row["unseen_observed_oof_log_loss"]
        ),
        "test_like_observed_reweighted_log_loss": float(
            row["test_like_observed_reweighted_log_loss"]
        ),
        "source": "existing_summary",
    }


def run_variant(
    c1e,
    seed,
    experiment,
    X_train,
    X_test,
    train_ids,
    test_ids,
    y,
    folds,
    train_angle,
    test_angle,
):
    set_seed(c1e, seed)

    print("\n" + "=" * 78)
    print(f"{experiment} | seed={seed}")
    print("=" * 78)

    oof = np.full(len(train_ids), np.nan, dtype=np.float64)
    oof_seen = np.zeros(len(train_ids), dtype=bool)
    test_matrix = np.full(
        (N_FOLDS, len(test_ids)),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []

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
        oof_seen[va_idx] = result["valid_seen"]
        test_matrix[fold] = result["gated_test_prob"]

        fold_rows.append({
            "seed": seed,
            "experiment": experiment,
            "fold": fold,
            "gated_log_loss": float(result["gated_loss"]),
            "seen_gated_log_loss": float(result["seen_loss"]),
            "unseen_observed_gated_log_loss": float(
                result["unseen_observed_loss"]
            ),
            "test_seen_rate": float(result["test_seen_rate"]),
            "test_like_observed_reweighted_log_loss": float(
                result["test_like"]
            ),
        })

        print(
            f"Fold {fold} | "
            f"gated={result['gated_loss']:.6f} | "
            f"seen={result['seen_loss']:.6f} | "
            f"unseen={result['unseen_observed_loss']:.6f} | "
            f"test-like={result['test_like']:.6f}"
        )

    if np.isnan(oof).any():
        raise ValueError(f"{experiment}, seed {seed}: incomplete OOF")

    if np.isnan(test_matrix).any():
        raise ValueError(f"{experiment}, seed {seed}: incomplete test predictions")

    oof = clip_prob(oof)

    observed = np.isfinite(train_angle)
    seen = oof_seen
    unseen_observed = (~seen) & observed

    overall = safe_log_loss(y, oof)
    seen_loss = safe_log_loss(y[seen], oof[seen])
    unseen_loss = safe_log_loss(
        y[unseen_observed],
        oof[unseen_observed],
    )

    fold_df = pd.DataFrame(fold_rows)
    mean_test_seen_rate = float(fold_df["test_seen_rate"].mean())

    # Match the same global reweighting used by C1e/C3c summaries.
    test_like = (
        mean_test_seen_rate * seen_loss
        + (1.0 - mean_test_seen_rate) * unseen_loss
    )

    return {
        "seed": seed,
        "experiment": experiment,
        "overall_oof_log_loss": overall,
        "seen_group_oof_log_loss": seen_loss,
        "unseen_observed_oof_log_loss": unseen_loss,
        "test_like_observed_reweighted_log_loss": test_like,
        "source": "retrained",
    }, fold_rows


def main():
    print("=" * 78)
    print("C3d — C1e vs C3c POOL10 SEED ROBUSTNESS")
    print("=" * 78)
    print("Seeds:", SEEDS)
    print("Seed 42 reuses existing summaries.")
    print("Seeds 52/62 retrain both C1e and C3c.")
    print("Do not select the single best seed.")

    c1e = load_module(C1E_SCRIPT, "c1e_seed_robustness")
    c3c = load_module(C3C_SCRIPT, "c3c_feature_module")

    train_df, test_df, train_ids, test_ids, y, folds = c1e.load_data()

    train_angle = pd.to_numeric(
        train_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    test_angle = pd.to_numeric(
        test_df["inc_angle"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    print("\nBuilding C1e 62 features...")
    X_c1e_train = c1e.build_richer_image_features(train_df)
    X_c1e_test = c1e.build_richer_image_features(test_df)

    print("Building C3c 362 features...")
    train_arrays = c3c.extract_image_arrays(train_df)
    test_arrays = c3c.extract_image_arrays(test_df)

    X_c3c_train = c3c.build_pool10_features(
        X_c1e_train,
        train_arrays,
    )
    X_c3c_test = c3c.build_pool10_features(
        X_c1e_test,
        test_arrays,
    )

    print("C1e feature count:", X_c1e_train.shape[1])
    print("C3c feature count:", X_c3c_train.shape[1])

    rows = []
    fold_rows = []

    c1e42 = read_existing_summary(
        C1E_SUMMARY_42,
        "C1e",
        42,
    )
    c3c42 = read_existing_summary(
        C3C_SUMMARY_42,
        "C3c_pool10",
        42,
    )

    if c1e42 is None or c3c42 is None:
        raise FileNotFoundError(
            "Seed-42 summary missing. Run C1e and C3c first."
        )

    rows.extend([c1e42, c3c42])

    for seed in NEW_SEEDS:
        c1e_row, c1e_folds = run_variant(
            c1e,
            seed,
            "C1e",
            X_c1e_train,
            X_c1e_test,
            train_ids,
            test_ids,
            y,
            folds,
            train_angle,
            test_angle,
        )

        c3c_row, c3c_folds = run_variant(
            c1e,
            seed,
            "C3c_pool10",
            X_c3c_train,
            X_c3c_test,
            train_ids,
            test_ids,
            y,
            folds,
            train_angle,
            test_angle,
        )

        rows.extend([c1e_row, c3c_row])
        fold_rows.extend(c1e_folds)
        fold_rows.extend(c3c_folds)

    results = pd.DataFrame(rows).sort_values(
        ["seed", "experiment"]
    ).reset_index(drop=True)

    paired_rows = []

    for seed in SEEDS:
        one = results[
            results["seed"] == seed
        ].set_index("experiment")

        c1 = one.loc["C1e"]
        c3 = one.loc["C3c_pool10"]

        paired_rows.append({
            "seed": seed,
            "delta_oof_c3c_minus_c1e":
                c3["overall_oof_log_loss"]
                - c1["overall_oof_log_loss"],
            "delta_seen_c3c_minus_c1e":
                c3["seen_group_oof_log_loss"]
                - c1["seen_group_oof_log_loss"],
            "delta_unseen_c3c_minus_c1e":
                c3["unseen_observed_oof_log_loss"]
                - c1["unseen_observed_oof_log_loss"],
            "delta_test_like_c3c_minus_c1e":
                c3["test_like_observed_reweighted_log_loss"]
                - c1["test_like_observed_reweighted_log_loss"],
        })

    paired = pd.DataFrame(paired_rows)

    summary = pd.DataFrame([{
        "seeds": "42,52,62",
        "n_seeds": 3,
        "c3c_better_oof_count":
            int((paired["delta_oof_c3c_minus_c1e"] < 0).sum()),
        "c3c_better_seen_count":
            int((paired["delta_seen_c3c_minus_c1e"] < 0).sum()),
        "c3c_better_unseen_count":
            int((paired["delta_unseen_c3c_minus_c1e"] < 0).sum()),
        "c3c_better_test_like_count":
            int((paired["delta_test_like_c3c_minus_c1e"] < 0).sum()),
        "mean_delta_oof":
            float(paired["delta_oof_c3c_minus_c1e"].mean()),
        "mean_delta_seen":
            float(paired["delta_seen_c3c_minus_c1e"].mean()),
        "mean_delta_unseen":
            float(paired["delta_unseen_c3c_minus_c1e"].mean()),
        "mean_delta_test_like":
            float(paired["delta_test_like_c3c_minus_c1e"].mean()),
    }])

    results.to_csv(OUT_RESULTS, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(
            OUT_FOLDS,
            index=False,
        )

    print("\n" + "=" * 78)
    print("PAIRED DELTAS: C3c - C1e")
    print("Negative = C3c better")
    print("=" * 78)
    print(paired.to_string(index=False))

    print("\n" + "=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    print("\nDecision rule:")
    print("  • Promote C3c only if OOF/unseen/test-like improve on most/all seeds.")
    print("  • Do not pick the single best seed.")
    print("  • If signs materially flip, keep C1e as the stable reference.")

    print("\nSaved:")
    print(" ", OUT_RESULTS)
    print(" ", OUT_PAIRED)
    print(" ", OUT_SUMMARY)
    if fold_rows:
        print(" ", OUT_FOLDS)


if __name__ == "__main__":
    main()
