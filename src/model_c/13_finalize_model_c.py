"""Validate existing predictions and freeze Model C. This script never trains."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from prediction_validation import (
    atomic_write, check_write, checked_ids, checked_integers, csv_text,
    load_shared_folds, read_csv, read_raw_json, sha256_file, validate_oof,
    validate_test,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"
DEFAULT_RUN_ID = "lightgbm_c_c5b_c4_plus_joint_pca30_multiseed_42_52_62"
TRAIN_PATH = REPO_ROOT / "data" / "raw" / "train.json"
TEST_PATH = REPO_ROOT / "data" / "raw" / "test.json"
FOLD_PATH = REPO_ROOT / "artifacts" / "folds.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze an explicitly selected Model C prediction run.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID,
                        help="Source run ID; default is the accepted C5b ensemble.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Intentionally replace existing official predictions/manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report only; do not write or retrain.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a plain experiment identifier, not a path")

    source_oof = PRED_DIR / f"oof_{args.run_id}.csv"
    source_test = PRED_DIR / f"test_{args.run_id}.csv"
    train = read_raw_json(TRAIN_PATH)
    test = read_raw_json(TEST_PATH)
    train_ids = checked_ids(train["id"], "train.json")
    test_ids = checked_ids(test["id"], "test.json")
    y = checked_integers(train["is_iceberg"], [0, 1], "train targets")
    fold_table = load_shared_folds(train, FOLD_PATH)
    folds = fold_table["fold"].to_numpy(dtype=np.int64)
    final_oof = validate_oof(read_csv(source_oof), train_ids, y, folds, args.run_id)
    final_test = validate_test(read_csv(source_test), test_ids, args.run_id)

    pred = np.clip(final_oof["prediction"].to_numpy(), 1e-7, 1 - 1e-7)
    loss = float(-np.mean(y * np.log(pred) + (1 - y) * np.log1p(-pred)))
    print("=" * 78)
    print("FINALIZE MODEL C — SELECTED LIGHTGBM RUN")
    print("=" * 78)
    print("Selected run:", args.run_id)
    print(f"Validation passed: train={len(train_ids)}, test={len(test_ids)}")
    print(f"OOF Log Loss: {loss:.6f}")
    print("This script does NOT train a model or change its probabilities.")
    if args.dry_run:
        print("DRY RUN: no official files were changed.")
        return

    outputs = {
        PRED_DIR / "oof_c.csv": csv_text(final_oof),
        PRED_DIR / "test_c.csv": csv_text(final_test),
    }
    manifest_path = METRIC_DIR / "final_model_c_manifest.json"
    manifest = {
        "official_model_c": {
            "model_name": "lightgbm_c", "run_id": args.run_id,
            "selected_architecture": (
                "C5b: 62 richer image features + pool10 + separate PCA90 + "
                "joint PCA30 + existing angle features/gate; equal-weight seeds 42/52/62"
                if args.run_id == DEFAULT_RUN_ID else
                "Explicit --run-id selection; inspect the source experiment config"
            ),
            "selection_reason": "Explicit selected run; no model selection is performed here",
            "overall_oof_log_loss": loss,
        },
        "source_files": {
            "oof": str(source_oof.relative_to(REPO_ROOT)),
            "test": str(source_test.relative_to(REPO_ROOT)),
            "oof_sha256": sha256_file(source_oof),
            "test_sha256": sha256_file(source_test),
        },
        "data_versions": {
            "train_json_sha256": sha256_file(TRAIN_PATH),
            "test_json_sha256": sha256_file(TEST_PATH),
            "folds_csv_sha256": sha256_file(FOLD_PATH),
        },
        "interface_checks": {
            "n_train": len(train_ids), "n_test": len(test_ids),
            "oof_columns": list(final_oof.columns),
            "test_columns": list(final_test.columns),
            "unique_ids": True, "ids_aligned": True,
            "targets_and_folds_match": True, "finite_probabilities_in_0_1": True,
        },
        "official_files": {},
    }
    import hashlib
    for path, text in outputs.items():
        manifest["official_files"][path.stem] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    outputs[manifest_path] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


    statuses = {p: check_write(p, text, args.overwrite) for p, text in outputs.items()}
    for path, text in outputs.items():
        if statuses[path] != "already_identical":
            atomic_write(path, text)
        print(f"{statuses[path]}: {path}")
    validate_oof(read_csv(PRED_DIR / "oof_c.csv"), train_ids, y, folds, args.run_id)
    validate_test(read_csv(PRED_DIR / "test_c.csv"), test_ids, args.run_id)
    print("Model C official files validated. Next: python src/model_c/14_make_kaggle_submission_c.py")


if __name__ == "__main__":
    main()
