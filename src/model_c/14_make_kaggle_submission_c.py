"""Convert a validated internal Model C test file into a two-column submission."""
from __future__ import annotations

import argparse
from pathlib import Path

from prediction_validation import (
    atomic_write, check_write, checked_ids, checked_probabilities,
    csv_text, read_csv, read_raw_json, validate_test
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRED_DIR = REPO_ROOT / "artifacts" / "predictions"
DEFAULT_INPUT = PRED_DIR / "test_c.csv"
DEFAULT_OUTPUT = PRED_DIR / "submission_c.csv"
DEFAULT_TEST_JSON = REPO_ROOT / "data" / "raw" / "test.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON,
                        help="Reference test IDs; do not validate by row count alone.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Intentionally replace an existing different submission.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw_test = read_raw_json(args.test_json)
    test_ids = checked_ids(raw_test["id"], args.test_json.name)
    internal = validate_test(read_csv(args.input), test_ids, run_id=None)
    submission = internal[["id", "prediction"]].rename(columns={"prediction": "is_iceberg"})
    print("=" * 78)
    print("MAKE KAGGLE SUBMISSION — MODEL C")
    print("=" * 78)
    print("Source run:", internal["run_id"].iloc[0])
    print("Input :", args.input)
    print("Output:", args.output)
    print(f"Validated {len(test_ids)} unique test IDs, aligned to test.json.")
    if args.dry_run:
        print("DRY RUN: no file was written.")
        return
    if args.output.resolve() in {args.input.resolve(), args.test_json.resolve()}:
        raise ValueError("Output must not overwrite the input or reference data")
    text = csv_text(submission)
    status = check_write(args.output, text, args.overwrite)
    if status != "already_identical":
        atomic_write(args.output, text)
    check = read_csv(args.output)
    if list(check.columns) != ["id", "is_iceberg"]:
        raise ValueError("Submission must contain exactly id,is_iceberg")
    actual_ids = checked_ids(check["id"], "submission")
    if actual_ids.tolist() != test_ids.tolist():
        raise ValueError("Submission ID order changed during serialization")
    checked_probabilities(check["is_iceberg"], "submission")
    print(status + ":", args.output)


if __name__ == "__main__":
    main()
