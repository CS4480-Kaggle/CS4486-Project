from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from calibration import apply_temperature,crossfit_temperature,find_temperature


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

PREDICTION_DIR = REPO_ROOT / "artifacts" / "predictions"
METRIC_DIR = REPO_ROOT / "artifacts" / "metrics"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply temperature scaling to OOF and test predictions.")

    parser.add_argument("--source-run-id", required=True, help="Run ID used by the source OOF and test files.")
    parser.add_argument("--output-run-id", default="", help="Optional calibrated run ID.")

    return parser.parse_args()


# def validate_oof(frame, source_run_id):
#     required_columns = {
#         "id",
#         "fold",
#         "target",
#         "prediction",
#         "model_name",
#         "run_id"
#     }

#     if not required_columns.issubset(frame.columns):
#         raise ValueError("Invalid OOF prediction interface.")

#     if frame["id"].duplicated().any():
#         raise ValueError("Duplicate IDs in OOF predictions.")

#     if frame["prediction"].isna().any():
#         raise ValueError("Missing OOF predictions.")

#     if not frame["prediction"].between(0.0, 1.0).all():
#         raise ValueError("OOF probabilities are outside [0, 1].")

#     if frame["run_id"].nunique() != 1:
#         raise ValueError("OOF file contains multiple run IDs.")

#     if frame["run_id"].iloc[0] != source_run_id:
#         raise ValueError("OOF run ID does not match the command.")

#     expected_folds = [0, 1, 2, 3, 4]
#     actual_folds = sorted(frame["fold"].unique().tolist())

#     if actual_folds != expected_folds:
#         raise ValueError("Expected folds 0 through 4.")


# def validate_test(frame, source_run_id):
#     required_columns = {
#         "id",
#         "prediction",
#         "model_name",
#         "run_id"
#     }

#     if not required_columns.issubset(frame.columns):
#         raise ValueError("Invalid test prediction interface.")

#     if frame["id"].duplicated().any():
#         raise ValueError("Duplicate IDs in test predictions.")

#     if frame["prediction"].isna().any():
#         raise ValueError("Missing test predictions.")

#     if not frame["prediction"].between(0.0, 1.0).all():
#         raise ValueError("Test probabilities are outside [0, 1].")

#     if frame["run_id"].nunique() != 1:
#         raise ValueError("Test file contains multiple run IDs.")

#     if frame["run_id"].iloc[0] != source_run_id:
#         raise ValueError("Test run ID does not match the command.")


def main():
    arguments = parse_arguments()

    source_run_id = arguments.source_run_id

    output_run_id = arguments.output_run_id if arguments.output_run_id else f"{source_run_id}_temperature_scaled"
 
    source_oof_path = PREDICTION_DIR / f"oof_{source_run_id}.csv"
    source_test_path = PREDICTION_DIR / f"test_{source_run_id}.csv"

    oof_frame = pd.read_csv(source_oof_path, dtype={"id": str})
    test_frame = pd.read_csv(source_test_path, dtype={"id": str})

    # validate_oof(oof_frame, source_run_id)
    # validate_test(test_frame, source_run_id)

    targets = oof_frame["target"].to_numpy(dtype=np.int64)
    raw_oof = oof_frame["prediction"].to_numpy(dtype=np.float64)
    folds = oof_frame["fold"].to_numpy(dtype=np.int64)
    raw_test = test_frame["prediction"].to_numpy(dtype=np.float64)

    calibrated_oof, fold_records = crossfit_temperature(
        targets=targets,
        probabilities=raw_oof,
        folds=folds
    )

    final_temperature = find_temperature(targets=targets, probabilities=raw_oof)

    calibrated_test = apply_temperature(probabilities=raw_test, temperature=final_temperature)

    raw_oof_loss = log_loss(targets, raw_oof, labels=[0, 1])
    calibrated_oof_loss = log_loss(targets, calibrated_oof, labels=[0, 1])

    source_model_name = oof_frame["model_name"].iloc[0]
    calibrated_model_name = f"{source_model_name}_temperature_scaled"

    calibrated_oof_frame = pd.DataFrame(
        {
            "id": oof_frame["id"],
            "fold": oof_frame["fold"],
            "target": oof_frame["target"],
            "prediction": calibrated_oof,
            "model_name": calibrated_model_name,
            "run_id": output_run_id
        }
    )

    calibrated_test_frame = pd.DataFrame(
        {
            "id": test_frame["id"],
            "prediction": calibrated_test,
            "model_name": calibrated_model_name,
            "run_id": output_run_id
        }
    )

    submission_frame = pd.DataFrame(
        {
            "id": test_frame["id"],
            "is_iceberg": calibrated_test
        }
    )

    fold_metric_frame = pd.DataFrame(fold_records)

    summary_frame = pd.DataFrame(
        [
            {
                "run_id": output_run_id,
                "source_run_id": source_run_id,
                "calibration": "temperature_scaling",
                "raw_oof_log_loss": raw_oof_loss,
                "calibrated_oof_log_loss": calibrated_oof_loss,
                "improvement": (
                    calibrated_oof_loss
                    - raw_oof_loss
                ),
                "final_temperature": final_temperature
            }
        ]
    )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    output_oof_path = PREDICTION_DIR / f"oof_{output_run_id}.csv"
    output_test_path = PREDICTION_DIR / f"test_{output_run_id}.csv"
    output_submission_path = PREDICTION_DIR / f"submission_{output_run_id}.csv"
    output_fold_metric_path = METRIC_DIR / f"folds_{output_run_id}.csv"
    output_summary_path = METRIC_DIR / f"summary_{output_run_id}.csv"

    calibrated_oof_frame.to_csv(output_oof_path, index=False)
    calibrated_test_frame.to_csv(output_test_path, index=False)
    submission_frame.to_csv(output_submission_path, index=False)
    fold_metric_frame.to_csv(output_fold_metric_path, index=False)
    summary_frame.to_csv(output_summary_path, index=False)

    print("=" * 70)
    print("TEMPERATURE SCALING COMPLETE")
    print("=" * 70)
    print(f"Source run ID: {source_run_id}")
    print(f"Output run ID: {output_run_id}")
    print(f"Raw OOF Log Loss: {raw_oof_loss:.9f}")
    print(f"Calibrated OOF Log Loss: {calibrated_oof_loss:.9f}")
    print(f"OOF improvement: {calibrated_oof_loss - raw_oof_loss:+.9f}")
    print(f"Final temperature: {final_temperature:.6f}")
    print()
    print("Outputs:")
    print(output_oof_path)
    print(output_test_path)
    print(output_submission_path)
    print(output_fold_metric_path)
    print(output_summary_path)


if __name__ == "__main__":
    main()
