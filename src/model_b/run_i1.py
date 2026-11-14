from pathlib import Path
import argparse

import pandas as pd
import torch

from data_pipeline import load_project_data
from training import run_experiment


SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OFFICIAL_NUM_EPOCHS = 10


EXPERIMENTS = {
    "b0": {
        "run_id": "b0_raw_2ch_seed42",
        "input_config": "raw_2ch",
        "in_channels": 2
    },
    "i1a": {
        "run_id": "i1a_mean_3ch_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3
    },
    "i1b": {
        "run_id": "i1b_difference_3ch_seed42",
        "input_config": "difference_3ch",
        "in_channels": 3
    }
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run I1 iceberg image configuration experiments."
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "b0",
            "i1a",
            "i1b",
            "all"
        ],
        default="i1a"
    )

    parser.add_argument("--epochs", type=int, default=OFFICIAL_NUM_EPOCHS)

    return parser.parse_args()


def create_submission(run_id, prediction_dir):
    test_prediction_path = prediction_dir / f"test_{run_id}.csv"
    test_prediction_df = pd.read_csv(test_prediction_path)
    submission_df = test_prediction_df[["id", "prediction"]].rename(columns={"prediction": "is_iceberg"})
    submission_path = prediction_dir / f"submission_{run_id}.csv"

    submission_df.to_csv(submission_path, index=False)

    print("Submission saved to:", submission_path)


def main():
    arguments = parse_arguments()

    repo_root = Path(__file__).resolve().parents[2]

    train_path = repo_root / "data" / "raw" / "train.json"
    test_path = repo_root / "data" / "raw" / "test.json"
    fold_path = repo_root / "artifacts" / "folds.csv"

    prediction_dir = repo_root / "artifacts" / "predictions"
    metric_dir = repo_root / "artifacts" / "metrics"
    preprocessing_dir = repo_root / "artifacts" / "preprocessing"
    
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    project_data = load_project_data(train_path=train_path, test_path=test_path, fold_path=fold_path)

    if arguments.experiment == "all":
        selected_names = ["b0", "i1a", "i1b"]
    else:
        selected_names = [arguments.experiment]

    summaries = []

    for experiment_name in selected_names:
        experiment = EXPERIMENTS[experiment_name]

        run_id = experiment["run_id"]

        if (arguments.epochs != OFFICIAL_NUM_EPOCHS):
            run_id = f"{run_id}_{arguments.epochs}ep"

        summary = run_experiment(
            project_data=project_data,
            run_id=run_id,
            input_config=experiment["input_config"],
            in_channels=experiment["in_channels"],
            num_epochs=arguments.epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            base_seed=SEED,
            device=device,
            prediction_dir=prediction_dir,
            metric_dir=metric_dir,
            preprocessing_dir=preprocessing_dir
        )

        summaries.append(summary)

    comparison_df = pd.DataFrame(summaries).sort_values("overall_oof_log_loss", ascending=True).reset_index(drop=True)

    if arguments.experiment == "all":
        if (arguments.epochs == OFFICIAL_NUM_EPOCHS):
            comparison_name = "comparison_i1.csv"
        else:
            comparison_name = f"comparison_i1_{arguments.epochs}ep.csv"
            

        comparison_path = metric_dir / comparison_name

        comparison_df.to_csv(comparison_path, index=False)

        print()
        print(comparison_df)
        print("Comparison saved to:", comparison_path)

    best_run_id = comparison_df.loc[0, "run_id"]

    create_submission(run_id=best_run_id, prediction_dir=prediction_dir)


if __name__ == "__main__":
    main()
