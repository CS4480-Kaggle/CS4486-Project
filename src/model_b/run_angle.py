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

MODEL_NAME = "simple_cnn_angle_fusion"

EXPERIMENTS = {
    "c0": {"run_id": "c0_image_only_seed42", "metadata_dim": 0},
    "c1": {"run_id": "c1_image_angle_seed42", "metadata_dim": 1},
    "c2": {"run_id": "c2_image_angle_missing_seed42", "metadata_dim": 2}
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fold-safe incidence-angle fusion experiments.")
    
    parser.add_argument("--experiment", choices=["c0", "c1", "c2", "all"],default="c1")
    parser.add_argument("--epochs", type=int, default=OFFICIAL_NUM_EPOCHS)
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    repo_root = Path(__file__).resolve().parents[2]

    train_path = repo_root / "data" / "raw" / "train.json"
    test_path = repo_root / "data" / "raw" / "test.json"
    fold_path = repo_root / "artifacts" / "folds.csv"

    prediction_dir = repo_root / "artifacts" / "predictions"
    metric_dir = repo_root / "artifacts" / "metrics"
    preprocessing_dir = repo_root / "artifacts" / "preprocessing"

    for directory in (prediction_dir, metric_dir, preprocessing_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    project_data = load_project_data(train_path, test_path, fold_path)

    selected_experiments = (
        ["c0", "c1", "c2"]
        if arguments.experiment == "all"
        else [arguments.experiment]
    )

    summaries = []

    for experiment_name in selected_experiments:
        experiment = EXPERIMENTS[experiment_name]
        run_id = experiment["run_id"]

        if arguments.epochs != OFFICIAL_NUM_EPOCHS:
            run_id = f"{run_id}_{arguments.epochs}ep"

        summary = run_experiment(
            project_data=project_data,
            run_id=run_id,
            input_config="mean_3ch",
            in_channels=3,
            num_epochs=arguments.epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            base_seed=SEED,
            device=device,
            prediction_dir=prediction_dir,
            metric_dir=metric_dir,
            preprocessing_dir=preprocessing_dir,
            model_name=MODEL_NAME,
            metadata_dim=experiment["metadata_dim"]
        )

        summaries.append(summary)

    comparison_df = (
        pd.DataFrame(summaries)
        .sort_values("overall_oof_log_loss", ascending=True)
        .reset_index(drop=True)
    )

    print()
    print(comparison_df)

    if arguments.experiment == "all":
        suffix = "" if arguments.epochs == OFFICIAL_NUM_EPOCHS else f"_{arguments.epochs}ep"

        comparison_path = metric_dir / f"comparison_angle{suffix}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print("Comparison saved to:", comparison_path)


if __name__ == "__main__":
    main()
