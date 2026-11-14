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
    "a2_0": {
        "run_id": "a2_0_mean3ch_control_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "control",
    },
    "a2_1": {
        "run_id": "a2_1_mean3ch_batchnorm_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "batchnorm",
    },
    "a2_2": {
        "run_id": "a2_2_mean3ch_gap_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "gap",
    },
    "a2_3": {
        "run_id": "a2_3_mean3ch_batchnorm_gap_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "batchnorm_gap",
    },
    "a2_4": {
        "run_id": "a2_4_mean3ch_coarsepool3x3_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "coarse_pool_3x3",
    },
    "a2_5": {
        "run_id": "a2_5_mean3ch_coarsepool5x5_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "coarse_pool_5x5",
    },
    "a2_6": {
        "run_id": "a2_6_mean3ch_less_downsample_12x12_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_12x12",
    },
    "a2_7": {
        "run_id": "a2_7_mean3ch_12x12_fc32_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_12x12_fc32",
    },
    "a2_8": {
        "run_id": "a2_8_mean3ch_12x12_wd1e-4_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_12x12",
        "weight_decay": 1e-4,
    },
    "a2_9": {
        "run_id": "a2_9_mean3ch_12x12_wd3e-4_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_12x12",
        "weight_decay": 3e-4,
    },
    "a2_10": {
        "run_id": "a2_10_mean3ch_10x10_fc64_wd1e-4_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_10x10",
        "weight_decay": 1e-4,
    },
    "a2_11": {
        "run_id": "a2_11_mean3ch_12x12_wd1e-4_plateau_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "less_downsample_12x12",
        "weight_decay": 1e-4,
        "use_lr_scheduler": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 1,
        "min_learning_rate": 1e-5,
    },
    "a2_12": {
        "run_id": "a2_12_mean3ch_deep_extra_conv64_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "deep_extra_conv64",
    },
    "a2_13": {
        "run_id": "a2_13_mean3ch_deep_two_extra_conv64_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "deep_two_extra_conv64",
    },
    "a2_14": {
        "run_id": "a2_14_mean3ch_wide_24_48_96_seed42",
        "input_config": "mean_3ch",
        "in_channels": 3,
        "model_variant": "wide_24_48_96",
    }
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run A2 model variant experiments."
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "a2_0",
            "a2_1",
            "a2_2",
            "a2_3",
            "a2_4",
            "a2_5",
            "a2_6",
            "a2_7",
            "a2_8",
            "a2_9",
            "a2_10",
            "a2_11",
            "a2_12",
            "a2_13",
            "a2_14",
            "all",
        ],
        default="a2_1"
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
        selected_names = ["a2_0", "a2_1", "a2_2", "a2_3", "a2_4", "a2_5", "a2_6", "a2_7", "a2_8", "a2_9", "a2_10", "a2_11", "a2_12", "a2_13", "a2_14"]
    else:
        selected_names = [arguments.experiment]

    summaries = []

    for experiment_name in selected_names:
        experiment = EXPERIMENTS[experiment_name]

        run_id = experiment["run_id"]
        model_variant = experiment["model_variant"]
        weight_decay = experiment.get("weight_decay", 0.0)
        use_lr_scheduler = experiment.get("use_lr_scheduler", False)
        lr_scheduler_factor = experiment.get("lr_scheduler_factor", 0.5)
        lr_scheduler_patience = experiment.get("lr_scheduler_patience", 1)
        min_learning_rate = experiment.get("min_learning_rate", 1e-5)

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
            preprocessing_dir=preprocessing_dir,
            model_variant=model_variant,
            weight_decay=weight_decay,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            min_learning_rate=min_learning_rate,
        )

        summaries.append(summary)

    comparison_df = pd.DataFrame(summaries).sort_values("overall_oof_log_loss", ascending=True).reset_index(drop=True)

    if arguments.experiment == "all":
        if (arguments.epochs == OFFICIAL_NUM_EPOCHS):
            comparison_name = "comparison_a2.csv"
        else:
            comparison_name = f"comparison_a2_{arguments.epochs}ep.csv"
            

        comparison_path = metric_dir / comparison_name

        comparison_df.to_csv(comparison_path, index=False)

        print()
        print(comparison_df)
        print("Comparison saved to:", comparison_path)

    best_run_id = comparison_df.loc[0, "run_id"]

    create_submission(run_id=best_run_id, prediction_dir=prediction_dir)


if __name__ == "__main__":
    main()
