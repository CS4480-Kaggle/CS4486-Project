from pathlib import Path
import argparse

import pandas as pd
import torch

from data_pipeline import load_project_data
from training import run_experiment


SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.3
DEFAULT_NUM_EPOCHS = 15

EXPERIMENTS = {
    "r0": {"label": "image_only", "metadata_dim": 0},
    "r1": {"label": "image_angle", "metadata_dim": 1},
    "r2": {"label": "image_angle_missing", "metadata_dim": 2},
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ResNet18 iceberg experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=["r0", "r1", "r2", "all"],
        default="r0",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adamw",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "flip"],
        default="none",
    )
    parser.add_argument("--pretrained", action="store_true")

    stem_group = parser.add_mutually_exclusive_group()
    stem_group.add_argument("--standard-stem", action="store_true")
    stem_group.add_argument("--high-resolution-stem", action="store_true")

    parser.add_argument(
        "--tta",
        choices=["none", "flip4"],
        default="none",
    )
    parser.add_argument("--run-tag", type=str, default="")

    arguments = parser.parse_args()

    if arguments.high_resolution_stem and not arguments.pretrained:
        parser.error("--high-resolution-stem requires --pretrained")

    return arguments


def get_stem_label(arguments):
    if arguments.high_resolution_stem:
        return "highres"
    if arguments.standard_stem:
        return "standard"
    return "small"


def format_value(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


def build_run_id(experiment_name, experiment, arguments):
    parts = [
        experiment_name,
        "resnet18",
        get_stem_label(arguments),
        "pretrained" if arguments.pretrained else "random",
        experiment["label"],
        f"lr{format_value(arguments.learning_rate)}",
        f"wd{format_value(arguments.weight_decay)}",
        f"do{format_value(arguments.dropout)}",
        arguments.optimizer,
        f"aug{arguments.augmentation}",
        f"tta{arguments.tta}",
        f"{arguments.epochs}ep",
        f"seed{SEED}",
    ]

    if arguments.run_tag:
        parts.append(arguments.run_tag)

    return "_".join(parts)


def create_submission(run_id, prediction_dir):
    prediction_path = prediction_dir / f"test_{run_id}.csv"
    prediction_df = pd.read_csv(prediction_path)
    submission_df = prediction_df[["id", "prediction"]].rename(
        columns={"prediction": "is_iceberg"}
    )
    submission_path = prediction_dir / f"submission_{run_id}.csv"
    submission_df.to_csv(submission_path, index=False)
    print("Submission saved to:", submission_path)


def update_comparison(metric_dir):
    summary_paths = sorted(metric_dir.glob("summary_r*_resnet18_*.csv"))
    summaries = [pd.read_csv(path) for path in summary_paths]
    comparison_df = pd.concat(summaries, ignore_index=True)
    comparison_df = comparison_df.sort_values(
        "overall_oof_log_loss",
        ascending=True,
    ).reset_index(drop=True)
    comparison_path = metric_dir / "comparison_resnet18.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("ResNet comparison saved to:", comparison_path)
    return comparison_df


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
    selected_names = (
        ["r0", "r1", "r2"]
        if arguments.experiment == "all"
        else [arguments.experiment]
    )

    current_summaries = []

    for experiment_name in selected_names:
        experiment = EXPERIMENTS[experiment_name]
        run_id = build_run_id(experiment_name, experiment, arguments)

        summary = run_experiment(
            project_data=project_data,
            run_id=run_id,
            input_config="mean_3ch",
            in_channels=3,
            num_epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            base_seed=SEED,
            device=device,
            prediction_dir=prediction_dir,
            metric_dir=metric_dir,
            preprocessing_dir=preprocessing_dir,
            model_name=(
                "resnet18_pretrained"
                if arguments.pretrained
                else "resnet18_random"
            ),
            metadata_dim=experiment["metadata_dim"],
            model_arch="resnet18",
            optimizer_name=arguments.optimizer,
            weight_decay=arguments.weight_decay,
            dropout=arguments.dropout,
            pretrained=arguments.pretrained,
            small_stem=not (
                arguments.standard_stem
                or arguments.high_resolution_stem
            ),
            high_resolution_stem=arguments.high_resolution_stem,
            augmentation=arguments.augmentation,
            tta=arguments.tta,
        )
        current_summaries.append(summary)

    current_comparison = pd.DataFrame(current_summaries).sort_values(
        "overall_oof_log_loss",
        ascending=True,
    ).reset_index(drop=True)

    print()
    print("Current run comparison:")
    print(current_comparison)

    best_run_id = current_comparison.loc[0, "run_id"]
    create_submission(best_run_id, prediction_dir)
    update_comparison(metric_dir)


if __name__ == "__main__":
    main()
