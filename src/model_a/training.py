import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from data_pipeline import IcebergDataset,compute_channel_statistics
from model import (
    SimpleCNN,
    SimpleCNNDeepExtraConv64,
    SimpleCNNDeepTwoExtraConv64,
    SimpleCNNWide244896,
    SimpleCNNBatchNorm,
    SimpleCNNGAP,
    SimpleCNNBatchNormGAP,
    SimpleCNNCoarsePool,
    SimpleCNNCoarsePool5x5,
    SimpleCNNLessDownsample12,
    SimpleCNNLessDownsample12FC32,
    SimpleCNNLessDownsample10,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_one_fold(
    project_data,
    fold_number,
    input_config,
    in_channels,
    num_epochs,
    batch_size,
    learning_rate,
    base_seed,
    device,
    model_variant="control",
    weight_decay=0.0,
    use_lr_scheduler=False,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=1,
    min_learning_rate=1e-5,
    ):

    train_images = project_data["train_images"]
    test_images = project_data["test_images"]

    train_ids = project_data["train_ids"]
    test_ids = project_data["test_ids"]

    targets = project_data["targets"]
    fold_ids = project_data["fold_ids"]

    train_angle_raw = project_data["train_angle_raw"]
    test_angle_raw = project_data["test_angle_raw"]

    train_angle_missing = project_data["train_angle_missing"]
    test_angle_missing = project_data["test_angle_missing"]

    fold_seed = base_seed + fold_number
    seed_everything(fold_seed)

    train_mask = fold_ids != fold_number
    validation_mask = fold_ids == fold_number

    channel_mean, channel_std = compute_channel_statistics(canonical_images=train_images,row_mask=train_mask,input_config=input_config)

    train_dataset = IcebergDataset(
        images_2ch=train_images[train_mask],
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=train_ids[train_mask],
        angles=train_angle_raw[train_mask],
        angle_missing=(
            train_angle_missing[train_mask]
        ),
        targets=targets[train_mask],
        folds=fold_ids[train_mask]
    )

    validation_dataset = IcebergDataset(
        images_2ch=train_images[
            validation_mask
        ],
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=train_ids[validation_mask],
        angles=train_angle_raw[
            validation_mask
        ],
        angle_missing=(
            train_angle_missing[
                validation_mask
            ]
        ),
        targets=targets[validation_mask],
        folds=fold_ids[validation_mask]
    )

    test_dataset = IcebergDataset(
        images_2ch=test_images,
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=test_ids,
        angles=test_angle_raw,
        angle_missing=test_angle_missing,
        targets=None,
        folds=None
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(fold_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    if model_variant == "control":
        model = SimpleCNN(in_channels=in_channels)
    elif model_variant == "deep_extra_conv64":
        model = SimpleCNNDeepExtraConv64(in_channels=in_channels)
    elif model_variant == "deep_two_extra_conv64":
        model = SimpleCNNDeepTwoExtraConv64(in_channels=in_channels)
    elif model_variant == "wide_24_48_96":
        model = SimpleCNNWide244896(in_channels=in_channels)
    elif model_variant == "batchnorm":
        model = SimpleCNNBatchNorm(in_channels=in_channels)
    elif model_variant == "gap":
        model = SimpleCNNGAP(in_channels=in_channels)
    elif model_variant == "batchnorm_gap":
        model = SimpleCNNBatchNormGAP(in_channels=in_channels)
    elif model_variant == "coarse_pool_3x3":
        model = SimpleCNNCoarsePool(in_channels=in_channels)
    elif model_variant == "coarse_pool_5x5":
        model = SimpleCNNCoarsePool5x5(in_channels=in_channels)
    elif model_variant == "less_downsample_12x12":
        model = SimpleCNNLessDownsample12(in_channels=in_channels)
    elif model_variant == "less_downsample_12x12_fc32":
        model = SimpleCNNLessDownsample12FC32(in_channels=in_channels)
    elif model_variant == "less_downsample_10x10":
        model = SimpleCNNLessDownsample10(in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")

    model = model.to(device)

    print("Model variant:", model_variant)
    print("Model class:", model.__class__.__name__)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_learning_rate,
        )

    best_log_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_validation_probabilities = None

    for epoch in range(num_epochs):
        model.train()

        training_loss_sum = 0.0
        training_sample_count = 0

        for batch in train_loader:
            batch_images = batch["image"].to(device, non_blocking=True)

            batch_targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(batch_images)
            loss = criterion(logits, batch_targets)

            loss.backward()
            optimizer.step()

            batch_count = batch_images.size(0)

            training_loss_sum += loss.item() * batch_count
            

            training_sample_count += batch_count

        training_loss = training_loss_sum/training_sample_count
        model.eval()

        validation_probability_batches = []
        validation_target_batches = []

        with torch.no_grad():
            for batch in validation_loader:
                batch_images = batch["image"].to(device, non_blocking=True)
                batch_targets = batch["target"].to(device, non_blocking=True)

                logits = model(batch_images)
                probabilities = torch.sigmoid(logits)

                validation_probability_batches.append(probabilities.cpu().numpy())
                validation_target_batches.append(batch_targets.cpu().numpy())

        validation_probabilities = np.concatenate(validation_probability_batches).ravel()
        validation_targets = np.concatenate(validation_target_batches).ravel()

        current_log_loss = log_loss(validation_targets, validation_probabilities, labels=[0, 1])

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_epoch = epoch + 1

            best_validation_probabilities = (validation_probabilities.copy())

            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step(current_log_loss)

        next_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train loss: {training_loss:.6f} | "
            f"Validation Log Loss: {current_log_loss:.6f} | "
            f"Next LR: {next_learning_rate:.6g}"
        )

    model.load_state_dict(best_state)
    model.eval()

    test_probability_batches = []

    with torch.no_grad():
        for batch in test_loader:
            batch_images = batch["image"].to(device, non_blocking=True)

            logits = model(batch_images)

            probabilities = torch.sigmoid(logits)
            test_probability_batches.append(probabilities.cpu().numpy())

    test_probabilities = np.concatenate(test_probability_batches).ravel()

    return {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "fold_log_loss": best_log_loss,
        "validation_probabilities": (
            best_validation_probabilities
        ),
        "test_probabilities": (
            test_probabilities
        ),
        "channel_mean": channel_mean,
        "channel_std": channel_std
    }


def run_experiment(
    project_data,
    run_id,
    input_config,
    in_channels,
    num_epochs,
    batch_size,
    learning_rate,
    base_seed,
    device,
    prediction_dir,
    metric_dir,
    preprocessing_dir,
    model_variant="control",
    weight_decay=0.0,
    use_lr_scheduler=False,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=1,
    min_learning_rate=1e-5,
):
    train_images = project_data["train_images"]
    test_images = project_data["test_images"]

    train_ids = project_data["train_ids"]
    test_ids = project_data["test_ids"]

    targets = project_data["targets"]

    fold_ids = project_data["fold_ids"]

    oof_predictions = np.full(len(train_images), np.nan, dtype=np.float32)

    test_prediction_matrix = np.full((5, len(test_images)), np.nan, dtype=np.float32)

    fold_metric_rows = []
    preprocessing_rows = []

    for fold_number in range(5):
        print()
        print("=" * 70)
        print(f"Run: {run_id} | Fold: {fold_number}")
        print("=" * 70)

        fold_result = run_one_fold(
            project_data=project_data,
            fold_number=fold_number,
            input_config=input_config,
            in_channels=in_channels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            base_seed=base_seed,
            device=device,
            model_variant=model_variant,
            weight_decay=weight_decay,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            min_learning_rate=min_learning_rate,
        )

        validation_mask = (fold_ids == fold_number)

        oof_predictions[validation_mask] = fold_result["validation_probabilities"]

        test_prediction_matrix[fold_number] = fold_result["test_probabilities"]

        fold_metric_rows.append({
            "run_id": run_id,
            "input_config": input_config,
            "model_variant": model_variant,
            "weight_decay": weight_decay,
            "lr_scheduler": use_lr_scheduler,
            "lr_scheduler_factor": lr_scheduler_factor if use_lr_scheduler else None,
            "lr_scheduler_patience": lr_scheduler_patience if use_lr_scheduler else None,
            "min_learning_rate": min_learning_rate if use_lr_scheduler else None,
            "fold": fold_number,
            "best_epoch": fold_result["best_epoch"],
            "fold_log_loss": fold_result["fold_log_loss"]
        })

        preprocessing_rows.append({
            "run_id": run_id,
            "input_config": input_config,
            "model_variant": model_variant,
            "fold": fold_number,
            "channel_mean": json.dumps(fold_result["channel_mean"].tolist()),
            "channel_std": json.dumps(fold_result["channel_std"].tolist())
        })

    if np.isnan(oof_predictions).any():
        raise ValueError("OOF predictions are incomplete.")

    if np.isnan(test_prediction_matrix).any():
        raise ValueError("Test prediction matrix is incomplete.")

    mean_test_predictions = test_prediction_matrix.mean(axis=0)

    overall_oof_log_loss = log_loss(targets, oof_predictions, labels=[0, 1])

    model_name = f"simple_cnn_{model_variant}"

    oof_df = pd.DataFrame({
        "id": train_ids,
        "fold": fold_ids,
        "target": targets.astype(np.int64),
        "prediction": oof_predictions,
        "model_name": model_name,
        "run_id": run_id
    })

    test_prediction_df = pd.DataFrame({
        "id": test_ids,
        "prediction": mean_test_predictions,
        "model_name": model_name,
        "run_id": run_id,
    })

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    preprocessing_df = pd.DataFrame(preprocessing_rows)

    summary = {
        "run_id": run_id,
        "input_config": input_config,
        "in_channels": in_channels,
        "model_variant": model_variant,
        "weight_decay": weight_decay,
        "lr_scheduler": use_lr_scheduler,
        "lr_scheduler_factor": lr_scheduler_factor if use_lr_scheduler else None,
        "lr_scheduler_patience": lr_scheduler_patience if use_lr_scheduler else None,
        "min_learning_rate": min_learning_rate if use_lr_scheduler else None,
        "overall_oof_log_loss": (
            overall_oof_log_loss
        ),
    }

    summary_df = pd.DataFrame([summary])

    oof_df.to_csv(prediction_dir / f"oof_{run_id}.csv", index=False)

    test_prediction_df.to_csv(prediction_dir / f"test_{run_id}.csv", index=False)

    fold_metrics_df.to_csv(metric_dir / f"folds_{run_id}.csv", index=False)

    summary_df.to_csv(metric_dir / f"summary_{run_id}.csv", index=False)

    preprocessing_df.to_csv(preprocessing_dir / f"stats_{run_id}.csv", index=False)

    print()
    print("Run completed:", run_id)
    print("Overall OOF Log Loss:", overall_oof_log_loss)

    return summary
