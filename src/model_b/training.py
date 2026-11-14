import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from data_pipeline import IcebergDataset,compute_channel_statistics, fit_angle_preprocessor, transform_angles
from model import ResNet18WithMetadata, SimpleCNN, SimpleCNNWithMetadata


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict_batch_probabilities(
    model,
    batch_images,
    batch_angles,
    batch_angle_missing,
    metadata_dim,
    tta,
):
    if tta == "none":
        image_views = [batch_images]
    elif tta == "flip4":
        image_views = [
            batch_images,
            torch.flip(batch_images, dims=[3]),
            torch.flip(batch_images, dims=[2]),
            torch.flip(batch_images, dims=[2, 3]),
        ]
    else:
        raise ValueError(f"Unsupported TTA configuration: {tta}")

    probability_views = []

    for image_view in image_views:
        if metadata_dim is None:
            logits = model(image_view)
        else:
            logits = model(
                image_view,
                batch_angles,
                batch_angle_missing,
            )

        probability_views.append(torch.sigmoid(logits))

    return torch.stack(probability_views, dim=0).mean(dim=0)


def predict_loader(
    model,
    data_loader,
    device,
    metadata_dim,
    tta,
    include_targets,
):
    model.eval()

    probability_batches = []
    target_batches = []

    with torch.no_grad():
        for batch in data_loader:
            batch_images = batch["image"].to(device, non_blocking=True)

            if metadata_dim is None:
                batch_angles = None
                batch_angle_missing = None
            else:
                batch_angles = batch["angle"].to(
                    device,
                    non_blocking=True,
                )
                batch_angle_missing = batch["angle_missing"].to(
                    device,
                    non_blocking=True,
                )

            probabilities = predict_batch_probabilities(
                model=model,
                batch_images=batch_images,
                batch_angles=batch_angles,
                batch_angle_missing=batch_angle_missing,
                metadata_dim=metadata_dim,
                tta=tta,
            )
            probability_batches.append(probabilities.cpu().numpy())

            if include_targets:
                target_batches.append(batch["target"].cpu().numpy())

    all_probabilities = np.concatenate(probability_batches).ravel()

    if not include_targets:
        return all_probabilities, None

    all_targets = np.concatenate(target_batches).ravel()
    return all_probabilities, all_targets


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
    metadata_dim=None,
    model_arch="simple_cnn",
    optimizer_name="adam",
    weight_decay=0.0,
    dropout=0.3,
    pretrained=False,
    small_stem=True,
    high_resolution_stem=False,
    augmentation="none",
    tta="none",
):
    if tta not in {"none", "flip4"}:
        raise ValueError(f"Unsupported TTA configuration: {tta}")

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

    train_mask = (fold_ids != fold_number)
    validation_mask = (fold_ids == fold_number)

    angle_statistics = None

    train_angles_fold = train_angle_raw[train_mask]
    validation_angles_fold = train_angle_raw[validation_mask]

    train_angle_missing_fold = train_angle_missing[train_mask]
    validation_angle_missing_fold = train_angle_missing[validation_mask]

    if metadata_dim is None:
        training_angles = train_angles_fold
        validation_angles = validation_angles_fold
        testing_angles = test_angle_raw

        training_angle_missing = train_angle_missing_fold
        validation_angle_missing = validation_angle_missing_fold
        testing_angle_missing = test_angle_missing
    else:
        angle_statistics = fit_angle_preprocessor(train_angles_fold)

        training_angle_data = transform_angles(train_angles_fold, angle_statistics)
        validation_angle_data = transform_angles(validation_angles_fold, angle_statistics)
        testing_angle_data = transform_angles(test_angle_raw, angle_statistics)

        training_angles = training_angle_data["angle_normalized"]
        validation_angles = validation_angle_data["angle_normalized"]
        testing_angles = testing_angle_data["angle_normalized"]

        training_angle_missing = training_angle_data["angle_missing"]
        validation_angle_missing = validation_angle_data["angle_missing"]
        testing_angle_missing = testing_angle_data["angle_missing"]

    channel_mean, channel_std = compute_channel_statistics(
        canonical_images=train_images,
        row_mask=train_mask,
        input_config=input_config
    )

    train_dataset = IcebergDataset(
        images_2ch=train_images[train_mask],
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=train_ids[train_mask],
        angles=training_angles,
        angle_missing=training_angle_missing,
        targets=targets[train_mask],
        folds=fold_ids[train_mask],
        augment=augmentation == "flip"
    )

    validation_dataset = IcebergDataset(
        images_2ch=train_images[validation_mask],
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=train_ids[validation_mask],
        angles=validation_angles,
        angle_missing=validation_angle_missing,
        targets=targets[validation_mask],
        folds=fold_ids[validation_mask]
    )

    test_dataset = IcebergDataset(
        images_2ch=test_images,
        input_config=input_config,
        channel_mean=channel_mean,
        channel_std=channel_std,
        ids=test_ids,
        angles=testing_angles,
        angle_missing=testing_angle_missing,
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


    if model_arch == "simple_cnn":
        if metadata_dim is None:
            model = SimpleCNN(in_channels=in_channels)
        else:
            model = SimpleCNNWithMetadata(
                in_channels=in_channels,
                metadata_dim=metadata_dim
            )
    elif model_arch == "resnet18":
        model = ResNet18WithMetadata(
            in_channels=in_channels,
            metadata_dim=0 if metadata_dim is None else metadata_dim,
            pretrained=pretrained,
            small_stem=small_stem,
            high_resolution_stem=high_resolution_stem,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    model = model.to(device)
    

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_log_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(num_epochs):
        model.train()

        training_loss_sum = 0.0
        training_sample_count = 0

        for batch in train_loader:
            batch_images = batch["image"].to(device, non_blocking=True)
            batch_targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if metadata_dim is None:
                logits = model(batch_images)
            else:
                batch_angles = batch["angle"].to(device, non_blocking=True)
                batch_angle_missing = batch["angle_missing"].to(device, non_blocking=True)
                logits = model(batch_images, batch_angles, batch_angle_missing)

            loss = criterion(logits, batch_targets)

            loss.backward()
            optimizer.step()

            batch_count = batch_images.size(0)
            training_loss_sum += loss.item() * batch_count
            training_sample_count += batch_count

        training_loss = training_loss_sum / training_sample_count
        validation_probabilities, validation_targets = predict_loader(
            model=model,
            data_loader=validation_loader,
            device=device,
            metadata_dim=metadata_dim,
            tta="none",
            include_targets=True,
        )

        current_log_loss = log_loss(validation_targets, validation_probabilities, labels=[0, 1])

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_epoch = epoch + 1
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train loss: {training_loss:.6f} | "
            f"Validation Log Loss: {current_log_loss:.6f}"
        )

    model.load_state_dict(best_state)

    final_validation_probabilities, final_validation_targets = predict_loader(
        model=model,
        data_loader=validation_loader,
        device=device,
        metadata_dim=metadata_dim,
        tta=tta,
        include_targets=True,
    )
    final_validation_log_loss = log_loss(
        final_validation_targets,
        final_validation_probabilities,
        labels=[0, 1],
    )

    test_probabilities, _ = predict_loader(
        model=model,
        data_loader=test_loader,
        device=device,
        metadata_dim=metadata_dim,
        tta=tta,
        include_targets=False,
    )

    return {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "selection_log_loss": best_log_loss,
        "fold_log_loss": final_validation_log_loss,
        "validation_probabilities": final_validation_probabilities,
        "test_probabilities": test_probabilities,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "angle_statistics": angle_statistics
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
    model_name="simple_cnn_i1",
    metadata_dim=None,
    model_arch="simple_cnn",
    optimizer_name="adam",
    weight_decay=0.0,
    dropout=0.3,
    pretrained=False,
    small_stem=True,
    high_resolution_stem=False,
    augmentation="none",
    tta="none",
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
    angle_preprocessing_rows = []

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
            metadata_dim=metadata_dim,
            model_arch=model_arch,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            dropout=dropout,
            pretrained=pretrained,
            small_stem=small_stem,
            high_resolution_stem=high_resolution_stem,
            augmentation=augmentation,
            tta=tta,
        )

        validation_mask = fold_ids == fold_number

        oof_predictions[validation_mask] = fold_result["validation_probabilities"]
        test_prediction_matrix[fold_number] = fold_result["test_probabilities"]

        fold_metric_rows.append({
            "run_id": run_id,
            "input_config": input_config,
            "fold": fold_number,
            "best_epoch": fold_result["best_epoch"],
            "selection_log_loss": fold_result["selection_log_loss"],
            "fold_log_loss": fold_result["fold_log_loss"]
        })

        preprocessing_rows.append({
            "run_id": run_id,
            "input_config": input_config,
            "fold": fold_number,
            "channel_mean": json.dumps(fold_result["channel_mean"].tolist()),
            "channel_std": json.dumps(fold_result["channel_std"].tolist())
        })

        angle_statistics = fold_result["angle_statistics"]

        if angle_statistics is not None:
            angle_preprocessing_rows.append({
                "run_id": run_id,
                "fold": fold_number,
                "angle_median": angle_statistics["angle_median"],
                "angle_mean": angle_statistics["angle_mean"],
                "angle_std": angle_statistics["angle_std"]
            })

    if np.isnan(oof_predictions).any():
        raise ValueError("OOF predictions are incomplete.")

    if np.isnan(test_prediction_matrix).any():
        raise ValueError("Test prediction matrix is incomplete.")

    mean_test_predictions = test_prediction_matrix.mean(axis=0)
    overall_oof_log_loss = log_loss(targets, oof_predictions, labels=[0, 1])

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
        "run_id": run_id
    })

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    preprocessing_df = pd.DataFrame(preprocessing_rows)

    summary = {
        "run_id": run_id,
        "input_config": input_config,
        "metadata_dim": metadata_dim,
        "model_name": model_name,
        "model_arch": model_arch,
        "in_channels": in_channels,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "pretrained": pretrained,
        "small_stem": small_stem,
        "high_resolution_stem": high_resolution_stem,
        "augmentation": augmentation,
        "tta": tta,
        "base_seed": base_seed,
        "overall_oof_log_loss": overall_oof_log_loss
    }

    summary_df = pd.DataFrame([summary])

    oof_df.to_csv(prediction_dir / f"oof_{run_id}.csv", index=False)
    test_prediction_df.to_csv(prediction_dir / f"test_{run_id}.csv", index=False)
    fold_metrics_df.to_csv(metric_dir / f"folds_{run_id}.csv", index=False)
    summary_df.to_csv(metric_dir / f"summary_{run_id}.csv", index=False)
    preprocessing_df.to_csv(preprocessing_dir / f"stats_{run_id}.csv", index=False)

    if angle_preprocessing_rows:
        angle_preprocessing_df = pd.DataFrame(angle_preprocessing_rows)
        angle_preprocessing_df.to_csv(
            preprocessing_dir / f"angle_stats_{run_id}.csv",
            index=False
        )

    print()
    print("Run completed:", run_id)
    print("Overall OOF Log Loss:", overall_oof_log_loss)

    return summary
