import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


IMAGE_HEIGHT = 75
IMAGE_WIDTH = 75
PIXELS_PER_IMAGE = IMAGE_HEIGHT * IMAGE_WIDTH

def build_canonical_images(df):
    band_1_lengths = df["band_1"].map(len)
    band_2_lengths = df["band_2"].map(len)

    if not (band_1_lengths == PIXELS_PER_IMAGE).all():
        raise ValueError("Invalid band_1 length detected.")

    if not (band_2_lengths == PIXELS_PER_IMAGE).all():
        raise ValueError("Invalid band_2 length detected.")

    band_1 = np.stack(df["band_1"].to_numpy()).astype(np.float32)
    band_2 = np.stack(df["band_2"].to_numpy()).astype(np.float32)

    band_1 = band_1.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)
    band_2 = band_2.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)

    images = np.stack([band_1, band_2], axis=1)
    images = np.ascontiguousarray(images, dtype=np.float32)

    if not np.isfinite(images).all():
        raise ValueError("Image data contains NaN or infinite values.")

    return images



def build_input_batch(images_2ch, input_config):
    
    images_2ch = np.asarray(images_2ch, dtype=np.float32)

    if input_config == "raw_2ch":
        output = images_2ch

    elif input_config == "mean_3ch":
        band_1 = images_2ch[:, 0:1]
        band_2 = images_2ch[:, 1:2]
        mean_channel = (band_1 + band_2) * np.float32(0.5)
        output = np.concatenate([band_1, band_2, mean_channel], axis=1)

    elif input_config == "difference_3ch":
        band_1 = images_2ch[:, 0:1]
        band_2 = images_2ch[:, 1:2]
        difference_channel = band_1 - band_2
        output = np.concatenate([band_1, band_2, difference_channel], axis=1)

    else:
        raise ValueError(f"Unsupported input configuration: {input_config}")

    output = np.ascontiguousarray(output, dtype=np.float32)

    return output


def build_input_image(image_2ch, input_config):
    image_2ch = np.asarray(image_2ch, dtype=np.float32)
    return build_input_batch(image_2ch[np.newaxis, ...], input_config)[0]



def compute_channel_statistics(canonical_images, row_mask,input_config):
    
    enhanced_images = build_input_batch(canonical_images[row_mask], input_config)

    channel_mean = enhanced_images.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    channel_std = enhanced_images.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)

    if (channel_std <= 1e-8).any():
        raise ValueError("A channel has near-zero standard deviation.")

    return channel_mean, channel_std

def load_project_data(train_path, test_path, fold_path):
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)
    fold_df = pd.read_csv(fold_path)

    train_images = build_canonical_images(train_df)
    test_images = build_canonical_images(test_df)

    train_ids = train_df["id"].astype(str).to_numpy()
    test_ids = test_df["id"].astype(str).to_numpy()

    targets = train_df["is_iceberg"].to_numpy(dtype=np.float32)

    train_angle_raw = pd.to_numeric(train_df["inc_angle"], errors="coerce").to_numpy(dtype=np.float32)
    test_angle_raw = pd.to_numeric(test_df["inc_angle"], errors="coerce").to_numpy(dtype=np.float32)

    train_angle_missing = np.isnan(train_angle_raw)
    test_angle_missing = np.isnan(test_angle_raw)

    fold_df["id"] = fold_df["id"].astype(str)
    aligned_folds = fold_df.set_index("id").loc[train_ids]
    fold_ids = aligned_folds["fold"].to_numpy(dtype=np.int64)

    return {
        "train_images": train_images,
        "test_images": test_images,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "targets": targets,
        "fold_ids": fold_ids,
        "train_angle_raw": train_angle_raw,
        "test_angle_raw": test_angle_raw,
        "train_angle_missing": train_angle_missing,
        "test_angle_missing": test_angle_missing,
    }

class IcebergDataset(Dataset):
    def __init__(
        self,
        images_2ch,
        input_config,
        channel_mean,
        channel_std,
        ids,
        angles,
        angle_missing,
        targets=None,
        folds=None
    ):
        self.images_2ch = images_2ch
        self.input_config = input_config

        self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
        self.channel_std = np.asarray(channel_std, dtype=np.float32)

        self.ids = np.asarray(ids)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.angle_missing = np.asarray(angle_missing, dtype=bool)
        self.targets = targets
        self.folds = folds

    def __len__(self):
        return len(self.images_2ch)

    def __getitem__(self, index):
        image = build_input_image(self.images_2ch[index], self.input_config)
        image = (image - self.channel_mean[:, None, None]) / self.channel_std[:, None, None]
        image = np.ascontiguousarray(image, dtype=np.float32)

        sample = {
            "image": torch.from_numpy(image),
            "angle": torch.tensor([self.angles[index]], dtype=torch.float32),
            "angle_missing": torch.tensor([float(self.angle_missing[index])], dtype=torch.float32),
            "id": str(self.ids[index]),
        }

        if self.targets is not None:
            sample["target"] = torch.tensor([self.targets[index]], dtype=torch.float32)

        if self.folds is not None:
            sample["fold"] = int(self.folds[index])

        return sample
    


