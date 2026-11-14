from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "model_b"
sys.path.insert(0, str(SRC_DIR))

from data_pipeline import IcebergDataset, build_input_image


class TestTrainingAugmentation(unittest.TestCase):
    def test_flip_augmentation_changes_only_spatial_orientation(self):
        images = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        expected = build_input_image(images[0], "mean_3ch")
        expected = np.flip(expected, axis=2).copy()

        dataset = IcebergDataset(
            images_2ch=images,
            input_config="mean_3ch",
            channel_mean=np.zeros(3, dtype=np.float32),
            channel_std=np.ones(3, dtype=np.float32),
            ids=np.array(["sample"]),
            angles=np.array([0.0], dtype=np.float32),
            angle_missing=np.array([False]),
            targets=np.array([1.0], dtype=np.float32),
            augment=True,
        )

        with patch("numpy.random.random", side_effect=[0.1, 0.9]):
            actual = dataset[0]["image"].numpy()

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
