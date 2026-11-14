from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "model_b"
sys.path.insert(0, str(SRC_DIR))

from model import ResNet18WithMetadata


class TestResNet18WithMetadata(unittest.TestCase):
    def test_returns_one_logit_per_sample(self):
        images = torch.randn(2, 3, 75, 75)
        angles = torch.randn(2, 1)
        angle_missing = torch.zeros(2, 1)

        for metadata_dim in (0, 1, 2):
            with self.subTest(metadata_dim=metadata_dim):
                model = ResNet18WithMetadata(
                    in_channels=3,
                    metadata_dim=metadata_dim,
                    pretrained=False,
                    small_stem=True,
                    dropout=0.2,
                ).eval()

                with torch.no_grad():
                    logits = model(images, angles, angle_missing)

                self.assertEqual(logits.shape, (2, 1))

    def test_small_stem_preserves_more_spatial_resolution(self):
        model = ResNet18WithMetadata(
            in_channels=3,
            metadata_dim=0,
            pretrained=False,
            small_stem=True,
            dropout=0.2,
        )

        self.assertEqual(model.backbone.conv1.kernel_size, (3, 3))
        self.assertEqual(model.backbone.conv1.stride, (1, 1))
        self.assertIsInstance(model.backbone.maxpool, nn.Identity)

    def test_metadata_models_start_from_the_same_image_prediction(self):
        images = torch.randn(2, 3, 75, 75)
        angles = torch.randn(2, 1)
        angle_missing = torch.ones(2, 1)

        torch.manual_seed(42)
        image_only_model = ResNet18WithMetadata(
            in_channels=3,
            metadata_dim=0,
            pretrained=False,
            small_stem=True,
            dropout=0.0,
        ).eval()

        torch.manual_seed(42)
        multimodal_model = ResNet18WithMetadata(
            in_channels=3,
            metadata_dim=2,
            pretrained=False,
            small_stem=True,
            dropout=0.0,
        ).eval()

        with torch.no_grad():
            image_only_logits = image_only_model(images)
            multimodal_logits = multimodal_model(
                images,
                angles,
                angle_missing,
            )

        self.assertTrue(
            torch.allclose(
                image_only_logits,
                multimodal_logits,
                atol=1e-7,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
