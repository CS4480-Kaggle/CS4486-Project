import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        if in_channels not in {2, 3}:
            raise ValueError(f"Unsupported input channel count: {in_channels}")

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=3,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 9 * 9, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNWithMetadata(nn.Module):
    def __init__(self, in_channels, metadata_dim):
        super().__init__()

        if metadata_dim not in {0, 1, 2}:
            raise ValueError("metadata_dim must be 0, 1, or 2.")

        self.metadata_dim = metadata_dim

        base_model = SimpleCNN(in_channels=in_channels)

        self.features = base_model.features

        self.flatten = base_model.classifier[0]
        self.embedding_linear = base_model.classifier[1]
        self.embedding_relu = base_model.classifier[2]
        self.dropout = base_model.classifier[3]

        base_output_layer = base_model.classifier[4]

        if metadata_dim == 0:
            self.output_layer = base_output_layer
            return

        self.output_layer = nn.Linear(64 + metadata_dim, 1)

        with torch.no_grad():
            self.output_layer.weight[:, :64].copy_(base_output_layer.weight)
            self.output_layer.weight[:, 64:].zero_()
            self.output_layer.bias.copy_(base_output_layer.bias)

    def forward(self, image, angle=None, angle_missing=None):
        features = self.features(image)
        features = self.flatten(features)
        features = self.embedding_linear(features)
        features = self.embedding_relu(features)
        image_embedding = self.dropout(features)

        fusion_parts = [image_embedding]

        if self.metadata_dim >= 1:
            if angle is None:
                raise ValueError("The model requires normalized angle input.")
            fusion_parts.append(angle)

        if self.metadata_dim == 2:
            if angle_missing is None:
                raise ValueError("The model requires angle_missing input.")
            fusion_parts.append(angle_missing)

        logits = self.output_layer(torch.cat(fusion_parts, dim=1))
        return logits


class ResNet18WithMetadata(nn.Module):
    def __init__(
        self,
        in_channels,
        metadata_dim=0,
        pretrained=False,
        small_stem=True,
        high_resolution_stem=False,
        dropout=0.3,
    ):
        super().__init__()

        if metadata_dim not in {0, 1, 2}:
            raise ValueError("metadata_dim must be 0, 1, or 2.")

        if high_resolution_stem:
            if not pretrained:
                raise ValueError("High-resolution stem requires pretrained weights.")

            if small_stem:
                raise ValueError("High-resolution stem cannot be combined with small stem.")

            if in_channels != 3:
                raise ValueError("High-resolution stem requires three input channels.")

        if pretrained and (small_stem or in_channels != 3):
            raise ValueError("Pretrained weights require the standard stem and three input channels.")

        self.metadata_dim = metadata_dim
        self.high_resolution_stem = high_resolution_stem
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        if small_stem:
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.backbone.maxpool = nn.Identity()
        elif high_resolution_stem:
            self.backbone.conv1.stride = (1, 1)
            self.backbone.maxpool = nn.Identity()
        elif in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

        base_output_layer = nn.Linear(embedding_dim, 1)

        if metadata_dim == 0:
            self.output_layer = base_output_layer
        else:
            self.output_layer = nn.Linear(embedding_dim + metadata_dim, 1)

            with torch.no_grad():
                self.output_layer.weight[:, :embedding_dim].copy_(base_output_layer.weight)
                self.output_layer.weight[:, embedding_dim:].zero_()
                self.output_layer.bias.copy_(base_output_layer.bias)

    def forward(self, image, angle=None, angle_missing=None):
        image_embedding = self.dropout(self.backbone(image))
        fusion_parts = [image_embedding]

        if self.metadata_dim >= 1:
            if angle is None:
                raise ValueError("The model requires normalized angle input.")
            fusion_parts.append(angle)

        if self.metadata_dim == 2:
            if angle_missing is None:
                raise ValueError("The model requires angle_missing input.")
            fusion_parts.append(angle_missing)

        logits = self.output_layer(torch.cat(fusion_parts, dim=1))
        return logits
