import torch.nn as nn

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


class SimpleCNNDeepExtraConv64(nn.Module):
    """A2-12: A0 with one extra 64-to-64 convolution before the final pool."""

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

            nn.Conv2d(
                in_channels=64,
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


class SimpleCNNDeepTwoExtraConv64(nn.Module):
    """A2-13: A0 with two extra 64-to-64 convolutions before the final pool."""

    def __init__(self, in_channels):
        super().__init__()

        if in_channels not in {2, 3}:
            raise ValueError(f"Unsupported input channel count: {in_channels}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNWide244896(nn.Module):
    """A2-14: A0 widened from 16-32-64 channels to 24-48-96."""

    def __init__(self, in_channels):
        super().__init__()

        if in_channels not in {2, 3}:
            raise ValueError(f"Unsupported input channel count: {in_channels}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 9 * 9, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNBatchNorm(nn.Module):
    """A2-1: SimpleCNN with BatchNorm2d after each convolution, before ReLU."""

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
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(num_features=64),
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


class SimpleCNNGAP(nn.Module):
    """A2-2: SimpleCNN with a global average pooling classifier head."""

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
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNCoarsePool(nn.Module):
    """A2-4: SimpleCNN with a 3x3 adaptive average pooling classifier head."""

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
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(in_features=64 * 3 * 3, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNCoarsePool5x5(nn.Module):
    """A2-5: SimpleCNN with a 5x5 adaptive average pooling classifier head."""

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
            nn.AdaptiveAvgPool2d(output_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(in_features=64 * 5 * 5, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNLessDownsample12(nn.Module):
    """A2-6: SimpleCNN that keeps a 12x12 feature map before the classifier."""

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
            nn.AdaptiveMaxPool2d(output_size=(12, 12))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 12 * 12, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNLessDownsample12FC32(nn.Module):
    """A2-7: 12x12 less-downsampling model with a 32-unit FC bottleneck."""

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
            nn.AdaptiveMaxPool2d(output_size=(12, 12))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 12 * 12, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNLessDownsample10(nn.Module):
    """A2-10: SimpleCNN that keeps a 10x10 feature map before FC64."""

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
            nn.AdaptiveMaxPool2d(output_size=(10, 10))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 10 * 10, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits


class SimpleCNNBatchNormGAP(nn.Module):
    """A2-3: SimpleCNN with BatchNorm2d and a global average pooling head."""

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
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, image):
        features = self.features(image)
        logits = self.classifier(features)
        return logits
