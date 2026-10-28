import torch
import torch.nn as nn


def conv_bn(in_channels, out_channels):
    """3x3 convolution -> batch norm -> ReLU. Padding keeps the image size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolution block: 1x28x28 -> 32x14x14
        self.block1 = nn.Sequential(
            conv_bn(1, 32),
            conv_bn(32, 32),
            nn.MaxPool2d(2, 2),
        )

        # Second convolution block: 32x14x14 -> 64x7x7
        self.block2 = nn.Sequential(
            conv_bn(32, 64),
            conv_bn(64, 64),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers, with dropout so the model can't lean on
        # any single feature (helps with inputs that don't look like MNIST)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):

        # Convolution blocks
        x = self.block1(x)
        x = self.block2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.classifier(x)

        return x
