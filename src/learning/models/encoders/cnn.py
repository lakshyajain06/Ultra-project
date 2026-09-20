"""Compact convolutional image encoder."""

from torch import nn


class CNNImageEncoder(nn.Module):
    """Encode a low-resolution RGB image as one policy token."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512, output_dim),
        )

    def forward(self, image):
        return self.net(image)
