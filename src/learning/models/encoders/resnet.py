"""Torchvision ResNet image encoder."""

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

_RESNETS = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
}


class ResNetImageEncoder(nn.Module):
    """Encode RGB images with a ResNet-18, -34, or -50 backbone.

    Set ``weights`` to ``"DEFAULT"`` to initialize from torchvision's
    ImageNet weights. In that case inputs in [0, 1] are normalized with the
    statistics attached to those weights. ``None`` trains from scratch and
    leaves inputs unchanged.
    """

    def __init__(self, output_dim: int, variant: str = "resnet18", weights: str | None = None):
        super().__init__()
        if variant not in _RESNETS:
            raise ValueError(f"Unknown ResNet variant {variant!r}; available: {sorted(_RESNETS)}")
        if weights not in (None, "DEFAULT"):
            raise ValueError("ResNet weights must be null or 'DEFAULT'")

        factory, weights_type = _RESNETS[variant]
        resolved_weights = weights_type.DEFAULT if weights == "DEFAULT" else None
        self.net = factory(weights=resolved_weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, output_dim)

        if resolved_weights is None:
            mean, std = (), ()
        else:
            transforms = resolved_weights.transforms()
            mean, std = transforms.mean, transforms.std
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, image):
        if self.mean.numel():
            image = (image - self.mean) / self.std
        return self.net(image)
