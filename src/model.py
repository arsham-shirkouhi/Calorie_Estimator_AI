from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_efficientnet_b0(num_classes: int = 101, pretrained: bool = False) -> nn.Module:
    """Build an EfficientNet-B0 classifier for Food-101 classes."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, num_classes),
)
    return model
