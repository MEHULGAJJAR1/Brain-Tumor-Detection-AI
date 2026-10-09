"""ResNet50 classifier architecture used by the optional training workflow."""
from __future__ import annotations


def build_resnet50(num_classes: int = 3, pretrained: bool = False):
    """Build the reference ResNet50 backbone and 2048→512→classes head.

    PyTorch and torchvision are imported lazily so the base web app remains lightweight.
    """
    try:
        from torch import nn
        from torchvision import models
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "The optional ML dependencies are unavailable. Install requirements-ml.txt."
        ) from exc

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    network = models.resnet50(weights=weights)
    network.fc = nn.Sequential(
        nn.Linear(network.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    return network
