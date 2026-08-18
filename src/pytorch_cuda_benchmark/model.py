# pyright: reportMissingTypeStubs=false

"""CIFAR-10 model construction utilities."""

import torch.nn as nn
from torchvision.models import resnet18

_CIFAR10_CLASS_COUNT = 10


def create_cifar10_resnet18() -> nn.Module:
    """Create a ResNet-18 adapted for 32x32 CIFAR-10 images."""

    model = resnet18(
        weights=None,
        num_classes=_CIFAR10_CLASS_COUNT,
    )

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    nn.init.kaiming_normal_(
        model.conv1.weight,
        mode="fan_out",
        nonlinearity="relu",
    )

    setattr(model, "maxpool", nn.Identity())

    return model
