from __future__ import annotations

from ...dl.layers import Flatten, Linear
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, DepthwiseSeparableConv2D, GlobalAvgPool2D
from ..batchnorm2d import BatchNorm2D


def MobileNetV1(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """MobileNetV1 using Depthwise Separable Convolutions from Scratch."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(32))
    model.add(ReLU())

    # Depthwise Separable Blocks
    model.add(DepthwiseSeparableConv2D(32, 64, stride=1))
    model.add(BatchNorm2D(64))
    model.add(ReLU())

    model.add(DepthwiseSeparableConv2D(64, 128, stride=2))
    model.add(BatchNorm2D(128))
    model.add(ReLU())

    model.add(DepthwiseSeparableConv2D(128, 256, stride=2))
    model.add(BatchNorm2D(256))
    model.add(ReLU())

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(256, num_classes))
    return model
