from __future__ import annotations

from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D, MaxPool2D
from ..batchnorm2d import BatchNorm2D


class ResidualBlock(Module):
    """Basic Residual Skip-Connection Block: F(x) + x."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = Conv2D(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = BatchNorm2D(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2D(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = BatchNorm2D(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
            self.shortcut_bn = BatchNorm2D(out_channels)
        else:
            self.shortcut = None
            self.shortcut_bn = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = self.shortcut_bn(identity)

        out = out + identity
        return self.relu(out)


def ResNet18(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """ResNet-18 Deep Residual Network from Scratch."""
    model = Sequential(task="classification")
    # Initial Conv
    model.add(Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))

    # ResNet Stages
    model.add(ResidualBlock(64, 64, stride=1))
    model.add(ResidualBlock(64, 64, stride=1))

    model.add(ResidualBlock(64, 128, stride=2))
    model.add(ResidualBlock(128, 128, stride=1))

    model.add(ResidualBlock(128, 256, stride=2))
    model.add(ResidualBlock(256, 256, stride=1))

    model.add(ResidualBlock(256, 512, stride=2))
    model.add(ResidualBlock(512, 512, stride=1))

    # Classifier Head
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(512, num_classes))
    return model
