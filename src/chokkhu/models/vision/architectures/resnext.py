from __future__ import annotations

from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D, GroupedConv2D, MaxPool2D
from ..batchnorm2d import BatchNorm2D


class ResNeXtBlock(Module):
    """ResNeXt Bottleneck Block with Cardinality (Grouped Conv)."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        cardinality: int = 32,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = base_channels * 2
        self.conv1 = Conv2D(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2D(base_channels)

        self.conv2 = GroupedConv2D(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = BatchNorm2D(base_channels)

        self.conv3 = Conv2D(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2D(out_channels)

        self.relu = ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_bn = BatchNorm2D(out_channels)
        else:
            self.shortcut = None
            self.shortcut_bn = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.shortcut is not None:
            identity = self.shortcut_bn(self.shortcut(identity))

        out = out + identity
        return self.relu(out)


def ResNeXt50(
    num_classes: int = 10, in_channels: int = 3, cardinality: int = 32
) -> Sequential:
    """ResNeXt-50 Architecture from Scratch."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))

    model.add(ResNeXtBlock(64, 64, cardinality=cardinality, stride=1))
    model.add(ResNeXtBlock(128, 128, cardinality=cardinality, stride=2))
    model.add(ResNeXtBlock(256, 256, cardinality=cardinality, stride=2))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(512, num_classes))
    return model


def ResNeXt101(
    num_classes: int = 10, in_channels: int = 3, cardinality: int = 32
) -> Sequential:
    """ResNeXt-101 Architecture from Scratch."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))

    model.add(ResNeXtBlock(64, 64, cardinality=cardinality, stride=1))
    model.add(ResNeXtBlock(128, 64, cardinality=cardinality, stride=1))
    model.add(ResNeXtBlock(128, 128, cardinality=cardinality, stride=2))
    model.add(ResNeXtBlock(256, 128, cardinality=cardinality, stride=1))
    model.add(ResNeXtBlock(256, 256, cardinality=cardinality, stride=2))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(512, num_classes))
    return model
