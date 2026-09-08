from __future__ import annotations

from typing import List
from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D, MaxPool2D
from ..batchnorm2d import BatchNorm2D


class ResidualBlock(Module):
    """Basic Residual Block: F(x) + x."""

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


class BottleneckResidualBlock(Module):
    """Bottleneck Residual Block: 1x1 -> 3x3 -> 1x1."""

    expansion: int = 4

    def __init__(self, in_channels: int, base_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = base_channels * self.expansion
        self.conv1 = Conv2D(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2D(base_channels)

        self.conv2 = Conv2D(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
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


def _make_resnet(
    block_type: type,
    num_blocks: List[int],
    num_classes: int = 10,
    in_channels: int = 3,
) -> Sequential:
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))

    current_c = 64
    for stage_idx, (num_b, base_c) in enumerate(zip(num_blocks, [64, 128, 256, 512])):
        stride = 1 if stage_idx == 0 else 2
        for b_idx in range(num_b):
            s = stride if b_idx == 0 else 1
            if block_type == BottleneckResidualBlock:
                model.add(BottleneckResidualBlock(current_c, base_c, stride=s))
                current_c = base_c * 4
            else:
                model.add(ResidualBlock(current_c, base_c, stride=s))
                current_c = base_c

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(current_c, num_classes))
    return model


def ResNet18(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_resnet(
        ResidualBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels
    )


def ResNet34(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_resnet(
        ResidualBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels
    )


def ResNet50(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_resnet(
        BottleneckResidualBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
    )


def ResNet101(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_resnet(
        BottleneckResidualBlock,
        [3, 4, 23, 3],
        num_classes=num_classes,
        in_channels=in_channels,
    )


def ResNet152(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_resnet(
        BottleneckResidualBlock,
        [3, 8, 36, 3],
        num_classes=num_classes,
        in_channels=in_channels,
    )
