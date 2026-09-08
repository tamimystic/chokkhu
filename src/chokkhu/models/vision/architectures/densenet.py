from __future__ import annotations

from typing import List
import numpy as np
from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import AvgPool2D, Conv2D, GlobalAvgPool2D, MaxPool2D
from ..batchnorm2d import BatchNorm2D


class DenseLayer(Module):
    """DenseNet single composite layer: BN -> ReLU -> Conv 1x1 -> BN -> ReLU -> Conv 3x3."""

    def __init__(self, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()
        self.bn1 = BatchNorm2D(in_channels)
        self.relu = ReLU()
        self.conv1 = Conv2D(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2D(4 * growth_rate)
        self.conv2 = Conv2D(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        cat_data = np.concatenate([x.data, out.data], axis=1)
        return Tensor(cat_data, requires_grad=x.requires_grad)


class DenseBlock(Module):
    """DenseNet Dense Block with multiple DenseLayers."""

    def __init__(
        self, num_layers: int, in_channels: int, growth_rate: int = 32
    ) -> None:
        super().__init__()
        self.layers = []
        cur_c = in_channels
        for i in range(num_layers):
            layer = DenseLayer(cur_c, growth_rate=growth_rate)
            self.layers.append(layer)
            setattr(self, f"dense_layer_{i}", layer)
            cur_c += growth_rate
        self.out_channels = cur_c

    def forward(self, x: Tensor) -> Tensor:
        cur = x
        for layer in self.layers:
            cur = layer(cur)
        return cur


class TransitionBlock(Module):
    """Transition layer between DenseBlocks: BN -> Conv 1x1 -> AvgPool 2x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = BatchNorm2D(in_channels)
        self.relu = ReLU()
        self.conv = Conv2D(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = AvgPool2D(2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


def _make_densenet(
    num_layers_list: List[int],
    num_classes: int = 10,
    in_channels: int = 3,
    growth_rate: int = 32,
) -> Sequential:
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))

    cur_c = 64
    for i, num_l in enumerate(num_layers_list):
        db = DenseBlock(num_l, cur_c, growth_rate=growth_rate)
        model.add(db)
        cur_c = db.out_channels
        if i != len(num_layers_list) - 1:
            tb = TransitionBlock(cur_c, cur_c // 2)
            model.add(tb)
            cur_c = cur_c // 2

    model.add(BatchNorm2D(cur_c))
    model.add(ReLU())
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(cur_c, num_classes))
    return model


def DenseNet121(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_densenet(
        [6, 12, 24, 16], num_classes=num_classes, in_channels=in_channels
    )


def DenseNet169(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_densenet(
        [6, 12, 32, 32], num_classes=num_classes, in_channels=in_channels
    )


def DenseNet201(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    return _make_densenet(
        [6, 12, 48, 32], num_classes=num_classes, in_channels=in_channels
    )
