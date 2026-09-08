from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from ...dl.layers import Dropout, Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D, MaxPool2D


class FireModule(Module):
    """SqueezeNet Fire Module: Squeeze (1x1) -> Expand (1x1 + 3x3)."""

    def __init__(
        self,
        in_channels: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.squeeze = Conv2D(in_channels, squeeze_planes, kernel_size=1)
        self.expand1x1 = Conv2D(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = Conv2D(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        s = self.relu(self.squeeze(x))
        e1 = self.relu(self.expand1x1(s))
        e3 = self.relu(self.expand3x3(s))
        cat = np.concatenate([e1.data, e3.data], axis=1)
        return Tensor(cat, requires_grad=x.requires_grad)


def SqueezeNet(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """SqueezeNet v1.0 Architecture (Iandola et al., 2016)."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 96, kernel_size=7, stride=2, padding=3))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))

    model.add(FireModule(96, 16, 64, 64))
    model.add(FireModule(128, 16, 64, 64))
    model.add(MaxPool2D(3, stride=2))

    model.add(FireModule(128, 32, 128, 128))
    model.add(FireModule(256, 32, 128, 128))

    model.add(Dropout(0.5))
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(256, num_classes))
    return model
