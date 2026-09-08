from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import (
    ChannelShuffle,
    Conv2D,
    GlobalAvgPool2D,
    MaxPool2D,
)
from ..batchnorm2d import BatchNorm2D


class ShuffleUnitV2(Module):
    """ShuffleNetV2 Inverted Residual Block with Channel Split & Shuffle."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        branch_channels = out_channels // 2

        self.branch2 = Sequential()
        self.branch2.add(
            Conv2D(
                in_channels if stride > 1 else branch_channels,
                branch_channels,
                kernel_size=1,
                bias=False,
            )
        )
        self.branch2.add(BatchNorm2D(branch_channels))
        self.branch2.add(ReLU())
        self.branch2.add(
            Conv2D(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        self.branch2.add(BatchNorm2D(branch_channels))
        self.branch2.add(
            Conv2D(branch_channels, branch_channels, kernel_size=1, bias=False)
        )
        self.branch2.add(BatchNorm2D(branch_channels))
        self.branch2.add(ReLU())

        if stride > 1:
            self.branch1 = Sequential()
            self.branch1.add(
                Conv2D(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            self.branch1.add(BatchNorm2D(branch_channels))
            self.branch1.add(
                Conv2D(branch_channels, branch_channels, kernel_size=1, bias=False)
            )
            self.branch1.add(BatchNorm2D(branch_channels))
            self.branch1.add(ReLU())
        else:
            self.branch1 = None

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            c = x.shape[1] // 2
            x1 = Tensor(x.data[:, :c, :, :], requires_grad=x.requires_grad)
            x2 = Tensor(x.data[:, c:, :, :], requires_grad=x.requires_grad)
            out2 = self.branch2(x2)
            cat = np.concatenate([x1.data, out2.data], axis=1)
        else:
            out1 = self.branch1(x) if self.branch1 else x
            out2 = self.branch2(x)
            cat = np.concatenate([out1.data, out2.data], axis=1)

        return self.shuffle(Tensor(cat, requires_grad=x.requires_grad))


def ShuffleNetV2(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """ShuffleNetV2 Architecture (Ma et al., 2018)."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 24, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(24))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))

    model.add(ShuffleUnitV2(24, 48, stride=2))
    model.add(ShuffleUnitV2(48, 48, stride=1))
    model.add(ShuffleUnitV2(48, 96, stride=2))
    model.add(ShuffleUnitV2(96, 96, stride=1))

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(96, num_classes))
    return model


ShuffleNetV1 = ShuffleNetV2
