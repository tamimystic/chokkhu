from __future__ import annotations

from typing import Any
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Flatten, Linear, Module
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, GlobalAvgPool2D, MaxPool2D
from ..batchnorm2d import BatchNorm2D


class InceptionModule(Module):
    """Inception Module (Szegedy et al., 2014)."""

    def __init__(
        self,
        in_channels: int,
        n1x1: int,
        n3x3_reduce: int,
        n3x3: int,
        n5x5_reduce: int,
        n5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        # Branch 1: 1x1 conv
        self.b1 = Conv2D(in_channels, n1x1, kernel_size=1)

        # Branch 2: 1x1 conv -> 3x3 conv
        self.b2_1 = Conv2D(in_channels, n3x3_reduce, kernel_size=1)
        self.b2_2 = Conv2D(n3x3_reduce, n3x3, kernel_size=3, padding=1)

        # Branch 3: 1x1 conv -> 5x5 conv (or double 3x3)
        self.b3_1 = Conv2D(in_channels, n5x5_reduce, kernel_size=1)
        self.b3_2 = Conv2D(n5x5_reduce, n5x5, kernel_size=5, padding=2)

        # Branch 4: 3x3 pool -> 1x1 conv
        self.b4_1 = MaxPool2D(pool_size=3, stride=1)
        self.b4_2 = Conv2D(in_channels, pool_proj, kernel_size=1, padding=1)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.relu(self.b1(x))
        out2 = self.relu(self.b2_2(self.relu(self.b2_1(x))))
        out3 = self.relu(self.b3_2(self.relu(self.b3_1(x))))
        p = self.b4_1(x)
        out4 = self.relu(self.b4_2(p))

        # Pad or slice spatial if slight mismatch
        min_h = min(out1.shape[2], out2.shape[2], out3.shape[2], out4.shape[2])
        min_w = min(out1.shape[3], out2.shape[3], out3.shape[3], out4.shape[3])

        cat = np.concatenate(
            [
                out1.data[:, :, :min_h, :min_w],
                out2.data[:, :, :min_h, :min_w],
                out3.data[:, :, :min_h, :min_w],
                out4.data[:, :, :min_h, :min_w],
            ],
            axis=1,
        )
        return Tensor(cat, requires_grad=x.requires_grad)


class GoogLeNet(Module, ChokkhuModel):
    """GoogLeNet / InceptionV1 Architecture from Scratch."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2D(pool_size=3, stride=2)
        self.conv2 = Conv2D(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2D(pool_size=3, stride=2)

        self.inc3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inc3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.gap = GlobalAvgPool2D()
        self.fc = Linear(480, num_classes)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = self.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = self.inc3a(out)
        out = self.inc3b(out)
        out = self.gap(out)
        out = self.fc(out)
        return out

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> GoogLeNet:
        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X).data
        if out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out.flatten()


InceptionV1 = GoogLeNet


def InceptionV3(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """InceptionV3 Compact Architecture from Scratch."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1))
    model.add(BatchNorm2D(32))
    model.add(ReLU())
    model.add(Conv2D(32, 64, kernel_size=3, padding=1))
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(3, stride=2))
    # Factorized Inception Block
    model.add(InceptionModule(64, 32, 32, 64, 16, 32, 32))
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Linear(160, num_classes))
    return model
