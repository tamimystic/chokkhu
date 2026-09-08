from __future__ import annotations

from typing import Any
import numpy as np
from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Module
from ...dl.activations import ReLU
from ..conv_layers import Conv2D, ConvTranspose2D, MaxPool2D


class FCN8s(Module, ChokkhuModel):
    """Fully Convolutional Network (FCN-8s) for Semantic Segmentation."""

    def __init__(self, num_classes: int = 21, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = Conv2D(in_channels, 32, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool1 = MaxPool2D(2, stride=2)

        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2D(2, stride=2)

        self.conv3 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.pool3 = MaxPool2D(2, stride=2)

        self.score_pool3 = Conv2D(128, num_classes, kernel_size=1)
        self.upscore2 = ConvTranspose2D(
            num_classes, num_classes, kernel_size=2, stride=2
        )
        self.upscore8 = ConvTranspose2D(
            num_classes, num_classes, kernel_size=4, stride=4
        )

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.relu(self.conv1(x))
        p1 = self.pool1(h1)
        h2 = self.relu(self.conv2(p1))
        p2 = self.pool2(h2)
        h3 = self.relu(self.conv3(p2))
        p3 = self.pool3(h3)

        s3 = self.score_pool3(p3)
        u2 = self.upscore2(s3)
        return self.upscore8(u2)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> FCN8s:
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        return self.forward(X).data


FCN = FCN8s
