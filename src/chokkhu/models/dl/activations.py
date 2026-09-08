from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from .layers import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        pos = x.relu()
        neg = -((-x).relu()) * self.negative_slope
        return pos + neg


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        max_x = np.max(x.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(x.data - max_x)
        probs = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        return Tensor(probs, requires_grad=x.requires_grad)
