from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from ..dl.layers import Module, Parameter


class BatchNorm2D(Module):
    """2D Spatial Batch Normalization across spatial (H, W) per channel."""

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones((1, num_features, 1, 1)))
        self.beta = Parameter(np.zeros((1, num_features, 1, 1)))

        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        elif x.data.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        if x.data.size == 0 or x.shape[0] == 0:
            return x

        if self.training:
            mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            var = x.data.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var
            mean_t = Tensor(mean, requires_grad=False)
            std_t = Tensor(np.sqrt(var + self.eps), requires_grad=False)
            x_norm = (x - mean_t) / std_t
        else:
            mean_t = Tensor(self.running_mean, requires_grad=False)
            std_t = Tensor(np.sqrt(self.running_var + self.eps), requires_grad=False)
            x_norm = (x - mean_t) / std_t

        return x_norm * self.gamma + self.beta
