from __future__ import annotations

import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Linear, Module
from ..dl.activations import ReLU, Sigmoid
from .conv_layers import Conv2D


class SEBlock(Module):
    """Squeeze-and-Excitation (SE) Channel Attention Block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        reduced = max(1, channels // reduction)
        self.fc1 = Linear(channels, reduced)
        self.relu = ReLU()
        self.fc2 = Linear(reduced, channels)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        # Squeeze: Global Average Pooling (N, C)
        s = x.mean(axis=(2, 3))
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        z = self.fc1(s)
        z = self.relu(z)
        z = self.fc2(z)
        weights = self.sigmoid(z)  # (N, C)

        # Recalibrate: (N, C, 1, 1) * (N, C, H, W)
        w_reshaped = weights.reshape(N, C, 1, 1)
        return x * w_reshaped


class ChannelAttention(Module):
    """Channel Attention Module for CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = Linear(channels, reduced)
        self.relu = ReLU()
        self.fc2 = Linear(reduced, channels)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        # AvgPool
        avg_out = x.mean(axis=(2, 3))
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        # MaxPool
        max_data = np.max(x.data, axis=(2, 3))
        max_t = Tensor(max_data, requires_grad=x.requires_grad)
        max_out = self.fc2(self.relu(self.fc1(max_t)))

        scale = self.sigmoid(avg_out + max_out).reshape(N, C, 1, 1)
        return x * scale


class SpatialAttention(Module):
    """Spatial Attention Module for CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = Conv2D(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        avg_out = np.mean(x.data, axis=1, keepdims=True)
        max_out = np.max(x.data, axis=1, keepdims=True)
        cat_data = np.concatenate([avg_out, max_out], axis=1)
        cat_t = Tensor(cat_data, requires_grad=x.requires_grad)
        scale = self.sigmoid(self.conv(cat_t))
        return x * scale


class CBAM(Module):
    """Convolutional Block Attention Module (Channel + Spatial Attention)."""

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        return out
