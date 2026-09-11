"""ControlNet: Adding Conditional Control to Generative and Diffusion Models in pure NumPy."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class ZeroConv2D:
    """Zero-initialized 1x1 2D Convolution Layer.

    Crucial component of ControlNet architecture: weights and biases are
    strictly initialized to zero so that at initialization, the conditioned
    branch contributes exactly zero perturbation to the base network.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Initialized strictly with zeros
        self.weight: np.ndarray = np.zeros(
            (out_channels, in_channels, 1, 1), dtype=np.float32
        )
        self.bias: np.ndarray = np.zeros((out_channels,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying zero 1x1 convolution (B, C_in, H, W) -> (B, C_out, H, W)."""
        B, C_in, H, W = x.shape
        # Vectorized 1x1 convolution via tensordot
        out = np.tensordot(x, self.weight, axes=(1, 1))  # (B, H, W, C_out, 1, 1)
        out = out.squeeze(axis=(-1, -2)).transpose(0, 3, 1, 2)  # (B, C_out, H, W)
        out += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        return out


class ControlNetBlock:
    """Single ControlNet residual conditioning stage."""

    def __init__(self, channels: int, seed: int = 42) -> None:
        self.channels = channels
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (channels * 9))

        # Trainable duplicate convolution block
        self.w1 = rng.normal(0, scale, size=(channels, channels, 3, 3)).astype(
            np.float32
        )
        self.w2 = rng.normal(0, scale, size=(channels, channels, 3, 3)).astype(
            np.float32
        )

        # Zero-convolution layers
        self.zero_in = ZeroConv2D(channels, channels)
        self.zero_out = ZeroConv2D(channels, channels)

    def _conv3x3(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Simple padded 3x3 convolution in pure NumPy."""
        B, C, H, W = x.shape
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        out = np.zeros((B, C, H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                patch = x_pad[:, :, i : i + 3, j : j + 3]
                out[:, :, i, j] = np.tensordot(
                    patch, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        return out

    def forward(self, x: np.ndarray, condition: np.ndarray) -> np.ndarray:
        """Computes zero-conv modulated branch (B, C, H, W)."""
        # Condition injected via zero-conv
        h_cond = self.zero_in.forward(condition)
        h = x + h_cond

        # Trainable feature transformation with non-linearity
        h = np.maximum(0.0, self._conv3x3(h, self.w1))
        h = self._conv3x3(h, self.w2)

        # Zero-conv output modulation
        return self.zero_out.forward(h)


class ControlNet:
    """ControlNet Spatial Condition Adapter in pure NumPy.

    Injects spatial conditioning maps (e.g. edge maps, segmentation layouts,
    depth maps) into deep generative networks while preserving locked base model weights.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 3,
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.base_channels = base_channels
        self.num_stages = num_stages

        rng = np.random.default_rng(seed)
        scale_cond = np.sqrt(2.0 / (cond_channels * 9))

        # Condition pre-processing stem (cond_channels -> base_channels)
        self.w_cond_stem = rng.normal(
            0, scale_cond, size=(base_channels, cond_channels, 3, 3)
        ).astype(np.float32)

        # Multi-stage ControlNet blocks
        self.stages = [
            ControlNetBlock(channels=base_channels, seed=seed + i)
            for i in range(num_stages)
        ]

    def _conv_stem(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        B, _, H, W = x.shape
        C_out, _, _, _ = weight.shape
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        out = np.zeros((B, C_out, H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                patch = x_pad[:, :, i : i + 3, j : j + 3]
                out[:, :, i, j] = np.tensordot(
                    patch, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        return out

    def forward(
        self,
        base_features: np.ndarray,
        condition: np.ndarray,
    ) -> Tuple[np.ndarray, list[np.ndarray]]:
        """Applies multi-stage ControlNet conditioning.

        Args:
            base_features: Latent feature tensor (B, base_channels, H, W).
            condition: Spatial guidance image/map (B, cond_channels, H, W).

        Returns:
            Tuple[np.ndarray, list[np.ndarray]]:
                - Combined modulated output: (B, base_channels, H, W).
                - List of intermediate zero-conv residual skip residuals per stage.
        """
        cond_feat = np.maximum(0.0, self._conv_stem(condition, self.w_cond_stem))

        residuals: list[np.ndarray] = []
        curr_x = base_features

        for stage in self.stages:
            res = stage.forward(curr_x, cond_feat)
            residuals.append(res)
            curr_x = curr_x + res

        return curr_x, residuals
