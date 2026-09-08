"""Advanced Convolution Operations (Dilated/Atrous Convolution) from Scratch."""

from __future__ import annotations

import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Module, Parameter


class DilatedConv2D(Module):
    """2D Dilated (Atrous) Convolution from First Principles (Yu & Koltun, 2015)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        # Effective kernel size with dilation
        self.effective_k = (kernel_size - 1) * dilation + 1

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )
        if bias:
            self.bias: Parameter | None = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None

    def _inflate_kernel(self, w: np.ndarray) -> np.ndarray:
        """Insert zeros between kernel elements according to dilation rate."""
        out_c, in_c, k_h, k_w = w.shape
        eff_k = self.effective_k
        inflated = np.zeros((out_c, in_c, eff_k, eff_k), dtype=w.dtype)
        inflated[:, :, :: self.dilation, :: self.dilation] = w
        return inflated

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        p = self.padding
        eff_k = self.effective_k
        s = self.stride

        # Pad input
        if p > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        else:
            x_padded = x.data

        H_pad, W_pad = x_padded.shape[2], x_padded.shape[3]
        out_h = (H_pad - eff_k) // s + 1
        out_w = (W_pad - eff_k) // s + 1

        inflated_weight = self._inflate_kernel(self.weight.data)
        out = np.zeros((N, self.out_channels, out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                patch = x_padded[
                    :, :, h_start : h_start + eff_k, w_start : w_start + eff_k
                ]
                out[:, :, i, j] = np.tensordot(
                    patch, inflated_weight, axes=([1, 2, 3], [1, 2, 3])
                )

        if self.bias is not None:
            out += self.bias.data[np.newaxis, :, np.newaxis, np.newaxis]

        requires_grad = x.requires_grad or self.weight.requires_grad
        return Tensor(out, requires_grad=requires_grad)


AtrousConv2D = DilatedConv2D
