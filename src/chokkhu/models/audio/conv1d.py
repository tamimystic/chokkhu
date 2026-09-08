"""1D Convolution, Depthwise Conv1D, and BatchNorm1D from First Principles."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from chokkhu.core.tensor import Function, Tensor
from ..dl.layers import Module, Parameter


def im2col1d(
    x: np.ndarray,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[np.ndarray, int]:
    """Vectorized im2col for 1D sequences."""
    N, C, L = x.shape
    eff_kernel = (kernel_size - 1) * dilation + 1
    if padding > 0:
        x_pad: np.ndarray = np.pad(
            x, ((0, 0), (0, 0), (padding, padding)), mode="constant"
        )
    else:
        x_pad = x

    L_pad = x_pad.shape[2]
    out_len = int((L_pad - eff_kernel) / stride + 1)
    if out_len <= 0:
        raise ValueError(
            f"Output length is <= 0 (L_pad={L_pad}, eff_kernel={eff_kernel})"
        )

    k_idx: np.ndarray = np.arange(0, kernel_size * dilation, dilation)[:, None]
    s_idx: np.ndarray = np.arange(out_len, dtype=np.int64)[None, :] * stride
    time_indices = (k_idx + s_idx).reshape(-1)

    cols_4d = x_pad[:, :, time_indices].reshape(N, C, kernel_size, out_len)
    cols: np.ndarray = cols_4d.transpose(1, 2, 0, 3).reshape(
        C * kernel_size, N * out_len
    )
    return cols, out_len


class Conv1DFunction(Function):
    """Autograd Function for 1D Convolution."""

    def __init__(
        self,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(  # type: ignore[override]
        self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.x = x
        self.weight = weight
        self.bias = bias
        N, C, L = x.shape
        out_channels, in_channels_per_group, kernel_size = weight.shape

        cols, out_len = im2col1d(
            x, kernel_size, self.stride, self.padding, self.dilation
        )
        self.cols = cols
        self.out_len = out_len

        w_row = weight.reshape(out_channels, -1)
        out = np.matmul(w_row, cols)
        out = out.reshape(out_channels, N, out_len).transpose(1, 0, 2)

        if bias is not None:
            out += bias.reshape(1, out_channels, 1)

        return out

    def backward(
        self, gy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        out_channels, in_channels_per_group, kernel_size = self.weight.shape
        N, C, L = self.x.shape

        gbias = np.sum(gy, axis=(0, 2)) if self.bias is not None else None

        gy_reshaped = gy.transpose(1, 0, 2).reshape(out_channels, -1)
        gweight = np.matmul(gy_reshaped, self.cols.T).reshape(self.weight.shape)

        w_row = self.weight.reshape(out_channels, -1)
        gcols = np.matmul(w_row.T, gy_reshaped)
        gcols = gcols.reshape(C, kernel_size, N, self.out_len).transpose(2, 0, 1, 3)

        gx_pad: np.ndarray = np.zeros((N, C, L + 2 * self.padding), dtype=np.float64)
        for k in range(kernel_size):
            k_offset = k * self.dilation
            for t in range(self.out_len):
                gx_pad[:, :, t * self.stride + k_offset] += gcols[:, :, k, t]

        if self.padding > 0:
            gx = gx_pad[:, :, self.padding : self.padding + L]
        else:
            gx = gx_pad

        return gx, gweight, gbias


class Conv1D(Module):
    """1D Convolutional Layer for Sequences and Audio Waveforms."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        std = np.sqrt(2.0 / (in_channels * kernel_size))
        w_data = np.random.randn(out_channels, in_channels // groups, kernel_size) * std
        self.weight = Parameter(w_data)

        if bias:
            self.bias: Optional[Parameter] = Parameter(
                np.zeros(out_channels, dtype=np.float64)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        fn = Conv1DFunction(
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            return fn(x, self.weight, self.bias)
        return fn(x, self.weight)


class DepthwiseConv1D(Module):
    """1D Depthwise Separable Convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Conv1D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class BatchNorm1D(Module):
    """1D Batch Normalization Layer along Sequence / Channel Dimension."""

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones((1, num_features, 1), dtype=np.float64))
        self.beta = Parameter(np.zeros((1, num_features, 1), dtype=np.float64))

        self.running_mean: np.ndarray = np.zeros((1, num_features, 1), dtype=np.float64)
        self.running_var: np.ndarray = np.ones((1, num_features, 1), dtype=np.float64)

    def forward(self, x: Tensor) -> Tensor:
        x_data = x.data
        if self.training:
            mean = np.mean(x_data, axis=(0, 2), keepdims=True)
            var = np.var(x_data, axis=(0, 2), keepdims=True)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            x_norm = (x_data - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x_data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = x_norm * self.gamma.data + self.beta.data
        return Tensor(out, requires_grad=x.requires_grad)
