from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Function, Tensor
from ..dl.layers import Module, Parameter


def _get_im2col_indices(
    x_shape: Tuple[int, int, int, int],
    field_height: int,
    field_width: int,
    padding: int = 0,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0: np.ndarray = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1: np.ndarray = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k: np.ndarray = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(
    x: np.ndarray,
    field_height: int,
    field_width: int,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j = _get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    field_height: int = 3,
    field_width: int = 3,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded: np.ndarray = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = _get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2DFunction(Function):
    def __init__(self, kh: int, kw: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.kh = kh
        self.kw = kw
        self.stride = stride
        self.padding = padding
        self.x_col: Optional[np.ndarray] = None

    def forward(  # type: ignore[override]
        self, x: Any, weight: Any = None, bias: Any = None
    ) -> Any:
        N, C, H, W = x.shape
        out_channels = weight.shape[0]
        out_h = int((H + 2 * self.padding - self.kh) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kw) / self.stride + 1)

        self.x_col = im2col_indices(
            x, self.kh, self.kw, padding=self.padding, stride=self.stride
        )
        w_row = weight.reshape(out_channels, -1)

        out = np.dot(w_row, self.x_col)
        if bias is not None:
            out += bias.reshape(out_channels, 1)

        out = out.reshape(out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
        return out

    def backward(
        self, gy: np.ndarray
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        # gy shape: (N, Out_Channels, Out_H, Out_W)
        x_shape = self.inputs[0].shape
        weight = self.inputs[1].data
        out_channels = weight.shape[0]

        dout_reshaped = gy.transpose(1, 2, 3, 0).reshape(out_channels, -1)
        dW = np.dot(dout_reshaped, self.x_col.T).reshape(weight.shape)

        w_row = weight.reshape(out_channels, -1)
        dX_col = np.dot(w_row.T, dout_reshaped)
        shape_4d: Tuple[int, int, int, int] = (
            int(x_shape[0]),
            int(x_shape[1]),
            int(x_shape[2]),
            int(x_shape[3]),
        )
        dX = col2im_indices(
            dX_col,
            shape_4d,
            self.kh,
            self.kw,
            padding=self.padding,
            stride=self.stride,
        )

        if len(self.inputs) > 2 and self.inputs[2] is not None:
            db = np.sum(dout_reshaped, axis=1, keepdims=True)
            return dX, dW, db
        return dX, dW


class Conv2D(Module):
    """Vectorized 2D Convolution Layer (im2col + GEMM)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        padding: Union[int, str] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size

        self.stride = stride
        if padding == "same":
            self.padding = self.kh // 2
        elif padding == "valid":
            self.padding = 0
        else:
            self.padding = int(padding)

        fan_in = in_channels * self.kh * self.kw
        w_data = np.random.randn(out_channels, in_channels, self.kh, self.kw) * np.sqrt(
            2.0 / fan_in
        )
        self.weight = Parameter(w_data)
        if bias:
            self.bias = Parameter(np.zeros((out_channels, 1)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        elif x.data.ndim == 2:
            side = int(np.sqrt(x.shape[1] // self.in_channels))
            x = x.reshape(x.shape[0], self.in_channels, side, side)

        fn = Conv2DFunction(self.kh, self.kw, self.stride, self.padding)
        if self.bias is not None:
            return fn(x, self.weight, self.bias)
        return fn(x, self.weight)


class MaxPool2DFunction(Function):
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.max_idx: Optional[np.ndarray] = None
        self.x_col: Optional[np.ndarray] = None

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        N, C, H, W = x.shape
        out_h = int((H - self.pool_size) / self.stride + 1)
        out_w = int((W - self.pool_size) / self.stride + 1)

        x_reshaped = x.reshape(N * C, 1, H, W)
        self.x_col = im2col_indices(
            x_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride
        )

        self.max_idx = np.argmax(self.x_col, axis=0)
        out = self.x_col[self.max_idx, np.arange(self.max_idx.size)]
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)
        return out

    def backward(self, gy: np.ndarray) -> np.ndarray:
        N, C, H, W = self.inputs[0].shape
        dx_col = np.zeros_like(self.x_col)
        dout_flat = gy.transpose(2, 3, 0, 1).ravel()
        dx_col[self.max_idx, np.arange(self.max_idx.size)] = dout_flat
        dx = col2im_indices(
            dx_col,
            (N * C, 1, H, W),
            self.pool_size,
            self.pool_size,
            padding=0,
            stride=self.stride,
        )
        return dx.reshape(N, C, H, W)


class MaxPool2D(Module):
    """2D Max Pooling Layer."""

    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return MaxPool2DFunction(self.pool_size, self.stride)(x)


class AvgPool2DFunction(Function):
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.x_col: Optional[np.ndarray] = None

    def forward(self, x: Any, *args: Any) -> Any:  # type: ignore[override]
        N, C, H, W = x.shape
        out_h = int((H - self.pool_size) / self.stride + 1)
        out_w = int((W - self.pool_size) / self.stride + 1)

        x_reshaped = x.reshape(N * C, 1, H, W)
        self.x_col = im2col_indices(
            x_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride
        )
        out = (
            np.mean(self.x_col, axis=0)
            .reshape(out_h, out_w, N, C)
            .transpose(2, 3, 0, 1)
        )
        return out

    def backward(self, gy: np.ndarray) -> np.ndarray:
        N, C, H, W = self.inputs[0].shape
        numel = self.pool_size * self.pool_size
        dout_flat = gy.transpose(2, 3, 0, 1).ravel() / numel
        dx_col = np.repeat(dout_flat[np.newaxis, :], numel, axis=0)
        dx = col2im_indices(
            dx_col,
            (N * C, 1, H, W),
            self.pool_size,
            self.pool_size,
            padding=0,
            stride=self.stride,
        )
        return dx.reshape(N, C, H, W)


class AvgPool2D(Module):
    """2D Average Pooling Layer."""

    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return AvgPool2DFunction(self.pool_size, self.stride)(x)


class GlobalAvgPool2D(Module):
    """Global Average Pooling Layer: reduces (N, C, H, W) -> (N, C)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(axis=(2, 3))


class DepthwiseSeparableConv2D(Module):
    """Depthwise Separable Convolution (Depthwise Spatial Conv + Pointwise 1x1 Conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = Conv2D(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pointwise = Conv2D(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise(x)
        return self.pointwise(out)


class ConvTranspose2D(Module):
    """Fractionally Strided Transposed 2D Convolution for Upsampling."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        w_data = (
            np.random.randn(in_channels, out_channels, kernel_size, kernel_size) * 0.05
        )
        self.weight = Parameter(w_data)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        out_h = H * self.stride
        out_w = W * self.stride
        out: np.ndarray = np.zeros(
            (N, self.out_channels, out_h, out_w), dtype=np.float64
        )

        for n in range(N):
            for oc in range(self.out_channels):
                for ic in range(C):
                    for h in range(H):
                        for w in range(W):
                            out[
                                n,
                                oc,
                                h * self.stride : h * self.stride + self.kernel_size,
                                w * self.stride : w * self.stride + self.kernel_size,
                            ] += (
                                x.data[n, ic, h, w] * self.weight.data[ic, oc]
                            )
        return Tensor(out, requires_grad=x.requires_grad)
