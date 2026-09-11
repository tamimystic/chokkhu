from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


def _conv2d_simple(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """Vectorized 2D Convolution with dilation support in pure NumPy."""
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    # Apply dilation to kernel size
    eff_kH = kH + (kH - 1) * (dilation - 1)
    eff_kW = kW + (kW - 1) * (dilation - 1)

    if padding > 0:
        x_pad = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        x_pad = x

    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
    out_H = (H_pad - eff_kH) // stride + 1
    out_W = (W_pad - eff_kW) // stride + 1

    out = np.zeros((B, C_out, out_H, out_W), dtype=np.float32)

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = x_pad[
                :,
                :,
                h_start : h_start + eff_kH : dilation,
                w_start : w_start + eff_kW : dilation,
            ]
            # patch: (B, C_in, kH, kW)
            val = np.tensordot(patch, weight, axes=((1, 2, 3), (1, 2, 3)))
            if bias is not None:
                val += bias
            out[:, :, i, j] = val

    return out


def _bilinear_resize_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resizes a 4D tensor (B, C, H, W) to (B, C, target_h, target_w) via bilinear interpolation."""
    B, C, H, W = x.shape
    if H == target_h and W == target_w:
        return x

    grid_y = np.linspace(0, H - 1, target_h)
    grid_x = np.linspace(0, W - 1, target_w)

    y0 = np.floor(grid_y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.floor(grid_x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wa = (y1 - grid_y)[:, np.newaxis] * (x1 - grid_x)[np.newaxis, :]
    wb = (y1 - grid_y)[:, np.newaxis] * (grid_x - x0)[np.newaxis, :]
    wc = (grid_y - y0)[:, np.newaxis] * (x1 - x0)[np.newaxis, :]
    wd = (grid_y - y0)[:, np.newaxis] * (grid_x - x0)[np.newaxis, :]

    out = np.zeros((B, C, target_h, target_w), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            im = x[b, c]
            out[b, c] = (
                wa * im[y0[:, None], x0]
                + wb * im[y0[:, None], x1]
                + wc * im[y1[:, None], x0]
                + wd * im[y1[:, None], x1]
            )

    return out


class ASPPModule:
    """Atrous Spatial Pyramid Pooling (ASPP) Module."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        atrous_rates: Sequence[int] = (1, 6, 12, 18),
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = list(atrous_rates)

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_channels)

        # 1x1 conv branch
        self.w_1x1 = rng.normal(
            0, scale, size=(out_channels, in_channels, 1, 1)
        ).astype(np.float32)

        # Atrous branches
        self.w_atrous = [
            rng.normal(0, scale, size=(out_channels, in_channels, 3, 3)).astype(
                np.float32
            )
            for _ in range(len(self.atrous_rates) - 1)
        ]

        # Global pooling branch 1x1 conv
        self.w_image = rng.normal(
            0, scale, size=(out_channels, in_channels, 1, 1)
        ).astype(np.float32)

        # Output projection 1x1 conv
        total_in = out_channels * (len(self.atrous_rates) + 1)
        scale_proj = np.sqrt(2.0 / total_in)
        self.w_proj = rng.normal(
            0, scale_proj, size=(out_channels, total_in, 1, 1)
        ).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        branches = []

        # 1. 1x1 conv
        out_1x1 = np.maximum(0.0, _conv2d_simple(x, self.w_1x1, stride=1, padding=0))
        branches.append(out_1x1)

        # 2. Atrous 3x3 convs
        for w, rate in zip(self.w_atrous, self.atrous_rates[1:]):
            pad = rate
            out_rate = np.maximum(
                0.0, _conv2d_simple(x, w, stride=1, padding=pad, dilation=rate)
            )
            branches.append(out_rate)

        # 3. Image global average pooling
        img_pool = np.mean(x, axis=(2, 3), keepdims=True)  # (B, C, 1, 1)
        img_conv = np.maximum(0.0, _conv2d_simple(img_pool, self.w_image))
        img_upsampled = np.repeat(np.repeat(img_conv, H, axis=2), W, axis=3)
        branches.append(img_upsampled)

        # 4. Concatenate and project
        cat = np.concatenate(branches, axis=1)
        out = np.maximum(0.0, _conv2d_simple(cat, self.w_proj, stride=1, padding=0))
        return out


class DeepLabV3Plus:
    """DeepLabV3+ Semantic Segmentation Architecture in pure NumPy.

    Features Atrous Spatial Pyramid Pooling (ASPP) and a low-level feature
    decoder for precise boundary delineation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        backbone_channels: int = 32,
        aspp_channels: int = 32,
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_channels = backbone_channels
        self.aspp_channels = aspp_channels

        rng = np.random.default_rng(seed)

        # Initial stem conv (low level feature extractor)
        scale_stem = np.sqrt(2.0 / (in_channels * 9))
        self.w_stem = rng.normal(
            0, scale_stem, size=(backbone_channels, in_channels, 3, 3)
        ).astype(np.float32)

        # High level conv
        scale_high = np.sqrt(2.0 / (backbone_channels * 9))
        self.w_high = rng.normal(
            0, scale_high, size=(backbone_channels * 2, backbone_channels, 3, 3)
        ).astype(np.float32)

        # ASPP module on high-level features
        self.aspp = ASPPModule(
            in_channels=backbone_channels * 2,
            out_channels=aspp_channels,
            atrous_rates=(1, 2, 4),
            seed=seed,
        )

        # Low-level projection 1x1
        scale_low = np.sqrt(2.0 / backbone_channels)
        self.w_low_proj = rng.normal(
            0, scale_low, size=(16, backbone_channels, 1, 1)
        ).astype(np.float32)

        # Decoder fusion conv 3x3
        fusion_in = aspp_channels + 16
        scale_dec = np.sqrt(2.0 / (fusion_in * 9))
        self.w_dec = rng.normal(0, scale_dec, size=(32, fusion_in, 3, 3)).astype(
            np.float32
        )

        # Final classification conv 1x1
        scale_cls = np.sqrt(2.0 / 32)
        self.w_cls = rng.normal(0, scale_cls, size=(num_classes, 32, 1, 1)).astype(
            np.float32
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass predicting semantic segmentation logits (B, num_classes, H, W)."""
        B, C, H, W = x.shape

        # 1. Stem: low-level features
        f_low = np.maximum(0.0, _conv2d_simple(x, self.w_stem, stride=1, padding=1))

        # 2. High-level feature abstraction (stride 2)
        f_high = np.maximum(
            0.0, _conv2d_simple(f_low, self.w_high, stride=2, padding=1)
        )

        # 3. ASPP multi-scale pooling
        f_aspp = self.aspp.forward(f_high)

        # 4. Decoder
        f_aspp_up = _bilinear_resize_2d(f_aspp, f_low.shape[2], f_low.shape[3])
        f_low_proj = np.maximum(
            0.0, _conv2d_simple(f_low, self.w_low_proj, stride=1, padding=0)
        )

        # 5. Fuse & refine
        f_fused = np.concatenate([f_aspp_up, f_low_proj], axis=1)
        f_dec = np.maximum(
            0.0, _conv2d_simple(f_fused, self.w_dec, stride=1, padding=1)
        )

        # 6. Final classification logits
        logits_low = _conv2d_simple(f_dec, self.w_cls, stride=1, padding=0)
        logits = _bilinear_resize_2d(logits_low, H, W)

        return logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns pixel-wise class indices of shape (B, H, W)."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
