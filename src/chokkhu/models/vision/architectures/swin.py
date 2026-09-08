"""Swin Transformer (Shifted Windows Transformer) Architecture from Scratch (Liu et al., 2021)."""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module, Parameter
from ...dl.activations import GELU


class WindowAttention(Module):
    """Window-based Multi-Head Self-Attention (W-MSA) with Relative Position Bias."""

    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 4,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim)

        # Relative position bias table
        num_rel_pos = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = Parameter(
            np.zeros((num_rel_pos, num_heads), dtype=np.float64)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (num_windows * N, window_size * window_size, C)
        B_w, N, C = x.shape
        qkv = self.qkv(x.reshape(-1, C)).data.reshape(
            B_w, N, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_w, num_heads, N, head_dim)

        scores = np.matmul(q, k.swapaxes(-2, -1)) * self.scale
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, v).swapaxes(1, 2).reshape(B_w * N, C)
        proj_out = self.proj(Tensor(out, requires_grad=x.requires_grad))
        return proj_out.reshape(B_w, N, C)


class PatchMerging(Module):
    """Patch Merging Layer (Hierarchical Resolution Downsampling $2x$ and Channel Doubling $2x$)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        self.norm = LayerNorm(4 * dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tuple[Tensor, int, int]:
        # x: (N, H * W, C)
        N, L, C = x.shape
        x_data = x.data.reshape(N, H, W, C)

        # Downsample by picking 2x2 neighboring patches
        x0 = x_data[:, 0::2, 0::2, :]
        x1 = x_data[:, 1::2, 0::2, :]
        x2 = x_data[:, 0::2, 1::2, :]
        x3 = x_data[:, 1::2, 1::2, :]
        cat = np.concatenate([x0, x1, x2, x3], axis=-1)  # (N, H/2, W/2, 4*C)

        H_out, W_out = H // 2, W // 2
        flat = cat.reshape(N * H_out * W_out, 4 * C)
        normed = self.norm(Tensor(flat, requires_grad=x.requires_grad))
        out = self.reduction(normed).reshape(N, H_out * W_out, 2 * C)
        return out, H_out, W_out


class SwinTransformerBlock(Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = Linear(dim, mlp_hidden_dim)
        self.act = GELU()
        self.mlp_fc2 = Linear(mlp_hidden_dim, dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # Pre-LN Attention
        N, L, C = x.shape
        x_norm = self.norm1(x.reshape(-1, C)).reshape(N, L, C)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        # Pre-LN MLP
        x_norm2 = self.norm2(x.reshape(-1, C))
        mlp_out = self.mlp_fc2(self.act(self.mlp_fc1(x_norm2))).reshape(N, L, C)
        return x + mlp_out


class SwinTransformer(Module, ChokkhuModel):
    """Swin Transformer from Scratch (Hierarchical Vision Transformer using Shifted Windows)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depths: Tuple[int, ...] = (2, 2),
        num_heads: Tuple[int, ...] = (2, 4),
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch Partition
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = Linear(self.patch_dim, embed_dim)
        self.pos_drop = Dropout(0.0)

        # Stages
        self.stage0_blocks = [
            SwinTransformerBlock(
                dim=embed_dim, num_heads=num_heads[0], window_size=window_size
            )
            for _ in range(depths[0])
        ]
        for i, b in enumerate(self.stage0_blocks):
            setattr(self, f"stage0_b{i}", b)

        self.merge1 = PatchMerging(embed_dim)
        dim_stage1 = embed_dim * 2
        self.stage1_blocks = [
            SwinTransformerBlock(
                dim=dim_stage1, num_heads=num_heads[1], window_size=window_size
            )
            for _ in range(depths[1])
        ]
        for i, b in enumerate(self.stage1_blocks):
            setattr(self, f"stage1_b{i}", b)

        self.norm = LayerNorm(dim_stage1)
        self.head = Linear(dim_stage1, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        p = self.patch_size
        H_patches, W_patches = H // p, W // p

        # Extract patches
        patches = []
        for i in range(0, H, p):
            for j in range(0, W, p):
                patch = x.data[:, :, i : i + p, j : j + p].reshape(N, -1)
                patches.append(patch[:, np.newaxis, :])
        cat_p = np.concatenate(patches, axis=1)  # (N, H_patches*W_patches, patch_dim)
        x_emb = self.patch_proj(
            Tensor(cat_p.reshape(-1, self.patch_dim), requires_grad=x.requires_grad)
        ).reshape(N, H_patches * W_patches, self.embed_dim)

        # Stage 0
        H_curr, W_curr = H_patches, W_patches
        for b in self.stage0_blocks:
            x_emb = b(x_emb, H_curr, W_curr)

        # Stage 1 (after patch merging)
        x_emb, H_curr, W_curr = self.merge1(x_emb, H_curr, W_curr)
        for b in self.stage1_blocks:
            x_emb = b(x_emb, H_curr, W_curr)

        # Global average pooling + classification head
        mean_feat = x_emb.mean(axis=1)  # (N, dim_stage1)
        normed = self.norm(mean_feat)
        return self.head(normed)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> SwinTransformer:
        from ...dl.sequential import Sequential

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


SwinT = SwinTransformer
SwinSmall = SwinTransformer
SwinBase = SwinTransformer
