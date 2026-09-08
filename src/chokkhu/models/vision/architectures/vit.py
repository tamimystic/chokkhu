from __future__ import annotations

from typing import Any
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import LayerNorm, Linear, Module, Parameter
from ...dl.activations import GELU


class PatchEmbedding(Module):
    """Splits image into non-overlapping patches and projects them to embedding dimension."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = Linear(self.patch_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        p = self.patch_size
        # Extract patches (N, num_patches, patch_dim)
        patches = []
        for i in range(0, H, p):
            for j in range(0, W, p):
                patch = x.data[:, :, i : i + p, j : j + p].reshape(N, -1)
                patches.append(patch[:, np.newaxis, :])

        cat_patches = np.concatenate(patches, axis=1)  # (N, num_patches, patch_dim)
        cat_t = Tensor(
            cat_patches.reshape(-1, self.patch_dim), requires_grad=x.requires_grad
        )
        projected = self.projection(cat_t)
        return projected.reshape(N, self.num_patches, -1)


class MultiHeadSelfAttention(Module):
    """Multi-Head Self-Attention (MHSA) for Vision Transformers."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, seq_len, embed_dim)
        N, seq_len, D = x.shape
        x_flat = x.reshape(-1, D)

        q = (
            self.q_proj(x_flat)
            .data.reshape(N, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x_flat)
            .data.reshape(N, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x_flat)
            .data.reshape(N, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # Attention: Softmax(Q K^T / sqrt(d)) V
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_scores)
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        attn_out = np.matmul(attn_weights, v)  # (N, num_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(N * seq_len, D)

        out = self.out_proj(Tensor(attn_out, requires_grad=x.requires_grad))
        return out.reshape(N, seq_len, D)


class ViTBlock(Module):
    """Transformer Encoder Block for Vision Transformers."""

    def __init__(
        self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads)
        self.norm2 = LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_fc1 = Linear(embed_dim, hidden_dim)
        self.act = GELU()
        self.mlp_fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN MHSA
        N, seq_len, D = x.shape
        x_norm = self.norm1(x.reshape(-1, D)).reshape(N, seq_len, D)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        # Pre-LN MLP
        x_norm2 = self.norm2(x.reshape(-1, D))
        mlp_out = self.mlp_fc2(self.act(self.mlp_fc1(x_norm2))).reshape(N, seq_len, D)
        return x + mlp_out


class VisionTransformer(Module, ChokkhuModel):
    """Vision Transformer (ViT) Architecture from Scratch (Dosovitskiy et al., 2020)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(np.zeros((1, 1, embed_dim)))
        self.pos_embed = Parameter(
            np.random.randn(1, num_patches + 1, embed_dim) * 0.02
        )

        self.blocks = [
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)

        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        patches = self.patch_embed(x)  # (N, num_patches, D)
        cls_tokens = np.repeat(self.cls_token.data, N, axis=0)
        x_tok = np.concatenate([cls_tokens, patches.data], axis=1)
        x_emb = Tensor(x_tok, requires_grad=x.requires_grad) + self.pos_embed

        for b in self.blocks:
            x_emb = b(x_emb)

        # Class token output
        cls_out = x_emb.data[:, 0, :]
        cls_t = Tensor(cls_out, requires_grad=x.requires_grad)
        normed = self.norm(cls_t)
        return self.head(normed)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> VisionTransformer:
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


ViT = VisionTransformer
ViTTiny = VisionTransformer
ViTBase = VisionTransformer
