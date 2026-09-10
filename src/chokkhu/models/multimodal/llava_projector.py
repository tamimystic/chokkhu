"""LLaVA Projector: Multi-Modal Vision-Language Alignment Bridge.

Pure NumPy implementations of vision-to-language projection adapters:
- Linear Projector: Direct linear projection
- MLP Projector: 2-layer MLP with GELU activation (LLaVA-1.5 standard)
- Perceiver Resampler: Multi-head cross-attention resampling (Flamingo / IDEFICS standard)
"""

import math
from typing import Optional
import numpy as np


class LLaVALinearProjector:
    """Single linear projection bridge."""

    def __init__(
        self, vision_dim: int, text_dim: int, seed: Optional[int] = 42
    ) -> None:
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        rng = np.random.RandomState(seed)
        self.weight = (
            rng.randn(vision_dim, text_dim) * (1.0 / math.sqrt(vision_dim))
        ).astype(np.float32)
        self.bias: np.ndarray = np.zeros(text_dim, dtype=np.float32)

    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        # vision_features: (B, N_patches, vision_dim)
        return np.dot(vision_features, self.weight) + self.bias


class LLaVAMLPProjector:
    """Two-layer MLP projection bridge with GELU non-linearity (LLaVA-1.5)."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        mlp_dim: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.mlp_dim = mlp_dim or text_dim

        rng = np.random.RandomState(seed)
        self.w1 = (
            rng.randn(vision_dim, self.mlp_dim) * (1.0 / math.sqrt(vision_dim))
        ).astype(np.float32)
        self.b1: np.ndarray = np.zeros(self.mlp_dim, dtype=np.float32)
        self.w2 = (
            rng.randn(self.mlp_dim, text_dim) * (1.0 / math.sqrt(self.mlp_dim))
        ).astype(np.float32)
        self.b2: np.ndarray = np.zeros(text_dim, dtype=np.float32)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return (
            0.5
            * x
            * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )

    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        # vision_features: (B, N_patches, vision_dim)
        h = self._gelu(np.dot(vision_features, self.w1) + self.b1)
        return np.dot(h, self.w2) + self.b2


class PerceiverResampler:
    r"""Perceiver Cross-Attention Resampler for fixed-length visual token compression.

    Projects variable visual tokens :math:`(B, N_v, D_v)` to fixed latent tokens :math:`(B, K, D_t)`.

    Parameters
    ----------
    vision_dim : int
        Vision token dimension :math:`D_v`.
    text_dim : int
        Language model token dimension :math:`D_t`.
    num_latents : int, default=64
        Fixed number of output visual query tokens :math:`K`.
    num_heads : int, default=8
        Number of attention heads.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        seed: Optional[int] = 42,
    ) -> None:
        if text_dim % num_heads != 0:
            raise ValueError("text_dim must be divisible by num_heads")

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads

        rng = np.random.RandomState(seed)
        self.latents = (rng.randn(1, num_latents, text_dim) * 0.02).astype(np.float32)

        # Linear projection for vision input to text_dim
        self.vis_proj = (
            rng.randn(vision_dim, text_dim) * (1.0 / math.sqrt(vision_dim))
        ).astype(np.float32)

        # Cross-attention weights
        scale = 1.0 / math.sqrt(text_dim)
        self.w_q = (rng.randn(text_dim, text_dim) * scale).astype(np.float32)
        self.w_k = (rng.randn(text_dim, text_dim) * scale).astype(np.float32)
        self.w_v = (rng.randn(text_dim, text_dim) * scale).astype(np.float32)
        self.w_out = (rng.randn(text_dim, text_dim) * scale).astype(np.float32)

    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        # vision_features: (B, N_v, vision_dim)
        B, N_v, _ = vision_features.shape
        vis_tokens = np.dot(vision_features, self.vis_proj)  # (B, N_v, text_dim)

        # Repeat learnable latents across batch
        queries = np.repeat(self.latents, B, axis=0)  # (B, K, text_dim)
        K = self.num_latents

        # Keys and Values come from concatenated [latents, vis_tokens]
        kv_input = np.concatenate(
            [queries, vis_tokens], axis=1
        )  # (B, K + N_v, text_dim)
        S_kv = kv_input.shape[1]

        Q = (
            np.dot(queries, self.w_q)
            .reshape(B, K, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        K_mat = (
            np.dot(kv_input, self.w_k)
            .reshape(B, S_kv, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        V_mat = (
            np.dot(kv_input, self.w_v)
            .reshape(B, S_kv, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        scores = np.matmul(Q, K_mat.swapaxes(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, H, K, S_kv)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.maximum(
            np.sum(exp_scores, axis=-1, keepdims=True), 1e-12
        )

        context = np.matmul(attn_weights, V_mat)  # (B, H, K, head_dim)
        context = context.swapaxes(1, 2).reshape(B, K, self.text_dim)
        return np.dot(context, self.w_out)
