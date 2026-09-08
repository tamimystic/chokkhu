"""Scaled Dot-Product Attention, Multi-Head Attention, Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and Sliding Window Attention (SWA)."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Linear, Module
from .embeddings import RotaryPositionEmbedding


class ScaledDotProductAttention(Module):
    """Vectorized Scaled Dot-Product Attention with Masking and Log-Sum-Exp Stability."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Scaled Dot-Product Attention.

        Args:
            q: Query array of shape (batch_size, num_heads, seq_len_q, head_dim)
            k: Key array of shape (batch_size, num_heads, seq_len_k, head_dim)
            v: Value array of shape (batch_size, num_heads, seq_len_k, head_dim)
            mask: Optional boolean or additive mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
            scale: Optional scale factor (defaults to 1 / sqrt(head_dim))
        Returns:
            (context, attention_weights)
        """
        d_k = q.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(d_k)

        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale

        if mask is not None:
            if mask.dtype == bool:
                scores = np.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        output = np.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(Module):
    """Multi-Head Attention (MHA) Supporting Self-Attention and Cross-Attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[np.ndarray] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass for Multi-Head Attention.

        Args:
            query: (batch_size, seq_len_q, embed_dim)
            key: Optional (batch_size, seq_len_k, embed_dim), defaults to query
            value: Optional (batch_size, seq_len_k, embed_dim), defaults to key
            mask: Optional attention mask
            is_causal: If True, applies lower-triangular causal mask
        """
        if key is None:
            key = query
        if value is None:
            value = key

        N, seq_len_q, D = query.shape
        seq_len_k = key.shape[1]

        q = (
            self.q_proj(query.reshape(-1, D))
            .data.reshape(N, seq_len_q, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(key.reshape(-1, D))
            .data.reshape(N, seq_len_k, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(value.reshape(-1, D))
            .data.reshape(N, seq_len_k, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        if is_causal:
            causal_mask: np.ndarray = np.tril(
                np.ones((seq_len_q, seq_len_k), dtype=bool)
            )[np.newaxis, np.newaxis, :, :]
            if mask is not None:
                mask = mask & causal_mask
            else:
                mask = causal_mask

        context, _ = self.attention(q, k, v, mask=mask)
        # context: (N, num_heads, seq_len_q, head_dim) -> (N, seq_len_q, embed_dim)
        context = context.swapaxes(1, 2).reshape(N * seq_len_q, self.embed_dim)
        out = self.out_proj(Tensor(context, requires_grad=query.requires_grad))
        return out.reshape(N, seq_len_q, self.embed_dim)


class GroupedQueryAttention(Module):
    """Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) for Modern LLMs (Ainslie et al., 2023)."""

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_query_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_query_heads ({num_query_heads})"
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = embed_dim // num_query_heads

        self.q_proj = Linear(embed_dim, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass for Grouped Query Attention."""
        N, seq_len, D = x.shape
        q = (
            self.q_proj(x.reshape(-1, D))
            .data.reshape(N, seq_len, self.num_query_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(x.reshape(-1, D))
            .data.reshape(N, seq_len, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(x.reshape(-1, D))
            .data.reshape(N, seq_len, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        # Repeat KV heads to match query heads
        if self.num_queries_per_kv > 1:
            k = np.repeat(k, self.num_queries_per_kv, axis=1)
            v = np.repeat(v, self.num_queries_per_kv, axis=1)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale

        if is_causal:
            causal_mask: np.ndarray = np.tril(np.ones((seq_len, seq_len), dtype=bool))[
                np.newaxis, np.newaxis, :, :
            ]
            scores = np.where(causal_mask, scores, -1e9)

        if mask is not None:
            if mask.dtype == bool:
                scores = np.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn_w = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        context = (
            np.matmul(attn_w, v).swapaxes(1, 2).reshape(N * seq_len, self.embed_dim)
        )
        out = self.out_proj(Tensor(context, requires_grad=x.requires_grad))
        return out.reshape(N, seq_len, self.embed_dim)


MultiQueryAttention = GroupedQueryAttention


class MultiHeadLatentAttention(Module):
    """Multi-Head Latent Attention (DeepSeek-V2 / DeepSeek-V3 MLA) with Low-Rank KV Compression."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        kv_latent_dim: int = 128,
        q_latent_dim: Optional[int] = None,
        rope_dim: int = 32,
        max_seq_len: int = 4096,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.rope_dim = rope_dim
        self.total_head_dim = head_dim + rope_dim

        # Query Projections
        if q_latent_dim is not None:
            self.q_down = Linear(embed_dim, q_latent_dim, bias=bias)
            self.q_up = Linear(q_latent_dim, num_heads * head_dim, bias=bias)
            self.q_rope = Linear(q_latent_dim, num_heads * rope_dim, bias=bias)
        else:
            self.q_down = None
            self.q_up = None
            self.q_proj = Linear(embed_dim, num_heads * head_dim, bias=bias)
            self.q_rope = Linear(embed_dim, num_heads * rope_dim, bias=bias)

        # Key-Value Projections via compressed latent
        self.kv_down = Linear(embed_dim, kv_latent_dim, bias=bias)
        self.k_up = Linear(kv_latent_dim, num_heads * head_dim, bias=bias)
        self.v_up = Linear(kv_latent_dim, num_heads * head_dim, bias=bias)
        self.k_rope = Linear(embed_dim, num_heads * rope_dim, bias=bias)

        # Output projection
        self.out_proj = Linear(num_heads * head_dim, embed_dim, bias=bias)
        self.rope = RotaryPositionEmbedding(dim=rope_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass of Multi-Head Latent Attention."""
        N, S, D = x.shape
        x_flat = x.reshape(-1, D)

        if (
            self.q_latent_dim is not None
            and self.q_down is not None
            and self.q_up is not None
        ):
            c_q = self.q_down(x_flat)
            q_c = (
                self.q_up(c_q)
                .data.reshape(N, S, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            q_r = (
                self.q_rope(c_q)
                .data.reshape(N, S, self.num_heads, self.rope_dim)
                .swapaxes(1, 2)
            )
        else:
            q_c = (
                self.q_proj(x_flat)
                .data.reshape(N, S, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            q_r = (
                self.q_rope(x_flat)
                .data.reshape(N, S, self.num_heads, self.rope_dim)
                .swapaxes(1, 2)
            )

        q_r = self.rope.apply_rope(q_r, seq_len=S)

        c_kv = self.kv_down(x_flat)
        k_c = (
            self.k_up(c_kv)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v_c = (
            self.v_up(c_kv)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k_r = (
            self.k_rope(x_flat)
            .data.reshape(N, S, self.num_heads, self.rope_dim)
            .swapaxes(1, 2)
        )
        k_r = self.rope.apply_rope(k_r, seq_len=S)

        q = np.concatenate([q_c, q_r], axis=-1)
        k = np.concatenate([k_c, k_r], axis=-1)

        scale = 1.0 / np.sqrt(self.total_head_dim)
        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale

        if is_causal:
            causal_mask: np.ndarray = np.tril(np.ones((S, S), dtype=bool))[
                np.newaxis, np.newaxis, :, :
            ]
            scores = np.where(causal_mask, scores, -1e9)

        if mask is not None:
            if mask.dtype == bool:
                scores = np.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn_w = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        context = (
            np.matmul(attn_w, v_c)
            .swapaxes(1, 2)
            .reshape(N * S, self.num_heads * self.head_dim)
        )
        out = self.out_proj(Tensor(context, requires_grad=x.requires_grad))
        return out.reshape(N, S, self.embed_dim)


DeepSeekMLA = MultiHeadLatentAttention


class SlidingWindowAttention(Module):
    """Sliding Window Attention (SWA) for Long-Context LLMs (Mistral 7B, Jiang et al., 2023)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        window_size: int = 256,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        """Forward pass with sliding window causal mask."""
        N, S, D = x.shape
        x_flat = x.reshape(-1, D)

        q = (
            self.q_proj(x_flat)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(x_flat)
            .data.reshape(N, S, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(x_flat)
            .data.reshape(N, S, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        if self.num_queries_per_kv > 1:
            k = np.repeat(k, self.num_queries_per_kv, axis=1)
            v = np.repeat(v, self.num_queries_per_kv, axis=1)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale

        row_idx = np.arange(S)[:, None]
        col_idx = np.arange(S)[None, :]
        sliding_causal = (col_idx <= row_idx) & (row_idx - col_idx < self.window_size)
        sliding_mask = sliding_causal[np.newaxis, np.newaxis, :, :]

        scores = np.where(sliding_mask, scores, -1e9)

        if mask is not None:
            if mask.dtype == bool:
                scores = np.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn_w = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        context = np.matmul(attn_w, v).swapaxes(1, 2).reshape(N * S, self.embed_dim)
        out = self.out_proj(Tensor(context, requires_grad=x.requires_grad))
        return out.reshape(N, S, self.embed_dim)
