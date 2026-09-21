"""Scaled Dot-Product Attention, Multi-Head Attention, Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and Sliding Window Attention (SWA)."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor, concat
from ..dl.layers import Linear, Module
from .embeddings import RotaryPositionEmbedding


class ScaledDotProductAttention(Module):
    """Vectorized Scaled Dot-Product Attention with Masking and Log-Sum-Exp Stability."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        q: Union[Tensor, np.ndarray],
        k: Union[Tensor, np.ndarray],
        v: Union[Tensor, np.ndarray],
        mask: Optional[Union[Tensor, np.ndarray]] = None,
        scale: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute Scaled Dot-Product Attention.

        Args:
            q: Query tensor/array of shape (batch_size, num_heads, seq_len_q, head_dim)
            k: Key tensor/array of shape (batch_size, num_heads, seq_len_k, head_dim)
            v: Value tensor/array of shape (batch_size, num_heads, seq_len_k, head_dim)
            mask: Optional boolean or additive mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
            scale: Optional scale factor (defaults to 1 / sqrt(head_dim))
        Returns:
            (context, attention_weights)
        """
        if not isinstance(q, Tensor):
            q = Tensor(q)
        if not isinstance(k, Tensor):
            k = Tensor(k)
        if not isinstance(v, Tensor):
            v = Tensor(v)

        d_k = q.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(d_k)

        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = (q @ k.swapaxes(-2, -1)) * scale

        if mask is not None:
            mask_arr = mask.data if isinstance(mask, Tensor) else mask
            if mask_arr.dtype == bool:
                mask_add = np.where(mask_arr, 0.0, -1e9).astype(np.float64)
                scores = scores + Tensor(mask_add, requires_grad=False)
            else:
                scores = scores + (
                    mask
                    if isinstance(mask, Tensor)
                    else Tensor(mask, requires_grad=False)
                )

        attn_weights = scores.softmax(axis=-1)
        output = attn_weights @ v
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
        mask: Optional[Union[Tensor, np.ndarray]] = None,
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
            .reshape(N, seq_len_q, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(key.reshape(-1, D))
            .reshape(N, seq_len_k, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(value.reshape(-1, D))
            .reshape(N, seq_len_k, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        if is_causal:
            causal_mask: np.ndarray = np.tril(
                np.ones((seq_len_q, seq_len_k), dtype=bool)
            )[np.newaxis, np.newaxis, :, :]
            if mask is not None:
                mask_arr = mask.data if isinstance(mask, Tensor) else mask
                mask = mask_arr & causal_mask
            else:
                mask = causal_mask

        context, _ = self.attention(q, k, v, mask=mask)
        # context: (N, num_heads, seq_len_q, head_dim) -> (N, seq_len_q, embed_dim)
        context = context.swapaxes(1, 2).reshape(N * seq_len_q, self.embed_dim)
        out = self.out_proj(context)
        return out.reshape(N, seq_len_q, self.embed_dim)


class GroupedQueryAttention(Module):
    """Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) for Modern LLMs (Ainslie et al., 2023)."""

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        bias: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 4096,
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
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
            if use_rope
            else None
        )

        self.q_proj = Linear(embed_dim, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Union[Tensor, np.ndarray]] = None,
        is_causal: bool = True,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """Forward pass for Grouped Query Attention."""
        N, seq_len, D = x.shape
        q = (
            self.q_proj(x.reshape(-1, D))
            .reshape(N, seq_len, self.num_query_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(x.reshape(-1, D))
            .reshape(N, seq_len, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(x.reshape(-1, D))
            .reshape(N, seq_len, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        offset = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            offset = past_k.shape[2]

        if self.use_rope and self.rope is not None:
            q_rope = self.rope.apply_rope(q, seq_len=seq_len, offset=offset)
            k_rope = self.rope.apply_rope(k, seq_len=seq_len, offset=offset)
            assert isinstance(q_rope, Tensor)
            assert isinstance(k_rope, Tensor)
            q = q_rope
            k = k_rope

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = concat([past_k, k], axis=2)
            v = concat([past_v, v], axis=2)

        new_kv_cache = (k, v)

        # Repeat KV heads to match query heads
        if self.num_queries_per_kv > 1:
            repeated_k = []
            repeated_v = []
            for i in range(self.num_kv_heads):
                k_head = k[:, i : i + 1, :, :]
                v_head = v[:, i : i + 1, :, :]
                for _ in range(self.num_queries_per_kv):
                    repeated_k.append(k_head)
                    repeated_v.append(v_head)
            k_eval = concat(repeated_k, axis=1)
            v_eval = concat(repeated_v, axis=1)
        else:
            k_eval = k
            v_eval = v

        total_seq_len = k_eval.shape[2]
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k_eval.swapaxes(-2, -1)) * scale

        if is_causal:
            if seq_len == total_seq_len:
                causal_mask: np.ndarray = np.tril(
                    np.ones((seq_len, total_seq_len), dtype=bool)
                )[np.newaxis, np.newaxis, :, :]
            else:
                causal_mask = np.ones((1, 1, seq_len, total_seq_len), dtype=bool)
            if mask is not None:
                mask_arr = mask.data if isinstance(mask, Tensor) else mask
                mask_arr = mask_arr & causal_mask
            else:
                mask_arr = causal_mask
        else:
            mask_arr = mask.data if isinstance(mask, Tensor) else mask

        if mask_arr is not None:
            if mask_arr.dtype == bool:
                mask_add = np.where(mask_arr, 0.0, -1e9).astype(np.float64)
                scores = scores + Tensor(mask_add, requires_grad=False)
            else:
                scores = scores + Tensor(mask_arr, requires_grad=False)

        attn_w = scores.softmax(axis=-1)
        context = (attn_w @ v_eval).swapaxes(1, 2).reshape(N * seq_len, self.embed_dim)
        out = self.out_proj(context).reshape(N, seq_len, self.embed_dim)

        if use_cache:
            return out, new_kv_cache
        return out


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
            self.q_proj = None
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
        mask: Optional[Union[Tensor, np.ndarray]] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass of Multi-Head Latent Attention."""
        N, S, D = x.shape
        x_flat = x.reshape(-1, D)

        if (
            self.q_latent_dim is not None
            and self.q_down is not None
            and self.q_up is not None
            and self.q_rope is not None
        ):
            c_q = self.q_down(x_flat)
            q_c = (
                self.q_up(c_q)
                .reshape(N, S, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            q_r = (
                self.q_rope(c_q)
                .reshape(N, S, self.num_heads, self.rope_dim)
                .swapaxes(1, 2)
            )
        else:
            assert self.q_proj is not None
            q_c = (
                self.q_proj(x_flat)
                .reshape(N, S, self.num_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            q_r = (
                self.q_rope(x_flat)
                .reshape(N, S, self.num_heads, self.rope_dim)
                .swapaxes(1, 2)
            )

        q_r_rope = self.rope.apply_rope(q_r, seq_len=S)
        assert isinstance(q_r_rope, Tensor)

        c_kv = self.kv_down(x_flat)
        k_c = (
            self.k_up(c_kv).reshape(N, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        )
        v_c = (
            self.v_up(c_kv).reshape(N, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        )
        k_r = (
            self.k_rope(x_flat)
            .reshape(N, S, self.num_heads, self.rope_dim)
            .swapaxes(1, 2)
        )
        k_r_rope = self.rope.apply_rope(k_r, seq_len=S)
        assert isinstance(k_r_rope, Tensor)

        q = concat([q_c, q_r_rope], axis=-1)
        k = concat([k_c, k_r_rope], axis=-1)

        scale = 1.0 / np.sqrt(self.total_head_dim)
        scores = (q @ k.swapaxes(-2, -1)) * scale

        if is_causal:
            causal_mask: np.ndarray = np.tril(np.ones((S, S), dtype=bool))[
                np.newaxis, np.newaxis, :, :
            ]
            if mask is not None:
                mask_arr = mask.data if isinstance(mask, Tensor) else mask
                mask_arr = mask_arr & causal_mask
            else:
                mask_arr = causal_mask
        else:
            mask_arr = mask.data if isinstance(mask, Tensor) else mask

        if mask_arr is not None:
            if mask_arr.dtype == bool:
                mask_add = np.where(mask_arr, 0.0, -1e9).astype(np.float64)
                scores = scores + Tensor(mask_add, requires_grad=False)
            else:
                scores = scores + Tensor(mask_arr, requires_grad=False)

        attn_w = scores.softmax(axis=-1)
        context = (
            (attn_w @ v_c).swapaxes(1, 2).reshape(N * S, self.num_heads * self.head_dim)
        )
        out = self.out_proj(context)
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
        mask: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:
        """Forward pass with sliding window causal mask."""
        N, S, D = x.shape
        x_flat = x.reshape(-1, D)

        q = (
            self.q_proj(x_flat)
            .reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k = (
            self.k_proj(x_flat)
            .reshape(N, S, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(x_flat)
            .reshape(N, S, self.num_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        if self.num_queries_per_kv > 1:
            repeated_k = []
            repeated_v = []
            for i in range(self.num_kv_heads):
                k_head = k[:, i : i + 1, :, :]
                v_head = v[:, i : i + 1, :, :]
                for _ in range(self.num_queries_per_kv):
                    repeated_k.append(k_head)
                    repeated_v.append(v_head)
            k = concat(repeated_k, axis=1)
            v = concat(repeated_v, axis=1)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.swapaxes(-2, -1)) * scale

        row_idx = np.arange(S)[:, None]
        col_idx = np.arange(S)[None, :]
        sliding_causal = (col_idx <= row_idx) & (row_idx - col_idx < self.window_size)
        sliding_mask = sliding_causal[np.newaxis, np.newaxis, :, :]

        if mask is not None:
            mask_arr = mask.data if isinstance(mask, Tensor) else mask
            effective_mask = sliding_mask & mask_arr
        else:
            effective_mask = sliding_mask

        mask_add = np.where(effective_mask, 0.0, -1e9).astype(np.float64)
        scores = scores + Tensor(mask_add, requires_grad=False)

        attn_w = scores.softmax(axis=-1)
        context = (attn_w @ v).swapaxes(1, 2).reshape(N * S, self.embed_dim)
        out = self.out_proj(context)
        return out.reshape(N, S, self.embed_dim)
