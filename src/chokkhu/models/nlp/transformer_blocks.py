"""Transformer Encoder/Decoder Layers, RMSNorm, and Modern MLP Blocks (SwiGLU/GeGLU)."""

from __future__ import annotations

from typing import Optional
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Dropout, LayerNorm, Linear, Module, Parameter
from ..dl.activations import GELU, SiLU
from .attention import MultiHeadAttention


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones((dim,), dtype=np.float64))

    def forward(self, x: Tensor) -> Tensor:
        x_data = x.data
        rms = np.sqrt(np.mean(x_data**2, axis=-1, keepdims=True) + self.eps)
        normed = (x_data / rms) * self.weight.data
        return Tensor(normed, requires_grad=x.requires_grad)


class SwiGLU(Module):
    """SwiGLU (Swish Gated Linear Unit) Feed-Forward Block (Shazeer, 2020)."""

    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.w_gate = Linear(in_features, hidden_features, bias=bias)
        self.w_up = Linear(in_features, hidden_features, bias=bias)
        self.w_down = Linear(hidden_features, in_features, bias=bias)
        self.act = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
        N, S, D = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])
        x_flat = x.reshape(-1, D)

        gate = self.act(self.w_gate(x_flat))
        up = self.w_up(x_flat)
        hidden = gate * up
        out = self.w_down(hidden)
        return out.reshape(x.shape)


class TransformerEncoderLayer(Module):
    """Standard Transformer Encoder Layer (Pre-LN and Post-LN support)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.act = GELU()
        self.linear2 = Linear(dim_feedforward, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, src: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        N, S, D = src.shape
        if self.norm_first:
            # Pre-LN
            src_norm = self.norm1(src.reshape(-1, D)).reshape(N, S, D)
            attn_out = self.self_attn(src_norm, mask=mask)
            src = src + self.dropout(attn_out)

            src_norm2 = self.norm2(src.reshape(-1, D))
            ffn_out = self.linear2(self.act(self.linear1(src_norm2))).reshape(N, S, D)
            src = src + self.dropout(ffn_out)
        else:
            # Post-LN
            attn_out = self.self_attn(src, mask=mask)
            src = self.norm1((src + self.dropout(attn_out)).reshape(-1, D)).reshape(
                N, S, D
            )

            ffn_out = self.linear2(self.act(self.linear1(src.reshape(-1, D)))).reshape(
                N, S, D
            )
            src = self.norm2((src + self.dropout(ffn_out)).reshape(-1, D)).reshape(
                N, S, D
            )

        return src


class TransformerDecoderLayer(Module):
    """Transformer Decoder Layer with Causal Self-Attention and Optional Cross-Attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)

        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.act = GELU()
        self.linear2 = Linear(dim_feedforward, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[np.ndarray] = None,
        memory_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        N, S, D = tgt.shape
        if self.norm_first:
            # 1. Masked Causal Self-Attention
            tgt_norm = self.norm1(tgt.reshape(-1, D)).reshape(N, S, D)
            attn_out = self.self_attn(tgt_norm, is_causal=True, mask=tgt_mask)
            tgt = tgt + self.dropout(attn_out)

            # 2. Cross-Attention with Encoder Memory (if provided)
            if memory is not None:
                tgt_norm2 = self.norm2(tgt.reshape(-1, D)).reshape(N, S, D)
                cross_out = self.cross_attn(
                    query=tgt_norm2, key=memory, value=memory, mask=memory_mask
                )
                tgt = tgt + self.dropout(cross_out)

            # 3. Feedforward
            tgt_norm3 = self.norm3(tgt.reshape(-1, D))
            ffn_out = self.linear2(self.act(self.linear1(tgt_norm3))).reshape(N, S, D)
            tgt = tgt + self.dropout(ffn_out)
        else:
            attn_out = self.self_attn(tgt, is_causal=True, mask=tgt_mask)
            tgt = self.norm1((tgt + self.dropout(attn_out)).reshape(-1, D)).reshape(
                N, S, D
            )

            if memory is not None:
                cross_out = self.cross_attn(
                    query=tgt, key=memory, value=memory, mask=memory_mask
                )
                tgt = self.norm2(
                    (tgt + self.dropout(cross_out)).reshape(-1, D)
                ).reshape(N, S, D)

            ffn_out = self.linear2(self.act(self.linear1(tgt.reshape(-1, D)))).reshape(
                N, S, D
            )
            tgt = self.norm3((tgt + self.dropout(ffn_out)).reshape(-1, D)).reshape(
                N, S, D
            )

        return tgt
