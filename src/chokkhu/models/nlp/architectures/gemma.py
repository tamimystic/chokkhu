"""Gemma (Google DeepMind) Architecture with GeGLU, RMSNorm with Offset, and RoPE (Gemma Team, 2024)."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Linear, Module
from ...dl.activations import GELU
from ..embeddings import RoPE, TokenEmbedding
from ..attention import GroupedQueryAttention
from ..transformer_blocks import RMSNorm


class GeGLU(Module):
    """GeGLU (GELU Gated Linear Unit) for Gemma Architecture."""

    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.w_gate = Linear(in_features, hidden_features, bias=bias)
        self.w_up = Linear(in_features, hidden_features, bias=bias)
        self.w_down = Linear(hidden_features, in_features, bias=bias)
        self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        N, S, D = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])
        x_flat = x.reshape(-1, D)
        gate = self.act(self.w_gate(x_flat))
        up = self.w_up(x_flat)
        hidden = gate * up
        out = self.w_down(hidden)
        return out.reshape(x.shape)


class GemmaBlock(Module):
    """Gemma Transformer Decoder Block with RMSNorm, GQA, and GeGLU."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        hidden_dim: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(embed_dim, eps=eps)
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_query_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        )
        self.post_attention_layernorm = RMSNorm(embed_dim, eps=eps)
        self.feed_forward = GeGLU(
            in_features=embed_dim, hidden_features=hidden_dim, bias=False
        )

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        attn_out = self.attn(self.input_layernorm(x), mask=mask, is_causal=True)
        h = x + attn_out
        ffn_out = self.feed_forward(self.post_attention_layernorm(h))
        return h + ffn_out


class Gemma(Module, ChokkhuModel):
    """Gemma Architecture from Scratch."""

    def __init__(
        self,
        vocab_size: int = 256000,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        hidden_dim: int = 1024,
        max_seq_len: int = 2048,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.rope = RoPE(dim=embed_dim // num_heads, max_seq_len=max_seq_len)
        self.embed_scale = np.sqrt(float(embed_dim))

        self.blocks = [
            GemmaBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)

        self.norm = RMSNorm(embed_dim, eps=eps)
        self.output = Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        # Scale embedding by sqrt(d_model)
        h = self.token_embed(input_ids) * self.embed_scale
        for block in self.blocks:
            h = block(h, mask=attention_mask)

        normed = self.norm(h)
        N, S, D = normed.shape
        logits = self.output(normed.reshape(-1, D))
        return logits.reshape(N, S, -1)


GemmaForCausalLM = Gemma
