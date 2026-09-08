"""Mistral 7B Architecture with SWA, GQA, and SwiGLU (Jiang et al., 2023)."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Linear, Module
from ..embeddings import RoPE, TokenEmbedding
from ..attention import SlidingWindowAttention
from ..transformer_blocks import RMSNorm, SwiGLU


class MistralBlock(Module):
    """Mistral Decoder Block combining RMSNorm, Sliding Window Attention (SWA/GQA), and SwiGLU."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        window_size: int = 256,
        hidden_dim: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=eps)
        self.attn = SlidingWindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            bias=False,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=eps)
        self.feed_forward = SwiGLU(
            in_features=embed_dim, hidden_features=hidden_dim, bias=False
        )

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), mask=mask)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        return h + ffn_out


class Mistral(Module, ChokkhuModel):
    """Mistral Causal Language Model Architecture."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        window_size: int = 256,
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

        self.blocks = [
            MistralBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                window_size=window_size,
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
        h = self.token_embed(input_ids)
        for block in self.blocks:
            h = block(h, mask=attention_mask)

        normed = self.norm(h)
        N, S, D = normed.shape
        logits = self.output(normed.reshape(-1, D))
        return logits.reshape(N, S, -1)


MistralForCausalLM = Mistral
