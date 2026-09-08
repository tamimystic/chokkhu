"""LLaMA (Large Language Model Meta AI) Architecture from Scratch (Touvron et al., 2023)."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Linear, Module
from ..embeddings import RoPE, TokenEmbedding
from ..attention import GroupedQueryAttention
from ..transformer_blocks import RMSNorm, SwiGLU


class LLaMABlock(Module):
    """LLaMA Decoder Block with RMSNorm, RoPE, GQA, and SwiGLU."""

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        hidden_dim: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=eps)
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=eps)
        self.feed_forward = SwiGLU(
            in_features=embed_dim, hidden_features=hidden_dim, bias=False
        )

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), mask=mask, is_causal=True)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        return h + ffn_out


class LLaMA(Module, ChokkhuModel):
    """Modern LLaMA / Mistral Architecture from Scratch."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        hidden_dim: int = 1024,
        max_seq_len: int = 2048,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.rope = RoPE(dim=embed_dim // num_query_heads, max_seq_len=max_seq_len)

        self.blocks = [
            LLaMABlock(
                embed_dim=embed_dim,
                num_query_heads=num_query_heads,
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
        h = self.token_embed(input_ids)
        for block in self.blocks:
            h = block(h, mask=attention_mask)

        normed = self.norm(h)
        N, S, D = normed.shape
        logits = self.output(normed.reshape(-1, D))
        return logits.reshape(N, S, -1)


LlamaForCausalLM = LLaMA
