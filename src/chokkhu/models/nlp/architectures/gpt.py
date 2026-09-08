"""GPT (Generative Pre-trained Transformer) Architecture from Scratch (Radford et al., 2018/2019)."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module
from ..embeddings import LearnedPositionalEmbedding, TokenEmbedding
from ..transformer_blocks import TransformerDecoderLayer


class GPT(Module, ChokkhuModel):
    """Autoregressive Causal GPT Architecture from First Principles."""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len, embed_dim)
        self.dropout = Dropout(dropout)

        self.layers = [
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

        self.ln_f = LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        tok_emb = self.token_embed(input_ids)
        x = self.dropout(self.pos_embed(tok_emb))

        for layer in self.layers:
            x = layer(x, memory=None, tgt_mask=attention_mask)

        N, S, D = x.shape
        x_norm = self.ln_f(x.reshape(-1, D))
        return x_norm.reshape(N, S, D)


class GPTForCausalLM(Module, ChokkhuModel):
    """GPT with Language Modeling Head for Next-Token Generation."""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gpt = GPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        hidden = self.gpt(input_ids, attention_mask=attention_mask)
        N, S, D = hidden.shape
        logits = self.lm_head(hidden.reshape(-1, D))
        return logits.reshape(N, S, -1)


MiniGPT = GPTForCausalLM
