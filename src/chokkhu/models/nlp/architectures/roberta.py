"""RoBERTa (Robustly Optimized BERT Approach) Architecture from Scratch (Liu et al., 2019)."""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module
from ...dl.activations import GELU
from ..embeddings import LearnedPositionalEmbedding, TokenEmbedding
from ..transformer_blocks import TransformerEncoderLayer


class RoBERTa(Module, ChokkhuModel):
    """RoBERTa Encoder Model from First Principles."""

    def __init__(
        self,
        vocab_size: int = 50265,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        pad_token_id: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # RoBERTa uses Byte-Pair Encoding token embeddings and learned positional embeddings with padding offset
        self.token_embed = TokenEmbedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )
        self.pos_embed = LearnedPositionalEmbedding(
            max_seq_len + pad_token_id + 1, embed_dim
        )
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

        self.layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,  # Modern RoBERTa Pre-LN
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        tok_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed(tok_emb)
        embeddings = pos_emb

        N, S, D = embeddings.shape
        x = self.norm(embeddings.reshape(-1, D)).reshape(N, S, D)
        x = self.dropout(x)

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, np.newaxis, np.newaxis, :].astype(bool)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class RobertaForSequenceClassification(Module, ChokkhuModel):
    """RoBERTa with Classification Head on <s> Token."""

    def __init__(
        self,
        vocab_size: int = 50265,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.roberta = RoBERTa(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.dense = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout)
        self.classifier = Linear(embed_dim, num_classes)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        encoder_out = self.roberta(input_ids, attention_mask=attention_mask)
        first_token = Tensor(
            encoder_out.data[:, 0, :], requires_grad=encoder_out.requires_grad
        )
        x = self.dropout(first_token)
        x = np.tanh(self.dense(x).data)
        x_t = self.dropout(Tensor(x, requires_grad=encoder_out.requires_grad))
        return self.classifier(x_t)

    def fit(
        self, X: Any, y: Any = None, **kwargs: Any
    ) -> RobertaForSequenceClassification:
        from ...dl.sequential import Sequential

        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.int64), requires_grad=False)
        out = self.forward(X).data
        if out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out.flatten()


class RobertaForMaskedLM(Module, ChokkhuModel):
    """RoBERTa for Masked Language Modeling."""

    def __init__(
        self,
        vocab_size: int = 50265,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.roberta = RoBERTa(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.dense = Linear(embed_dim, embed_dim)
        self.act = GELU()
        self.norm = LayerNorm(embed_dim)
        self.decoder = Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        encoder_out = self.roberta(input_ids, attention_mask=attention_mask)
        N, S, D = encoder_out.shape
        x_flat = encoder_out.reshape(-1, D)
        hidden = self.norm(self.act(self.dense(x_flat)))
        logits = self.decoder(hidden)
        return logits.reshape(N, S, -1)
