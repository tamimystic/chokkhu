"""Sequence-to-Sequence (Seq2Seq / T5) Full Encoder-Decoder Transformer from Scratch."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, Linear, Module
from ..embeddings import SinusoidalPositionalEncoding, TokenEmbedding
from ..transformer_blocks import TransformerDecoderLayer, TransformerEncoderLayer


class Seq2SeqTransformer(Module, ChokkhuModel):
    """Full Encoder-Decoder Transformer Architecture for Translation & Text Generation."""

    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        embed_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim

        self.src_embed = TokenEmbedding(src_vocab_size, embed_dim)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(max_seq_len, embed_dim)
        self.dropout = Dropout(dropout)

        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(num_encoder_layers)
        ]
        for i, enc_l in enumerate(self.encoder_layers):
            setattr(self, f"enc_layer_{i}", enc_l)

        self.decoder_layers = [
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(num_decoder_layers)
        ]
        for i, dec_l in enumerate(self.decoder_layers):
            setattr(self, f"dec_layer_{i}", dec_l)

        self.fc_out = Linear(embed_dim, tgt_vocab_size)

    def encode(
        self,
        src: Union[np.ndarray, Tensor],
        src_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        x = self.dropout(self.pos_enc(self.src_embed(src)))
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return x

    def decode(
        self,
        tgt: Union[np.ndarray, Tensor],
        memory: Tensor,
        tgt_mask: Optional[np.ndarray] = None,
        memory_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        y = self.dropout(self.pos_enc(self.tgt_embed(tgt)))
        for layer in self.decoder_layers:
            y = layer(y, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return y

    def forward(
        self,
        src: Union[np.ndarray, Tensor],
        tgt: Union[np.ndarray, Tensor],
        src_mask: Optional[np.ndarray] = None,
        tgt_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        memory = self.encode(src, src_mask=src_mask)
        dec_out = self.decode(tgt, memory, tgt_mask=tgt_mask)
        N, S, D = dec_out.shape
        logits = self.fc_out(dec_out.reshape(-1, D))
        return logits.reshape(N, S, -1)


T5 = Seq2SeqTransformer
