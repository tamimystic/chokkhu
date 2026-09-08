"""Wav2Vec 2.0 Self-Supervised Speech Feature Extractor & Transformer (Baevski et al., 2020)."""

from __future__ import annotations

from typing import Any, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module
from ...dl.activations import GELU
from ..conv1d import Conv1D
from ...nlp.transformer_blocks import TransformerEncoderLayer


class Wav2Vec2(Module, ChokkhuModel):
    """Wav2Vec 2.0 Architecture from First Principles."""

    def __init__(
        self,
        num_classes: int = 32,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Temporal Convolutional Feature Encoder (Downsamples raw 1D audio waveform)
        self.conv1 = Conv1D(1, 64, kernel_size=10, stride=5, padding=2)
        self.conv2 = Conv1D(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv1D(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.act = GELU()
        self.dropout = Dropout(dropout)

        # 2. Positional Conv Embedding
        self.pos_conv = Conv1D(
            embed_dim, embed_dim, kernel_size=15, padding=7, groups=1
        )

        # 3. Transformer Contextualizer Layers
        self.layers = [
            TransformerEncoderLayer(
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

        self.norm = LayerNorm(embed_dim)
        self.classifier = Linear(embed_dim, num_classes)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Forward pass for raw audio waveform tensor (batch_size, 1, num_samples) or (batch_size, num_samples)."""
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)

        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])

        # 1. Feature encoder
        h1 = self.act(self.conv1(x))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h2))

        # 2. Positional conv embedding
        pos = self.act(self.pos_conv(h3))
        feat = h3 + pos

        feat_t = Tensor(feat.data.swapaxes(1, 2), requires_grad=x.requires_grad)
        h = self.dropout(feat_t)

        # 3. Transformer Contextualizer
        for layer in self.layers:
            h = layer(h)

        N, T, D = h.shape
        h_norm = self.norm(h.reshape(-1, D)).reshape(N, T, D)
        logits = self.classifier(h_norm.reshape(-1, D)).reshape(N, T, self.num_classes)
        return logits

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Wav2Vec2:
        from ...dl.sequential import Sequential

        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X).data
        if out.ndim == 3:
            return np.argmax(out, axis=-1)
        elif out.ndim == 2:
            return np.argmax(out, axis=1)
        return out.flatten()
