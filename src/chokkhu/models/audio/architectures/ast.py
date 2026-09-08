"""Audio Spectrogram Transformer (AST) for Audio Classification (Gong et al., 2021)."""

from __future__ import annotations

from typing import Any, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module, Parameter
from ...vision.conv_layers import Conv2D
from ...nlp.transformer_blocks import TransformerEncoderLayer


class AST(Module, ChokkhuModel):
    """Audio Spectrogram Transformer (AST) from First Principles."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        embed_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 384,
        patch_size: int = 16,
        stride: int = 16,
        max_patches: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride

        # 2D Patch Projection Conv2D
        self.patch_embed = Conv2D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
        )

        # Learnable [CLS] token and positional embedding
        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = Parameter(
            np.random.randn(1, max_patches + 1, embed_dim) * 0.02
        )
        self.dropout = Dropout(dropout)

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
        self.head = Linear(embed_dim, num_classes)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Forward pass for spectrogram tensor (batch_size, 1, freq_bins, time_frames)."""
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)

        if x.ndim == 3:
            # (N, F, T) -> (N, 1, F, T)
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # Patch embedding: (N, embed_dim, H_out, W_out)
        patches = self.patch_embed(x)
        N, D, H_out, W_out = patches.shape
        num_patches = H_out * W_out

        # Flatten patches: (N, num_patches, D)
        patches_flat = patches.data.reshape(N, D, num_patches).swapaxes(1, 2)

        # Prepend [CLS] token
        cls_tokens = np.tile(self.cls_token.data, (N, 1, 1))
        tokens = np.concatenate(
            [cls_tokens, patches_flat], axis=1
        )  # (N, num_patches + 1, D)

        # Add positional embedding
        seq_len = tokens.shape[1]
        tokens = tokens + self.pos_embed.data[:, :seq_len, :]

        h = self.dropout(Tensor(tokens, requires_grad=x.requires_grad))
        for layer in self.layers:
            h = layer(h)

        h_norm = self.norm(h.reshape(-1, D)).reshape(N, seq_len, D)
        cls_out = Tensor(h_norm.data[:, 0, :], requires_grad=x.requires_grad)
        logits = self.head(cls_out)
        return logits

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> AST:
        from ...dl.sequential import Sequential

        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X).data
        if out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out.flatten()


AudioSpectrogramTransformer = AST
