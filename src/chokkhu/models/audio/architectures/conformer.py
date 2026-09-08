"""Conformer (Convolution-augmented Transformer for Speech Recognition) Architecture (Gulati et al., 2020)."""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module
from ...dl.activations import SiLU
from ..conv1d import BatchNorm1D, Conv1D, DepthwiseConv1D
from ...nlp.attention import MultiHeadAttention


class ConformerConvModule(Module):
    """Conformer Convolution Module: LayerNorm -> Pointwise Conv -> GLU -> 1D Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(dim)
        self.pointwise_conv1 = Conv1D(dim, 2 * dim, kernel_size=1, bias=True)
        self.depthwise_conv = DepthwiseConv1D(
            dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        self.batch_norm = BatchNorm1D(dim)
        self.act = SiLU()
        self.pointwise_conv2 = Conv1D(dim, dim, kernel_size=1, bias=True)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, L, D)
        N, L, D = x.shape
        x_norm = self.norm(x.reshape(-1, D)).reshape(N, L, D)

        # Transpose to (N, D, L) for 1D convolutions
        x_conv = Tensor(x_norm.data.swapaxes(1, 2), requires_grad=x.requires_grad)
        pw1 = self.pointwise_conv1(x_conv)  # (N, 2D, L)

        # Gated Linear Unit (GLU): split along channel axis into A and B -> A * sigmoid(B)
        a = pw1.data[:, :D, :]
        b = pw1.data[:, D:, :]
        glu_out = a * (1.0 / (1.0 + np.exp(-b)))
        glu_tensor = Tensor(glu_out, requires_grad=x.requires_grad)

        dw = self.depthwise_conv(glu_tensor)
        bn = self.batch_norm(dw)
        act_out = self.act(bn)
        pw2 = self.pointwise_conv2(act_out)
        drop_out = self.dropout(pw2)

        # Transpose back to (N, L, D)
        out = Tensor(drop_out.data.swapaxes(1, 2), requires_grad=x.requires_grad)
        return out


class ConformerBlock(Module):
    """Conformer Block with Macaron-Style Feed-Forward, Multi-Head Self-Attention, and Convolution Module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_dim: int = 512,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. First FFN (Macaron half-step)
        self.ffn1_norm = LayerNorm(dim)
        self.ffn1_linear1 = Linear(dim, ffn_dim)
        self.ffn1_act = SiLU()
        self.ffn1_linear2 = Linear(ffn_dim, dim)
        self.ffn1_dropout = Dropout(dropout)

        # 2. Multi-Head Self-Attention
        self.attn_norm = LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)

        # 3. Convolution Module
        self.conv_module = ConformerConvModule(
            dim, kernel_size=conv_kernel_size, dropout=dropout
        )

        # 4. Second FFN (Macaron half-step)
        self.ffn2_norm = LayerNorm(dim)
        self.ffn2_linear1 = Linear(dim, ffn_dim)
        self.ffn2_act = SiLU()
        self.ffn2_linear2 = Linear(ffn_dim, dim)
        self.ffn2_dropout = Dropout(dropout)

        # Final LayerNorm
        self.final_norm = LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        N, L, D = x.shape
        # 1. FFN 1
        x_norm1 = self.ffn1_norm(x.reshape(-1, D)).reshape(N, L, D)
        ffn1 = self.ffn1_linear2(
            self.ffn1_act(self.ffn1_linear1(x_norm1.reshape(-1, D)))
        ).reshape(N, L, D)
        x = x + 0.5 * self.ffn1_dropout(ffn1)

        # 2. Attention
        x_norm2 = self.attn_norm(x.reshape(-1, D)).reshape(N, L, D)
        attn_out = self.self_attn(x_norm2, mask=mask)
        x = x + self.attn_dropout(attn_out)

        # 3. Conv Module
        conv_out = self.conv_module(x)
        x = x + conv_out

        # 4. FFN 2
        x_norm3 = self.ffn2_norm(x.reshape(-1, D)).reshape(N, L, D)
        ffn2 = self.ffn2_linear2(
            self.ffn2_act(self.ffn2_linear1(x_norm3.reshape(-1, D)))
        ).reshape(N, L, D)
        x = x + 0.5 * self.ffn2_dropout(ffn2)

        # Final Norm
        x = self.final_norm(x.reshape(-1, D)).reshape(N, L, D)
        return x


class Conformer(Module, ChokkhuModel):
    """Conformer Architecture for Audio and Speech Recognition from Scratch."""

    def __init__(
        self,
        in_features: int = 80,
        num_classes: int = 32,
        embed_dim: int = 144,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 576,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Input linear projection
        self.input_proj = Linear(in_features, embed_dim)
        self.dropout = Dropout(dropout)

        self.blocks = [
            ConformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)

        self.classifier = Linear(embed_dim, num_classes)

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        """Forward pass for audio features / spectrogram (batch_size, time_frames, in_features)."""
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)

        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        N, T, F = x.shape
        h = self.dropout(self.input_proj(x.reshape(-1, F))).reshape(
            N, T, self.embed_dim
        )

        for block in self.blocks:
            h = block(h, mask=attention_mask)

        # Output logits per time frame (N, T, num_classes)
        logits = self.classifier(h.reshape(-1, self.embed_dim)).reshape(
            N, T, self.num_classes
        )
        return logits

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Conformer:
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
