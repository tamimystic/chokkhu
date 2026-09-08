"""Embedding Layers and Positional Encodings for NLP."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from chokkhu.core.tensor import Function, Tensor
from ..dl.layers import Module, Parameter


class EmbeddingFunction(Function):
    """Autograd Function for Token Embedding Lookup."""

    def __init__(
        self,
        indices: np.ndarray,
        vocab_size: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.indices = indices
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

    def forward(self, weight: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return weight[self.indices]

    def backward(self, gy: np.ndarray) -> np.ndarray:
        grad = np.zeros((self.vocab_size, gy.shape[-1]), dtype=np.float64)
        np.add.at(grad, self.indices, gy)
        if self.padding_idx is not None and 0 <= self.padding_idx < self.vocab_size:
            grad[self.padding_idx] = 0.0
        return grad


class TokenEmbedding(Module):
    """Learnable Token Embedding Layer with Autograd Support."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        # Normal initialization with std = 1.0 / sqrt(embed_dim)
        weight_data = np.random.randn(vocab_size, embed_dim) * (
            1.0 / np.sqrt(embed_dim)
        )
        if padding_idx is not None and 0 <= padding_idx < vocab_size:
            weight_data[padding_idx] = 0.0
        self.weight = Parameter(weight_data)

    def forward(self, x: Any) -> Tensor:
        """Forward lookup of token embeddings.

        Args:
            x: Integer array or Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        if isinstance(x, Tensor):
            indices = x.data.astype(np.int64)
        else:
            indices = np.asarray(x, dtype=np.int64)

        fn = EmbeddingFunction(indices, self.vocab_size, self.padding_idx)
        return fn(self.weight)


class SinusoidalPositionalEncoding(Module):
    """Fixed Sinusoidal Positional Encoding (Vaswani et al., 2017)."""

    def __init__(self, max_seq_len: int = 5000, embed_dim: int = 512) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        pe: np.ndarray = np.zeros((max_seq_len, embed_dim), dtype=np.float64)
        position: np.ndarray = np.arange(0, max_seq_len, dtype=np.float64)[
            :, np.newaxis
        ]
        div_term = np.exp(
            np.arange(0, embed_dim, 2, dtype=np.float64)
            * -(np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]  # (1, max_seq_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Add sinusoidal positional encoding to input token embeddings."""
        seq_len = x.shape[1]
        pe_slice = Tensor(self.pe[:, :seq_len, :], requires_grad=False)
        return x + pe_slice


class LearnedPositionalEmbedding(Module):
    """Learnable Positional Embedding Matrix (BERT / GPT Style)."""

    def __init__(self, max_seq_len: int = 512, embed_dim: int = 768) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.weight = Parameter(np.random.randn(max_seq_len, embed_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional embedding to input token embeddings."""
        seq_len = x.shape[1]
        pos_slice = Tensor(
            self.weight.data[:seq_len, :][np.newaxis, :, :],
            requires_grad=self.weight.requires_grad,
        )
        return x + pos_slice


class RotaryPositionEmbedding:
    """Rotary Position Embedding (RoPE) for Modern LLMs (Su et al., 2021)."""

    def __init__(
        self, dim: int, max_seq_len: int = 4096, base: float = 10000.0
    ) -> None:
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        t: np.ndarray = np.arange(max_seq_len, dtype=np.float64)
        freqs = np.outer(t, inv_freq)
        self.cos_cached = np.cos(freqs)  # (max_seq_len, dim // 2)
        self.sin_cached = np.sin(freqs)  # (max_seq_len, dim // 2)

    def apply_rope(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply 2D complex rotations to Query or Key numpy array.

        Args:
            x: Array of shape (batch, num_heads, seq_len, head_dim)
            seq_len: sequence length
        """
        cos = self.cos_cached[:seq_len, :][
            np.newaxis, np.newaxis, :, :
        ]  # (1, 1, seq_len, head_dim/2)
        sin = self.sin_cached[:seq_len, :][np.newaxis, np.newaxis, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        out = np.empty_like(x)
        out[..., 0::2] = rotated_1
        out[..., 1::2] = rotated_2
        return out


RoPE = RotaryPositionEmbedding


class ALiBi:
    """Attention with Linear Biases (ALiBi) (Press et al., 2022)."""

    @staticmethod
    def get_slopes(num_heads: int) -> np.ndarray:
        """Calculate geometric slopes for ALiBi heads."""

        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(np.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if np.log2(num_heads).is_integer():
            return np.array(get_slopes_power_of_2(num_heads), dtype=np.float64)
        else:
            closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : num_heads - closest_power_of_2
            ]
            return np.array(slopes_a + slopes_b, dtype=np.float64)

    @classmethod
    def get_bias(cls, num_heads: int, seq_len: int) -> np.ndarray:
        """Return ALiBi attention bias matrix (1, num_heads, seq_len, seq_len)."""
        slopes = cls.get_slopes(num_heads)[:, np.newaxis, np.newaxis]  # (H, 1, 1)
        positions: np.ndarray = np.arange(seq_len, dtype=np.float64)
        distance = (
            positions[np.newaxis, :] - positions[:, np.newaxis]
        )  # (seq_len, seq_len)
        alibi_bias = slopes * distance[np.newaxis, :, :]  # (H, seq_len, seq_len)
        return alibi_bias[np.newaxis, :, :, :]  # (1, H, seq_len, seq_len)
