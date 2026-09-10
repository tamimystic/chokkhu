from __future__ import annotations

import numpy as np


class _RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight: np.ndarray = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class BitLinear:
    """
    BitNet 1.58b Ternary Quantized Linear Layer (BitLinear).
    Quantizes weights to {-1, 0, +1} using absmean scaling:
    W_q = round(clip(W / gamma, -1, 1)), gamma = mean(|W|).
    Quantizes activations to 8-bit precision:
    x_q = round(clip(x * 127 / gamma_x, -128, 127)).

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.RandomState(seed)
        self.weight: np.ndarray = rng.randn(in_features, out_features).astype(
            np.float32
        ) * np.sqrt(2.0 / in_features)
        self.norm = _RMSNorm(in_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with 1.58-bit ternary weight quantization.
        x: [..., in_features]
        """
        # 1. Normalize activation
        x_norm = self.norm.forward(x)

        # 2. Absmean ternary weight quantization W -> {-1, 0, 1}
        gamma_w = float(np.mean(np.abs(self.weight)) + 1e-8)
        w_scaled = self.weight / gamma_w
        w_quant = np.clip(np.round(w_scaled), -1.0, 1.0)

        # 3. 8-bit activation quantization
        gamma_x = np.max(np.abs(x_norm), axis=-1, keepdims=True) + 1e-8
        x_quant = np.clip(np.round(x_norm * 127.0 / gamma_x), -128.0, 127.0)

        # 4. Integer matrix multiplication dequantized by scale factors
        out = np.dot(x_quant, w_quant) * (gamma_w * gamma_x / 127.0)
        return out.astype(np.float32)


class BitNet158:
    """
    BitNet 1.58b Large Language Model Architecture.
    All dense linear transformations utilize 1.58-bit ternary quantized weights.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_dim : int, default=64
        Model dimension.
    num_layers : int, default=2
        Number of BitLinear transformer layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rng = np.random.RandomState(seed)
        self.embed: np.ndarray = (
            rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        )
        self.layers = [
            BitLinear(hidden_dim, hidden_dim, seed=seed + i) for i in range(num_layers)
        ]
        self.norm = _RMSNorm(hidden_dim)
        self.lm_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed[input_ids]
        for layer in self.layers:
            x = x + np.maximum(0.0, layer.forward(x))
        final_norm = self.norm.forward(x)
        return np.dot(final_norm, self.lm_head)
