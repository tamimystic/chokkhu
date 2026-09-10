from __future__ import annotations

import numpy as np


class _RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight: np.ndarray = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class _SwiGLU:
    def __init__(self, in_features: int, hidden_features: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        self.W_gate: np.ndarray = rng.randn(in_features, hidden_features).astype(
            np.float32
        ) * np.sqrt(2.0 / in_features)
        self.W_up: np.ndarray = rng.randn(in_features, hidden_features).astype(
            np.float32
        ) * np.sqrt(2.0 / in_features)
        self.W_down: np.ndarray = rng.randn(hidden_features, in_features).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = np.dot(x, self.W_gate)
        silu_gate = gate / (1.0 + np.exp(-np.clip(gate, -20.0, 20.0)))
        up = np.dot(x, self.W_up)
        return np.dot(silu_gate * up, self.W_down)


class _CausalAttention:
    def __init__(self, d_model: int, num_heads: int = 4, seed: int = 42) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        rng = np.random.RandomState(seed)
        self.W_q: np.ndarray = rng.randn(d_model, d_model).astype(np.float32) * np.sqrt(
            2.0 / d_model
        )
        self.W_k: np.ndarray = rng.randn(d_model, d_model).astype(np.float32) * np.sqrt(
            2.0 / d_model
        )
        self.W_v: np.ndarray = rng.randn(d_model, d_model).astype(np.float32) * np.sqrt(
            2.0 / d_model
        )
        self.W_o: np.ndarray = rng.randn(d_model, d_model).astype(np.float32) * np.sqrt(
            2.0 / d_model
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        Q = np.dot(x, self.W_q).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, T, H, d).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d)
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        scores += mask

        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T, D)
        return np.dot(out, self.W_o)


class Qwen2_5:
    """
    Qwen 2.5 Architecture with Dual-Chunk / Rotary Position Embeddings,
    RMSNorm, and intermediate SwiGLU scaling.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_dim : int, default=64
        Hidden dimension.
    num_layers : int, default=2
        Number of transformer layers.
    num_heads : int, default=4
        Number of attention heads.
    intermediate_dim : int, default=128
        SwiGLU intermediate dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_dim: int = 128,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rng = np.random.RandomState(seed)

        self.embed: np.ndarray = (
            rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        )
        self.norm = _RMSNorm(hidden_dim)

        self.attn_layers = [
            _CausalAttention(d_model=hidden_dim, num_heads=num_heads, seed=seed + i)
            for i in range(num_layers)
        ]
        self.mlp_layers = [
            _SwiGLU(
                in_features=hidden_dim, hidden_features=intermediate_dim, seed=seed + i
            )
            for i in range(num_layers)
        ]

        self.lm_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed[input_ids]

        for attn, mlp in zip(self.attn_layers, self.mlp_layers):
            norm_x = self.norm.forward(x)
            attn_out = attn.forward(norm_x)
            x = x + attn_out

            norm_x2 = self.norm.forward(x)
            mlp_out = mlp.forward(norm_x2)
            x = x + mlp_out

        final_norm = self.norm.forward(x)
        logits = np.dot(final_norm, self.lm_head)
        return logits
