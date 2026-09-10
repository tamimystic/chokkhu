from __future__ import annotations

import numpy as np


class RWKV6TimeMix:
    """
    RWKV-v6 Time-Mixing Block (Linear Attention with Time-Decay Schedule).

    Parameters
    ----------
    hidden_dim : int
        Dimension of token embeddings.
    n_heads : int, default=4
        Number of attention heads.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        rng = np.random.RandomState(seed)

        # Projections for Receptance (R), Key (K), Value (V), Gate (G)
        self.W_r: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_k: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_v: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_g: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_o: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)

        # Time decay parameters w and initial bonus u
        self.time_decay: np.ndarray = rng.uniform(
            -6.0, -1.0, size=(n_heads, self.head_dim)
        ).astype(np.float32)
        self.time_first: np.ndarray = (
            rng.randn(n_heads, self.head_dim).astype(np.float32) * 0.5
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        x: [B, T, hidden_dim]
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim

        # Receptance, Key, Value, Gate
        R = 1.0 / (1.0 + np.exp(-np.dot(x, self.W_r)))  # Sigmoid receptance
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        G = 1.0 / (1.0 + np.exp(-np.dot(x, self.W_g)))  # Output gate

        R = R.reshape(B, T, H, d).transpose(0, 2, 1, 3)  # [B, H, T, d]
        K = K.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        u = self.time_first  # [H, d]

        # Vectorized WKV computation
        out = np.zeros((B, H, T, d), dtype=np.float32)

        for b in range(B):
            for h in range(H):
                state: np.ndarray = np.zeros((d, d), dtype=np.float32)
                decay_vec = np.exp(self.time_decay[h]).reshape(d, 1)  # [d, 1]

                for t in range(T):
                    k_t = K[b, h, t].reshape(d, 1)
                    v_t = V[b, h, t].reshape(1, d)
                    r_t = R[b, h, t].reshape(1, d)

                    # Output at step t
                    wkv_t = np.dot(r_t, state + u[h].reshape(d, 1) * np.dot(k_t, v_t))
                    out[b, h, t] = wkv_t.ravel()

                    # State update with exponential decay
                    state = state * decay_vec + np.dot(k_t, v_t)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = out * G  # Output gating
        return np.dot(out, self.W_o)


class RWKV6:
    """
    Sovereign RWKV-v6 Linear Attention Language Model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_dim : int, default=64
        Hidden embedding dimension.
    num_layers : int, default=2
        Number of RWKV layers.
    n_heads : int, default=4
        Number of attention heads.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rng = np.random.RandomState(seed)
        self.embed: np.ndarray = (
            rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        )
        self.blocks = [
            RWKV6TimeMix(hidden_dim, n_heads=n_heads, seed=seed + i)
            for i in range(num_layers)
        ]
        self.lm_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed[input_ids]  # [B, T, H]
        for block in self.blocks:
            x = x + block.forward(x)
            # LayerNorm
            x = (x - np.mean(x, axis=-1, keepdims=True)) / (
                np.std(x, axis=-1, keepdims=True) + 1e-6
            )
        logits = np.dot(x, self.lm_head)
        return logits


class RetNetRetention:
    """
    RetNet Multi-Scale Retention (MSR) Layer.
    Implements causal retention with explicit exponential decay mask.

    Parameters
    ----------
    hidden_dim : int
        Model dimension.
    n_heads : int, default=4
        Number of retention heads.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        rng = np.random.RandomState(seed)
        self.W_q: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_k: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_v: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_o: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)

        # Multi-scale head decays: gamma_h = 1 - 2^(-5 - h)
        self.gammas = [1.0 - 2.0 ** (-5.0 - h) for h in range(n_heads)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim

        Q = (
            np.dot(x, self.W_q).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        )  # [B, H, T, d]
        K = np.dot(x, self.W_k).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(B, T, H, d).transpose(0, 2, 1, 3)

        out = np.zeros((B, H, T, d), dtype=np.float32)

        for h in range(H):
            gamma = self.gammas[h]
            # Construct causal decay matrix D_ij = gamma^(i-j) if i >= j else 0
            indices = np.arange(T)
            diff = indices[:, None] - indices[None, :]
            decay_mask = np.where(diff >= 0, gamma**diff, 0.0).astype(np.float32)

            # Retention matrix: (Q K^T) * D
            q_h = Q[:, h]  # [B, T, d]
            k_h = K[:, h]  # [B, T, d]
            v_h = V[:, h]  # [B, T, d]

            scores = np.matmul(q_h, k_h.transpose(0, 2, 1)) / np.sqrt(d)
            scores = scores * decay_mask  # [B, T, T]

            out[:, h] = np.matmul(scores, v_h)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return np.dot(out, self.W_o)


class RetNet:
    """
    Retentive Network (RetNet) Architecture.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_dim : int, default=64
        Hidden representation dimension.
    num_layers : int, default=2
        Number of RetNet layers.
    n_heads : int, default=4
        Number of retention heads.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        self.embed: np.ndarray = (
            rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        )
        self.blocks = [
            RetNetRetention(hidden_dim, n_heads=n_heads, seed=seed + i)
            for i in range(num_layers)
        ]
        self.lm_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed[input_ids]
        for block in self.blocks:
            x = x + block.forward(x)
            x = (x - np.mean(x, axis=-1, keepdims=True)) / (
                np.std(x, axis=-1, keepdims=True) + 1e-6
            )
        return np.dot(x, self.lm_head)
