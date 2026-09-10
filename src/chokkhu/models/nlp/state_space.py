from __future__ import annotations

import numpy as np


class MambaSSM:
    """
    Selective State Space Model (S6 / Mamba) Core Block in pure NumPy.
    Implements input-dependent parameter discretization (Delta, B, C) and linear-time scan.

    Parameters
    ----------
    d_model : int
        Model input/output dimension.
    d_state : int, default=16
        Latent SSM state dimension N.
    d_conv : int, default=4
        Temporal 1D convolution kernel size.
    expand : int, default=2
        Inner dimension expansion factor.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        rng = np.random.RandomState(seed)

        # Input projection into [x, z] where z is SiLU gate
        self.W_in: np.ndarray = rng.randn(d_model, 2 * self.d_inner).astype(
            np.float32
        ) * np.sqrt(2.0 / d_model)

        # 1D Depthwise Convolution weights & bias
        self.conv1d_weight: np.ndarray = (
            rng.randn(self.d_inner, d_conv).astype(np.float32) * 0.1
        )
        self.conv1d_bias: np.ndarray = np.zeros(self.d_inner, dtype=np.float32)

        # Input-dependent projections for Delta, B, C
        self.W_delta: np.ndarray = (
            rng.randn(self.d_inner, self.d_inner).astype(np.float32) * 0.01
        )
        self.b_delta: np.ndarray = np.zeros(self.d_inner, dtype=np.float32)
        self.W_B: np.ndarray = (
            rng.randn(self.d_inner, d_state).astype(np.float32) * 0.01
        )
        self.W_C: np.ndarray = (
            rng.randn(self.d_inner, d_state).astype(np.float32) * 0.01
        )

        # Continuous state matrix A (HiPPO initialized diagonal)
        self.A_log: np.ndarray = np.tile(
            np.log(np.arange(1, d_state + 1, dtype=np.float32)), (self.d_inner, 1)
        )
        # Skip connection D
        self.D: np.ndarray = np.ones(self.d_inner, dtype=np.float32)

        # Output projection
        self.W_out: np.ndarray = rng.randn(self.d_inner, d_model).astype(
            np.float32
        ) * np.sqrt(2.0 / self.d_inner)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        u: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, _ = u.shape

        # Linear projection into x and z
        xz = np.dot(u, self.W_in)  # [B, T, 2 * d_inner]
        x_proj = xz[:, :, : self.d_inner]  # [B, T, d_inner]
        z_proj = xz[:, :, self.d_inner :]  # [B, T, d_inner]

        # 1D Depthwise causal convolution with padding
        x_conv = np.zeros_like(x_proj)
        for b in range(B):
            for ch in range(self.d_inner):
                sig = x_proj[b, :, ch]
                padded = np.pad(sig, (self.d_conv - 1, 0), mode="constant")
                conv_out = np.convolve(
                    padded, self.conv1d_weight[ch][::-1], mode="valid"
                )
                x_conv[b, :, ch] = conv_out[:T] + self.conv1d_bias[ch]

        # SiLU activation: x = x * sigmoid(x)
        x = x_conv / (1.0 + np.exp(-np.clip(x_conv, -20.0, 20.0)))

        # Selective parameter calculation: Delta, B, C
        delta_raw = np.dot(x, self.W_delta) + self.b_delta  # [B, T, d_inner]
        Delta = np.log1p(np.exp(np.clip(delta_raw, -20.0, 20.0)))  # Softplus(Delta)

        B_mat = np.dot(x, self.W_B)  # [B, T, d_state]
        C_mat = np.dot(x, self.W_C)  # [B, T, d_state]

        A = -np.exp(self.A_log)  # [d_inner, d_state]

        # Selective Scan: h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C_t * h_t + D * x_t
        y = np.zeros((B, T, self.d_inner), dtype=np.float32)

        for b in range(B):
            h_state: np.ndarray = np.zeros(
                (self.d_inner, self.d_state), dtype=np.float32
            )
            for t in range(T):
                dt = Delta[b, t].reshape(self.d_inner, 1)  # [d_inner, 1]
                b_t = B_mat[b, t].reshape(1, self.d_state)  # [1, d_state]
                c_t = C_mat[b, t].reshape(self.d_state, 1)  # [d_state, 1]
                x_t = x[b, t].reshape(self.d_inner, 1)  # [d_inner, 1]

                # Discretization: A_bar = exp(Delta * A), B_bar = Delta * B
                A_bar = np.exp(dt * A)  # [d_inner, d_state]
                B_bar = dt * b_t  # [d_inner, d_state]

                h_state = A_bar * h_state + B_bar * x_t
                y_t = np.dot(h_state, c_t).ravel() + self.D * x[b, t]
                y[b, t] = y_t

        # Gating with SiLU(z)
        z_gate = z_proj / (1.0 + np.exp(-np.clip(z_proj, -20.0, 20.0)))
        y_gated = y * z_gate

        return np.dot(y_gated, self.W_out)


class Mamba:
    """
    Sovereign Mamba Language Model Architecture.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    d_model : int, default=64
        Model embedding dimension.
    num_layers : int, default=2
        Number of Mamba SSM blocks.
    d_state : int, default=16
        State space dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        rng = np.random.RandomState(seed)
        self.embed: np.ndarray = (
            rng.randn(vocab_size, d_model).astype(np.float32) * 0.02
        )
        self.layers = [
            MambaSSM(d_model, d_state=d_state, seed=seed + i) for i in range(num_layers)
        ]
        self.lm_head: np.ndarray = (
            rng.randn(d_model, vocab_size).astype(np.float32) * 0.02
        )

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        x = self.embed[input_ids]
        for layer in self.layers:
            x = x + layer.forward(x)
            # RMSNorm
            x = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return np.dot(x, self.lm_head)
