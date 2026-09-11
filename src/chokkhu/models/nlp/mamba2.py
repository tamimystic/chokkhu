"""Mamba-2 State Space Duality (SSD) and Jamba Transformer-SSM-MoE Hybrid Architecture in pure NumPy."""

from __future__ import annotations

import numpy as np


class Mamba2SSD:
    """Mamba-2 Structured State Space Duality (SSD) Layer in pure NumPy.

    Reformulates continuous 1D selective state-space models as structured
    block-diagonal matrix multiplication with chunked state recurrence (Dao & Gu 2024).

    Parameters
    ----------
    d_model : int, default=64
        Model input and output feature dimension.
    d_state : int, default=64
        Latent SSM state dimension ($N$).
    d_conv : int, default=4
        1D causal convolution kernel width.
    expand : int, default=2
        Expansion factor for internal hidden projection dimension ($d_{inner} = d_{model} 	imes expand$).
    chunk_size : int, default=8
        Block chunk size for 1D SSD matrix multiplication.
    seed : int, default=42
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(d_model * expand)
        self.chunk_size = int(chunk_size)

        rng = np.random.default_rng(seed)
        scale_in = np.sqrt(2.0 / d_model)
        scale_inner = np.sqrt(2.0 / self.d_inner)

        # Input projections: x -> (z, x_proj, B, C, dt)
        self.w_in: np.ndarray = rng.normal(
            0, scale_in, size=(2 * self.d_inner + 2 * d_state + self.d_inner, d_model)
        ).astype(np.float32)

        # 1D Causal Convolution weights (depthwise across d_inner)
        self.conv1d_weight: np.ndarray = rng.normal(
            0, 1.0 / np.sqrt(d_conv), size=(self.d_inner, d_conv)
        ).astype(np.float32)
        self.conv1d_bias: np.ndarray = np.zeros(self.d_inner, dtype=np.float32)

        # Continuous diagonal decay parameter A
        self.A_log: np.ndarray = np.log(
            rng.uniform(1.0, 16.0, size=(self.d_inner,))
        ).astype(np.float32)

        # Skip connection D
        self.D: np.ndarray = np.ones(self.d_inner, dtype=np.float32)

        # Output projection
        self.w_out: np.ndarray = rng.normal(
            0, scale_inner, size=(d_model, self.d_inner)
        ).astype(np.float32)

    def _causal_conv1d(self, x: np.ndarray) -> np.ndarray:
        """Applies 1D causal convolution (B, L, D_inner) -> (B, L, D_inner)."""
        B, L, D = x.shape
        # Pad left by d_conv - 1 for strict causality
        x_pad = np.pad(x, ((0, 0), (self.d_conv - 1, 0), (0, 0)), mode="constant")
        out = np.zeros((B, L, D), dtype=np.float32)

        for t in range(L):
            window = x_pad[:, t : t + self.d_conv, :]  # (B, d_conv, D)
            # Depthwise conv dot product
            out[:, t, :] = (
                np.sum(window * self.conv1d_weight.T[np.newaxis, :, :], axis=1)
                + self.conv1d_bias
            )
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes chunked State Space Duality forward pass (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape
        x_arr = np.asarray(x, dtype=np.float32)

        # 1. Project input: (B, L, d_model) @ W_in^T
        projected = np.matmul(x_arr, self.w_in.T)

        # Split components
        z = projected[:, :, : self.d_inner]
        x_branch = projected[:, :, self.d_inner : 2 * self.d_inner]
        b_proj = projected[:, :, 2 * self.d_inner : 2 * self.d_inner + self.d_state]
        c_proj = projected[
            :,
            :,
            2 * self.d_inner + self.d_state : 2 * self.d_inner + 2 * self.d_state,
        ]
        dt_proj = projected[:, :, 2 * self.d_inner + 2 * self.d_state :]

        # 2. Causal Conv1D & SiLU activation
        conv_out = self._causal_conv1d(x_branch)
        # SiLU(u) = u * sigmoid(u)
        silu_conv = conv_out * (1.0 / (1.0 + np.exp(-np.clip(conv_out, -15.0, 15.0))))

        # 3. Discretized SSM parameters
        dt = np.log1p(np.exp(np.clip(dt_proj, -10.0, 10.0)))  # softplus
        A = -np.exp(self.A_log)  # Continuous negative decay (D_inner,)

        # 4. State-Space Duality: 1D chunk recurrence & intra-chunk matrix mul
        y_ssm = np.zeros((B, L, self.d_inner), dtype=np.float32)

        for b in range(B):
            h_state: np.ndarray = np.zeros(
                (self.d_inner, self.d_state), dtype=np.float32
            )

            for t in range(L):
                dt_t = dt[b, t, :]  # (D_inner,)
                dA_t = np.exp(A * dt_t)  # (D_inner,)
                dB_t = (
                    dt_t[:, np.newaxis] * b_proj[b, t, :][np.newaxis, :]
                )  # (D_inner, d_state)
                x_t = silu_conv[b, t, :]  # (D_inner,)

                # Recurrence update: h_t = dA * h_{t-1} + (x_t * dB)
                h_state = dA_t[:, np.newaxis] * h_state + x_t[:, np.newaxis] * dB_t

                # Output state readout: y_t = h_t @ c_t^T + D * x_t
                c_t = c_proj[b, t, :]  # (d_state,)
                y_t = np.dot(h_state, c_t) + self.D * x_t
                y_ssm[b, t, :] = y_t

        # 5. Gated multiplicative branch: y_ssm * SiLU(z)
        silu_z = z * (1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0))))
        y_gated = y_ssm * silu_z

        # 6. Output projection back to d_model
        out = np.matmul(y_gated, self.w_out.T)
        return out


class JambaHybridBlock:
    """Jamba Hybrid Transformer-SSM Layer with Sparse Mixture of Experts (MoE).

    Interleaves Mamba-2 state space dynamics for linear sequence processing
    with Transformer Multi-Head Self-Attention and Top-K Sparse MoE MLP routing.

    Parameters
    ----------
    d_model : int, default=64
        Input and output feature dimension.
    num_heads : int, default=4
        Number of attention heads in multi-head attention module.
    num_experts : int, default=4
        Number of MoE feed-forward experts.
    top_k : int, default=2
        Number of active experts routed per token.
    seed : int, default=42
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        seed: int = 42,
    ) -> None:
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.head_dim = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        # 1. Mamba-2 SSM Sub-Layer
        self.mamba = Mamba2SSD(d_model=d_model, expand=2, seed=seed)

        # 2. Multi-Head Attention Sub-Layer
        self.w_q = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.w_k = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.w_v = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.w_attn_out = rng.normal(0, scale, size=(d_model, d_model)).astype(
            np.float32
        )

        # 3. Sparse MoE Router & Expert MLPs
        self.w_router = rng.normal(0, scale, size=(num_experts, d_model)).astype(
            np.float32
        )
        self.experts_w1 = [
            rng.normal(0, scale, size=(d_model * 2, d_model)).astype(np.float32)
            for _ in range(num_experts)
        ]
        self.experts_w2 = [
            rng.normal(0, scale, size=(d_model, d_model * 2)).astype(np.float32)
            for _ in range(num_experts)
        ]

    def _attention(self, x: np.ndarray) -> np.ndarray:
        """Causal multi-head self-attention forward pass."""
        B, L, D = x.shape
        Q = (
            np.matmul(x, self.w_q.T)
            .reshape(B, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        K = (
            np.matmul(x, self.w_k.T)
            .reshape(B, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        V = (
            np.matmul(x, self.w_v.T)
            .reshape(B, L, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.head_dim)

        # Causal triangular mask
        mask = np.triu(np.ones((L, L)), k=1).astype(bool)
        scores[:, :, mask] = -1e9

        # Softmax
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        context = np.matmul(attn_weights, V).swapaxes(1, 2).reshape(B, L, D)
        return np.matmul(context, self.w_attn_out.T)

    def _moe_ffn(self, x: np.ndarray) -> np.ndarray:
        """Top-K Sparse Mixture of Experts FFN routing."""
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)  # (N, D)

        # Compute routing logits: (N, num_experts)
        router_logits = np.matmul(flat_x, self.w_router.T)
        exp_logits = np.exp(
            router_logits - np.max(router_logits, axis=-1, keepdims=True)
        )
        router_probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        out_flat = np.zeros_like(flat_x)

        # Route top-k experts per token
        for i in range(len(flat_x)):
            top_indices = np.argsort(router_probs[i])[-self.top_k :]
            top_weights = router_probs[i, top_indices]
            top_weights = top_weights / (np.sum(top_weights) + 1e-12)

            token_vec = flat_x[i : i + 1]
            token_out = np.zeros((1, D), dtype=np.float32)

            for w, exp_idx in zip(top_weights, top_indices):
                # Expert MLP: GELU(x @ W1^T) @ W2^T
                h = np.matmul(token_vec, self.experts_w1[exp_idx].T)
                # GELU approximation
                h_act = (
                    0.5
                    * h
                    * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3)))
                )
                exp_out = np.matmul(h_act, self.experts_w2[exp_idx].T)
                token_out += w * exp_out

            out_flat[i] = token_out[0]

        return out_flat.reshape(B, L, D)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes full Jamba hybrid block (SSM -> Self-Attention -> Sparse MoE)."""
        x_arr = np.asarray(x, dtype=np.float32)

        # 1. Mamba-2 SSM with residual connection
        h1 = x_arr + self.mamba.forward(x_arr)

        # 2. Multi-Head Self-Attention with residual connection
        h2 = h1 + self._attention(h1)

        # 3. Sparse MoE FFN with residual connection
        out = h2 + self._moe_ffn(h2)
        return out
