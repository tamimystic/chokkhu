from __future__ import annotations

import numpy as np


class _RMSNorm:
    """Pure NumPy Root Mean Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight: np.ndarray = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class _SwiGLU:
    """Pure NumPy SwiGLU Gated Feedforward Layer."""

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


class DeepSeekMoE:
    """
    DeepSeek Fine-Grained Mixture of Experts (MoE).
    Features dedicated Shared Experts (always executed) and fine-grained Routed Experts
    with softmax gating.

    Parameters
    ----------
    hidden_dim : int
        Input/output representation dimension.
    intermediate_dim : int
        Dimension of expert MLP.
    n_routed_experts : int, default=8
        Total number of routed experts.
    n_shared_experts : int, default=1
        Number of shared experts.
    top_k : int, default=2
        Number of routed experts selected per token.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        top_k: int = 2,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k

        rng = np.random.RandomState(seed)

        # Router Gate for routed experts
        self.router_gate: np.ndarray = (
            rng.randn(hidden_dim, n_routed_experts).astype(np.float32) * 0.02
        )

        # Shared Experts
        self.shared_experts = [
            _SwiGLU(hidden_dim, intermediate_dim, seed=seed + i)
            for i in range(n_shared_experts)
        ]

        # Routed Experts
        self.routed_experts = [
            _SwiGLU(hidden_dim, intermediate_dim, seed=seed + 100 + i)
            for i in range(n_routed_experts)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        # 1. Shared Experts
        shared_out = np.zeros_like(x_flat)
        for exp in self.shared_experts:
            shared_out += exp.forward(x_flat)

        # 2. Router Logits & Top-K Routing
        router_logits = np.dot(x_flat, self.router_gate)  # [N, n_routed]
        exp_logits = np.exp(
            router_logits - np.max(router_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        routed_out = np.zeros_like(x_flat)

        for i in range(N):
            top_indices = np.argsort(probs[i])[-self.top_k :]
            top_weights = probs[i, top_indices]
            top_weights /= np.sum(top_weights) + 1e-12

            for weight, exp_idx in zip(top_weights, top_indices):
                exp_res = self.routed_experts[exp_idx].forward(x_flat[i : i + 1])
                routed_out[i] += weight * exp_res[0]

        total_out = shared_out + routed_out
        return total_out.reshape(B, T, D)


class _DeepSeekMLA:
    """Pure NumPy Multi-Head Latent Attention (MLA)."""

    def __init__(
        self, dim: int, num_heads: int = 4, kv_latent_dim: int = 16, seed: int = 42
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_latent_dim = kv_latent_dim

        rng = np.random.RandomState(seed)
        self.W_dkv: np.ndarray = rng.randn(dim, kv_latent_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / dim)
        self.W_uk: np.ndarray = rng.randn(kv_latent_dim, dim).astype(
            np.float32
        ) * np.sqrt(2.0 / kv_latent_dim)
        self.W_uv: np.ndarray = rng.randn(kv_latent_dim, dim).astype(
            np.float32
        ) * np.sqrt(2.0 / kv_latent_dim)
        self.W_q: np.ndarray = rng.randn(dim, dim).astype(np.float32) * np.sqrt(
            2.0 / dim
        )
        self.W_o: np.ndarray = rng.randn(dim, dim).astype(np.float32) * np.sqrt(
            2.0 / dim
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Low-rank KV compression
        c_kv = np.dot(x, self.W_dkv)  # [B, T, kv_latent_dim]
        K = np.dot(c_kv, self.W_uk).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = np.dot(c_kv, self.W_uv).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        Q = np.dot(x, self.W_q).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d)

        # Causal mask
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        scores += mask

        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T, D)
        return np.dot(out, self.W_o)


class DeepSeekV3:
    """
    DeepSeek-V3 / DeepSeek-R1 Architecture with DeepSeekMLA, DeepSeekMoE,
    RMSNorm, and Multi-Token Prediction (MTP).

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_dim : int, default=64
        Model dimension.
    num_layers : int, default=2
        Number of Transformer layers.
    n_routed_experts : int, default=4
        Number of routed MoE experts.
    top_k : int, default=2
        Top-k routing capacity.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_routed_experts: int = 4,
        top_k: int = 2,
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
            _DeepSeekMLA(dim=hidden_dim, num_heads=4, kv_latent_dim=16, seed=seed + i)
            for i in range(num_layers)
        ]
        self.moe_layers = [
            DeepSeekMoE(
                hidden_dim=hidden_dim,
                intermediate_dim=hidden_dim * 2,
                n_routed_experts=n_routed_experts,
                top_k=top_k,
                seed=seed + i,
            )
            for i in range(num_layers)
        ]

        self.lm_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )
        self.mtp_head: np.ndarray = (
            rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02
        )

    def forward(
        self, input_ids: np.ndarray, return_mtp: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        x = self.embed[input_ids]  # [B, T, H]

        for attn, moe in zip(self.attn_layers, self.moe_layers):
            norm_x = self.norm.forward(x)
            attn_out = attn.forward(norm_x)
            x = x + attn_out

            norm_x2 = self.norm.forward(x)
            moe_out = moe.forward(norm_x2)
            x = x + moe_out

        final_norm = self.norm.forward(x)
        logits = np.dot(final_norm, self.lm_head)

        if return_mtp:
            mtp_logits = np.dot(final_norm, self.mtp_head)
            return logits, mtp_logits

        return logits
