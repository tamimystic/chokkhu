"""Mixture of Experts (MoE) Subsystem for Sovereign Large Language Models (Shazeer et al., Switch Transformers, Mixtral 8x7B, DeepSeekMoE)."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Linear, Module
from ..dl.activations import GELU
from .transformer_blocks import SwiGLU


class Expert(Module):
    """Individual Expert FFN Block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = "swiglu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation.lower()
        if self.activation_type == "swiglu":
            self.net = SwiGLU(dim, hidden_dim, bias=bias)
        else:
            self.w1 = Linear(dim, hidden_dim, bias=bias)
            self.w2 = Linear(hidden_dim, dim, bias=bias)
            self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_type == "swiglu":
            return self.net(x)
        else:
            N, S, D = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])
            x_flat = x.reshape(-1, D)
            out = self.w2(self.act(self.w1(x_flat)))
            return out.reshape(x.shape)


class TopKRouter(Module):
    """Top-K Router with Load Balancing Auxiliary Loss (Shazeer et al., 2017; Fedus et al., 2021)."""

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        noisy_gating: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef
        self.noisy_gating = noisy_gating
        self.gate = Linear(dim, num_experts, bias=False)
        if noisy_gating:
            self.w_noise = Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[np.ndarray, np.ndarray, Tensor]:
        """Compute routing weights and auxiliary load balancing loss.

        Args:
            x: Tensor of shape (batch_size, seq_len, dim) or (total_tokens, dim)
        Returns:
            (topk_indices, topk_weights, aux_loss_tensor)
        """
        N, S, D = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])
        total_tokens = N * S
        x_flat = x.reshape(total_tokens, D)

        logits = self.gate(x_flat).data  # (total_tokens, num_experts)
        if self.noisy_gating:
            noise_std = np.log1p(np.exp(self.w_noise(x_flat).data))  # softplus
            noise = np.random.randn(*logits.shape) * noise_std
            logits = logits + noise

        # Softmax over all experts to get dense probabilities P
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        # Top-k selection
        topk_indices = np.argsort(logits, axis=-1)[:, -self.top_k :]

        # Gather top-k probabilities and renormalize so they sum to 1 per token
        row_indices = np.arange(total_tokens)[:, None]
        topk_raw_probs = probs[row_indices, topk_indices]
        topk_weights = topk_raw_probs / (
            np.sum(topk_raw_probs, axis=-1, keepdims=True) + 1e-12
        )

        # Auxiliary load balancing loss:
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability allocated to expert i
        dispatch_mask: np.ndarray = np.zeros(
            (total_tokens, self.num_experts), dtype=np.float64
        )
        for k_idx in range(self.top_k):
            np.add.at(
                dispatch_mask, (np.arange(total_tokens), topk_indices[:, k_idx]), 1.0
            )

        f = np.mean(dispatch_mask, axis=0)
        P = np.mean(probs, axis=0)
        aux_loss_val = float(self.aux_loss_coef * self.num_experts * np.sum(f * P))
        aux_loss = Tensor(aux_loss_val, requires_grad=False)

        return topk_indices, topk_weights, aux_loss


class MixtureOfExperts(Module):
    """Sovereign Mixture of Experts (MoE) Layer (Shazeer et al., Mixtral 8x7B, DeepSeekMoE)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "swiglu",
        aux_loss_coef: float = 0.01,
        shared_experts: int = 0,
        shared_expert_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )
        self.experts: List[Expert] = [
            Expert(dim=dim, hidden_dim=hidden_dim, activation=activation)
            for _ in range(num_experts)
        ]
        for i, expert in enumerate(self.experts):
            setattr(self, f"expert_{i}", expert)

        self.shared_experts = shared_experts
        if shared_experts > 0:
            sh_dim = shared_expert_dim if shared_expert_dim is not None else hidden_dim
            self.shared_expert_layers: List[Expert] = [
                Expert(dim=dim, hidden_dim=sh_dim, activation=activation)
                for _ in range(shared_experts)
            ]
            for i, sh_exp in enumerate(self.shared_expert_layers):
                setattr(self, f"shared_expert_{i}", sh_exp)
        else:
            self.shared_expert_layers = []

        self.last_aux_loss: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass dispatching tokens to top-k routed experts and shared experts."""
        orig_shape = x.shape
        N, S, D = orig_shape if x.ndim == 3 else (1, orig_shape[0], orig_shape[1])
        total_tokens = N * S
        x_flat = x.reshape(total_tokens, D)

        topk_indices, topk_weights, aux_loss = self.router(x_flat)
        self.last_aux_loss = aux_loss

        out_data: np.ndarray = np.zeros((total_tokens, D), dtype=np.float64)

        for expert_id, expert in enumerate(self.experts):
            matches = topk_indices == expert_id
            token_mask = np.any(matches, axis=-1)
            if not np.any(token_mask):
                continue

            token_indices = np.where(token_mask)[0]
            expert_in = Tensor(
                x_flat.data[token_indices], requires_grad=x.requires_grad
            )
            expert_out = expert(expert_in).data

            for k_idx in range(self.top_k):
                slot_mask = topk_indices[token_indices, k_idx] == expert_id
                if np.any(slot_mask):
                    slot_token_indices = token_indices[slot_mask]
                    w = topk_weights[slot_token_indices, k_idx][:, None]
                    out_data[slot_token_indices] += expert_out[slot_mask] * w

        if self.shared_experts > 0:
            for sh_exp in self.shared_expert_layers:
                sh_out = sh_exp(x_flat).data
                out_data += sh_out

        out_tensor = Tensor(out_data.reshape(orig_shape), requires_grad=x.requires_grad)
        return out_tensor


MoE = MixtureOfExperts
