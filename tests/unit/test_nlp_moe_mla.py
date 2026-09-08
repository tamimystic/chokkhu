"""Unit tests for Mixture of Experts (MoE), Multi-Head Latent Attention (DeepSeek MLA), and Sliding Window Attention (SWA)."""

from __future__ import annotations
import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.nlp import (
    Expert,
    TopKRouter,
    MixtureOfExperts,
    MoE,
    MultiHeadLatentAttention,
    DeepSeekMLA,
    SlidingWindowAttention,
)


def test_expert_ffn_and_swiglu() -> None:
    x = Tensor(np.random.randn(2, 6, 32), requires_grad=True)

    expert_swiglu = Expert(dim=32, hidden_dim=64, activation="swiglu")
    out_swiglu = expert_swiglu(x)
    assert out_swiglu.shape == (2, 6, 32)

    expert_gelu = Expert(dim=32, hidden_dim=64, activation="gelu")
    out_gelu = expert_gelu(x)
    assert out_gelu.shape == (2, 6, 32)


def test_topk_router_and_load_balance() -> None:
    x = Tensor(np.random.randn(2, 6, 32), requires_grad=False)
    router = TopKRouter(
        dim=32,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.05,
        noisy_gating=True,
    )

    indices, weights, aux_loss = router(x)
    assert indices.shape == (12, 2)
    assert weights.shape == (12, 2)
    assert np.all(indices >= 0) and np.all(indices < 4)
    # Weights should sum to 1 per token
    np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones(12), atol=1e-5)
    assert aux_loss.data > 0.0


def test_mixture_of_experts_forward() -> None:
    x = Tensor(np.random.randn(3, 8, 32), requires_grad=True)

    # Standard MoE
    moe = MixtureOfExperts(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        activation="swiglu",
    )
    out = moe(x)
    assert out.shape == (3, 8, 32)
    assert moe.last_aux_loss is not None
    assert moe.last_aux_loss.data >= 0.0

    # DeepSeekMoE style with shared experts
    moe_shared = MoE(
        dim=32,
        hidden_dim=48,
        num_experts=4,
        top_k=1,
        shared_experts=2,
        shared_expert_dim=32,
    )
    out_shared = moe_shared(x)
    assert out_shared.shape == (3, 8, 32)


def test_deepseek_mla() -> None:
    # Test uncompressed query MLA
    mla = MultiHeadLatentAttention(
        embed_dim=64,
        num_heads=4,
        head_dim=16,
        kv_latent_dim=32,
        rope_dim=8,
        max_seq_len=128,
    )
    x = Tensor(np.random.randn(2, 10, 64), requires_grad=True)
    out = mla(x, is_causal=True)
    assert out.shape == (2, 10, 64)

    # Test query compressed MLA
    mla_qc = DeepSeekMLA(
        embed_dim=64,
        num_heads=4,
        head_dim=16,
        kv_latent_dim=32,
        q_latent_dim=32,
        rope_dim=8,
        max_seq_len=128,
    )
    out_qc = mla_qc(x, is_causal=True)
    assert out_qc.shape == (2, 10, 64)


def test_sliding_window_attention() -> None:
    # SWA with MHA
    swa = SlidingWindowAttention(
        embed_dim=64,
        num_heads=4,
        window_size=4,
    )
    x = Tensor(np.random.randn(2, 12, 64), requires_grad=True)
    out = swa(x)
    assert out.shape == (2, 12, 64)

    # SWA with GQA (grouped query)
    swa_gqa = SlidingWindowAttention(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        window_size=3,
    )
    out_gqa = swa_gqa(x)
    assert out_gqa.shape == (2, 12, 64)
