"""Unit tests for Frontier LLM generation, O(1) KV-Caching, and Sampling Decoders."""

from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.models.nlp.architectures.llama import LLaMA
from chokkhu.models.nlp.attention import (
    GroupedQueryAttention,
    ScaledDotProductAttention,
)
from chokkhu.models.nlp.generation import GenerationConfig, sample_next_token


def test_gqa_kv_cache_equivalence():
    """Verify that incremental 1-token step with KV cache equals full sequence evaluation."""
    np.random.seed(42)
    gqa = GroupedQueryAttention(
        embed_dim=16,
        num_query_heads=4,
        num_kv_heads=2,
        bias=False,
        use_rope=True,
    )

    # 1. Full sequence forward pass (3 tokens)
    x_full = Tensor(np.random.randn(1, 3, 16).astype(np.float64))
    out_full = gqa(x_full, is_causal=True)
    assert isinstance(out_full, Tensor)
    last_out_full = out_full.data[0, -1, :]

    # 2. Incremental forward pass with KV cache
    # Step 1: prompt (first 2 tokens)
    x_prompt = Tensor(x_full.data[:, :2, :])
    out_prompt_res = gqa(x_prompt, is_causal=True, use_cache=True)
    assert isinstance(out_prompt_res, tuple)
    _, kv_cache = out_prompt_res

    # Step 2: 3rd token with kv_cache
    x_step = Tensor(x_full.data[:, 2:3, :])
    out_step_res = gqa(x_step, is_causal=True, kv_cache=kv_cache, use_cache=True)
    assert isinstance(out_step_res, tuple)
    out_step, _ = out_step_res

    last_out_cached = out_step.data[0, 0, :]

    np.testing.assert_allclose(
        last_out_full,
        last_out_cached,
        rtol=1e-4,
        atol=1e-5,
        err_msg="KV cached output does not match full sequence evaluation",
    )


def test_llama_greedy_generation_cache_vs_non_cache():
    """Verify greedy autoregressive generation matches between use_cache=True and use_cache=False."""
    np.random.seed(42)
    model = LLaMA(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        hidden_dim=64,
        max_seq_len=64,
    )

    prompt = [10, 25, 42]

    # Generate with KV cache
    gen_cached = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        eos_token_id=None,
        use_cache=True,
    )

    # Generate without KV cache
    gen_uncached = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        eos_token_id=None,
        use_cache=False,
    )

    np.testing.assert_array_equal(
        gen_cached,
        gen_uncached,
        err_msg="Cached generation does not match non-cached generation",
    )
    assert gen_cached.shape == (1, 8)  # 3 prompt + 5 new tokens


def test_sampling_decoders_top_k_top_p():
    logits = np.array([-10.0, 5.0, 4.0, 1.0, -5.0, 0.5])
    generated = [1]

    # Test temperature=0.0 (greedy)
    config_greedy = GenerationConfig(temperature=0.0, do_sample=False)
    tok_greedy = sample_next_token(logits, generated, config_greedy)
    assert tok_greedy == 1  # Index of 5.0

    # Test top_k = 1
    config_k1 = GenerationConfig(temperature=1.0, top_k=1, do_sample=True)
    tok_k1 = sample_next_token(logits, generated, config_k1)
    assert tok_k1 == 1

    # Test repetition penalty
    config_rep = GenerationConfig(
        temperature=0.0, repetition_penalty=2.0, do_sample=False
    )
    # Token 1 is already in generated, penalized from 5.0 to 2.5, so token 2 (4.0) wins
    tok_rep = sample_next_token(logits, generated, config_rep)
    assert tok_rep == 2


def test_causal_attention_masking_no_future_leak():
    """Verify that causal attention scores strictly prevent future token leakage."""
    sdpa = ScaledDotProductAttention()
    q = Tensor(np.random.randn(1, 1, 4, 8))
    k = Tensor(np.random.randn(1, 1, 4, 8))
    v = Tensor(np.random.randn(1, 1, 4, 8))

    causal_mask = np.tril(np.ones((4, 4), dtype=bool))[np.newaxis, np.newaxis, :, :]
    _, attn_weights = sdpa(q, k, v, mask=causal_mask)

    weights_data = attn_weights.data[0, 0]  # (4, 4)

    # Upper triangular elements above diagonal must be strictly 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            assert (
                weights_data[i, j] == 0.0
            ), f"Future token leak detected at ({i}, {j})"
