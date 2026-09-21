"""Unit tests for SafeTensors bridge and RoPE positional integration in LLMs."""

import os
import tempfile
import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.io.safetensors import load_safetensors, save_safetensors
from chokkhu.models.nlp.architectures.llama import LLaMA


def test_safetensors_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.safetensors")

        tensors = {
            "weight1": Tensor(np.random.randn(10, 20).astype(np.float32)),
            "bias1": Tensor(np.random.randn(20).astype(np.float64)),
            "labels": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        }
        metadata = {"format": "chokkhu", "version": "1.6.0"}

        save_safetensors(tensors, filepath, metadata=metadata)
        assert os.path.exists(filepath)

        loaded = load_safetensors(filepath)
        assert "weight1" in loaded
        assert "bias1" in loaded
        assert "labels" in loaded

        np.testing.assert_allclose(
            loaded["weight1"].data, tensors["weight1"].data, rtol=1e-5
        )
        np.testing.assert_allclose(
            loaded["bias1"].data, tensors["bias1"].data, rtol=1e-5
        )
        np.testing.assert_array_equal(loaded["labels"].data, tensors["labels"])


def test_llama_forward_backward_with_rope():
    model = LLaMA(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        hidden_dim=64,
        max_seq_len=64,
    )
    input_ids = np.array([[1, 5, 20, 42, 99], [2, 10, 30, 40, 50]], dtype=np.int64)
    logits = model(input_ids)
    assert logits.shape == (2, 5, 100)

    loss = logits.sum()
    loss.backward()
    # Verify gradient reached token embeddings and projections
    assert model.token_embed.weight.grad is not None
    assert model.token_embed.weight.grad.shape == (100, 32)
