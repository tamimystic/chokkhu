"""Unit tests for Multi-DType SafeTensors, BF16 encoding/decoding, INT8 quantization, and Model weight loading."""

from __future__ import annotations

import os
import tempfile
import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.io.safetensors import (
    load_quantized_safetensors,
    load_safetensors,
    save_quantized_safetensors,
    save_safetensors,
)
from chokkhu.models.dl.layers import Linear
from chokkhu.models.nlp.architectures.llama import LLaMA


def test_safetensors_multidtype_roundtrip():
    """Verify roundtrip serialization across all supported dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "multi_dtype.safetensors")

        tensors = {
            "f64": np.array([1.5, -2.5, 3.5], dtype=np.float64),
            "f32": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "f16": np.array([1.0, 2.0, 3.0], dtype=np.float16),
            "i64": np.array([1000000, 2000000], dtype=np.int64),
            "i32": np.array([10, 20, 30], dtype=np.int32),
            "i16": np.array([1, 2, 3], dtype=np.int16),
            "i8": np.array([-5, 5], dtype=np.int8),
            "u8": np.array([250, 255], dtype=np.uint8),
            "b": np.array([True, False, True], dtype=bool),
        }

        save_safetensors(tensors, filepath, metadata={"creator": "chokkhu"})
        loaded = load_safetensors(filepath)

        for k, v in tensors.items():
            assert k in loaded
            np.testing.assert_allclose(
                loaded[k].data, v.astype(loaded[k].data.dtype), rtol=1e-3, atol=1e-3
            )


def test_safetensors_bf16_roundtrip():
    """Verify BFloat16 bit-level encoding and decoding without external dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model_bf16.safetensors")

        orig_f32 = np.array(
            [[1.0, -2.0, 0.5], [100.0, -0.001, 3.1415]], dtype=np.float32
        )
        save_safetensors({"weights": orig_f32}, filepath, bf16=True)

        loaded = load_safetensors(filepath)
        assert "weights" in loaded
        # BF16 has 7 bits of mantissa (~3 decimal digits precision)
        np.testing.assert_allclose(
            loaded["weights"].data, orig_f32, rtol=1e-2, atol=1e-2
        )


def test_safetensors_int8_quantization():
    """Verify symmetric INT8 weight quantization and dequantization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "quantized.safetensors")

        weights = np.random.randn(20, 20).astype(np.float32)
        save_quantized_safetensors({"layer.weight": weights}, filepath)

        dequantized = load_quantized_safetensors(filepath)
        assert "layer.weight" in dequantized

        # Relative error of 8-bit quantization is within ~1%
        np.testing.assert_allclose(
            dequantized["layer.weight"].data, weights, rtol=0.02, atol=0.05
        )


def test_module_state_dict_and_load_weights():
    """Verify state_dict serialization and load_weights on Neural Network modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "linear.safetensors")

        layer1 = Linear(in_features=4, out_features=2, bias=True)
        layer1.weight.data[...] = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64
        )
        layer1.bias.data[...] = np.array([[0.5, -0.5]], dtype=np.float64)

        state = layer1.state_dict()
        assert "weight" in state and "bias" in state
        save_safetensors(state, filepath)

        layer2 = Linear(in_features=4, out_features=2, bias=True)
        layer2.load_weights(filepath)

        np.testing.assert_allclose(layer2.weight.data, layer1.weight.data)
        np.testing.assert_allclose(layer2.bias.data, layer1.bias.data)


def test_llama_weights_binding_roundtrip():
    """Verify weight saving and loading on full frontier LLaMA model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "llama_test.safetensors")

        model1 = LLaMA(
            vocab_size=50,
            embed_dim=16,
            num_layers=1,
            num_query_heads=2,
            num_kv_heads=1,
            hidden_dim=32,
        )
        state1 = model1.state_dict()
        save_safetensors(state1, filepath)

        model2 = LLaMA(
            vocab_size=50,
            embed_dim=16,
            num_layers=1,
            num_query_heads=2,
            num_kv_heads=1,
            hidden_dim=32,
        )
        model2.load_weights(filepath)

        tokens = Tensor([[1, 5, 10]])
        out1 = model1(tokens)
        out2 = model2(tokens)

        np.testing.assert_allclose(out1.data, out2.data)
