# SafeTensors & Model Quantization API Reference

The `chokkhu.io.safetensors` module provides a zero-dependency binary serialization parser for HuggingFace SafeTensors files, with native support for `FLOAT32`, `FLOAT16`, `BFLOAT16`, `INT64`, `INT32`, and symmetric `INT8` quantized weights.

---

## 1. Zero-Dependency SafeTensors Reading & Writing

```python
import numpy as np
from chokkhu.io.safetensors import save_safetensors, load_safetensors

tensors = {
    "embedding.weight": np.random.randn(1000, 128).astype(np.float32),
    "layer1.weight": np.random.randn(128, 128).astype(np.float32),
    "layer1.bias": np.zeros(128, dtype=np.float32)
}

# Save in HuggingFace-compatible SafeTensors format
save_safetensors(tensors, "model.safetensors", metadata={"format": "chokkhu"})

# Load directly back into NumPy ndarrays
loaded_tensors = load_safetensors("model.safetensors")
```

---

## 2. Pure NumPy BFloat16 Bit-Shift Decoding

Because standard NumPy lacks a native `bfloat16` data type, Chokkhu uses bitwise uint16 bit-shifting to losslessly convert BF16 weights directly into IEEE 754 single-precision `float32`:

$$\text{float32\_bits} = \text{uint16\_bits} \ll 16$$

```python
# Decoding BFLOAT16 bytes without C-extensions:
bf16_u16 = np.frombuffer(raw_bytes, dtype="<u2")
f32_u32 = bf16_u16.astype(np.uint32) << 16
f32_arr = f32_u32.view(np.float32)
```

---

## 3. Symmetric INT8 Weight Quantization

Symmetric linear quantization maps floating-point weights $W \in \mathbb{R}$ to signed 8-bit integers $W_q \in [-127, 127]$:

$$S = \frac{\max(|W|)}{127}, \quad W_q = \text{round}\left( \text{clip}\left( \frac{W}{S}, -127, 127 \right) \right)$$

$$\hat{W} = W_q \cdot S$$

```python
from chokkhu.io.safetensors import save_quantized_safetensors, load_quantized_safetensors

# Save weights compressed into 8-bit integers with per-tensor scale factor
save_quantized_safetensors(tensors, "model_int8.safetensors")

# Load and automatically dequantize back to float32
dequantized = load_quantized_safetensors("model_int8.safetensors")
```

---

## 4. Universal `state_dict` & `load_state_dict`

Every `chokkhu.models.dl.layers.Module` subclass supports standard weight dictionary exporting and loading:

```python
# Export dictionary of trainable Tensor parameters
weights = model.state_dict()

# Load weights into a new model instance
new_model.load_state_dict(weights)
```
