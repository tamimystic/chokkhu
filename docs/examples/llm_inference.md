# Frontier LLM SafeTensors Weight Loading & Inference

In this tutorial, you will construct a sovereign LLaMA architecture from scratch, serialize and deserialize weights in HuggingFace-compatible SafeTensors format with BF16 and INT8 quantization, and run autoregressive generation with dynamic $O(1)$ KV-caching.

---

## 1. Instantiate the Sovereign LLaMA Model

```python
import numpy as np
import chokkhu as ck
from chokkhu.models.nlp.architectures.llama import LLaMA
from chokkhu.io.safetensors import save_safetensors, load_safetensors, save_quantized_safetensors, load_quantized_safetensors

# 1. Define architecture parameters
vocab_size = 32000
dim = 256
n_layers = 4
n_heads = 8
n_kv_heads = 2  # Grouped Query Attention

model = LLaMA(
    vocab_size=vocab_size,
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    max_seq_len=512
)
print("Sovereign LLaMA model initialized successfully!")
```

---

## 2. Weight Serialization with SafeTensors & INT8 Quantization

```python
# 1. Extract model state dictionary
weights = model.state_dict()

# 2. Save weights in standard SafeTensors format
save_safetensors(weights, "llama_weights.safetensors")

# 3. Save compressed INT8 quantized weights (4x reduction)
save_quantized_safetensors(weights, "llama_weights_int8.safetensors")

# 4. Load weights back into a clean model instance
loaded_weights = load_quantized_safetensors("llama_weights_int8.safetensors")
model.load_state_dict(loaded_weights)
print("Model loaded from INT8 SafeTensors!")
```

---

## 3. Autoregressive Generation with KV-Cache

```python
# Prompt token IDs
prompt = np.array([[101, 1024, 7592, 2003]])

# Generate response tokens
output = model.generate(
    prompt,
    max_new_tokens=25,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

print(f"Generated Sequence: {output.tolist()}")
```
