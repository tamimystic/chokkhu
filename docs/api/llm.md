# Frontier LLMs, Linear Attention & Alignment API Reference

The `chokkhu.models.nlp` package implements state-of-the-art Large Language Model architectures with mathematical precision from first principles.

---

## 1. Supported Sovereign Architectures

| Architecture | Class | Key Architectural Innovations |
| :--- | :--- | :--- |
| **LLaMA** | `chokkhu.models.nlp.architectures.llama.LLaMA` | SwiGLU, RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), $O(1)$ Step KV-Cache |
| **Mistral** | `chokkhu.models.nlp.architectures.mistral.Mistral` | Sliding Window Attention (SWA), Rolling Buffer KV-Cache, GQA |
| **Qwen 2.5** | `chokkhu.models.nlp.architectures.qwen.Qwen` | Dual RMSNorm, Enhanced RoPE Base Frequency, Deep MLP Projections |
| **DeepSeek-V3** | `chokkhu.models.nlp.architectures.deepseek_v3.DeepSeekV3` | Multi-Head Latent Attention (MLA), Top-2 Gated MoE, Multi-Token Prediction (MTP) |
| **RWKV-6** | `chokkhu.models.nlp.architectures.rwkv6.RWKV6` | Linear Time Attention with State Decay ($O(1)$ memory inference) |
| **Mamba** | `chokkhu.models.nlp.architectures.mamba.Mamba` | Selective State-Space Model (S6) with time-varying discretization |
| **BitNet 1.58b** | `chokkhu.models.nlp.architectures.bitnet158.BitNet158` | $\{-1, 0, +1\}$ Ternary Quantized Weight Matrix Multiplication |

---

## 2. Rotary Position Embeddings (RoPE)

RoPE encodes relative token distances by applying a complex coordinate rotation to query and key vectors:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$

$$\text{where } R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d}$$

In Chokkhu, `RotaryPositionEmbedding.apply_rope` supports positional offsets (`offset=k`), enabling seamless continuous generation with dynamic KV-caching.

---

## 3. Dynamic $O(1)$ KV-Cache Generation

```python
import numpy as np
from chokkhu.models.nlp.architectures.llama import LLaMA

# 1. Initialize sovereign LLaMA model
model = LLaMA(
    vocab_size=32000,
    dim=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=2,  # 4x GQA KV compression
    max_seq_len=512
)

# 2. Input prompt
prompt_tokens = np.array([[101, 2054, 2003, 1037]])

# 3. Autoregressive generation with KV-caching
output_tokens = model.generate(
    prompt_tokens,
    max_new_tokens=20,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15
)
```

### Sampling Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_new_tokens` | `int` | `20` | Maximum number of new autoregressive tokens to generate. |
| `temperature` | `float` | `1.0` | Logit scaling parameter. Set to `0.0` for deterministic greedy decoding. |
| `top_k` | `int` / `None` | `None` | Restricts sampling pool to the $K$ highest-probability candidate tokens. |
| `top_p` | `float` / `None` | `None` | Nucleus sampling: cumulative probability mass threshold (e.g. `0.9`). |
| `repetition_penalty` | `float` | `1.0` | Exponential penalty scaling factor applied to previously generated token logits. |
