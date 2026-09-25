# Model Compression & Edge AI API Reference

The `chokkhu.compression` module implements Post-Training Quantization (PTQ), Magnitude Pruning, and Knowledge Distillation to optimize models for resource-constrained edge devices.

---

## 1. Quantization Techniques

### Uniform & PTQ Quantization (`UniformQuantizer` / `PostTrainingQuantizer`)
Quantizes floating-point activations and weights into $b$-bit integer representations using Min-Max or Percentile calibration scales:

$$X_q = \text{round}\left( \frac{X - \text{zero\_point}}{\text{scale}} \right), \quad \hat{X} = X_q \cdot \text{scale} + \text{zero\_point}$$

```python
import numpy as np
from chokkhu.compression import PostTrainingQuantizer

weights = np.random.randn(256, 256).astype(np.float32)

ptq = PostTrainingQuantizer(bits=8, symmetric=True)
q_weights, scale, zero_point = ptq.quantize(weights)

print(f"Original size: {weights.nbytes} bytes, Quantized size: {q_weights.nbytes} bytes (4x reduction)")
```

---

## 2. Sparsity & Pruning (`GlobalMagnitudePruner`)

Prunes the lowest $|W|$ magnitude weights globally across all layers according to a target sparsity ratio $s \in (0, 1)$:

$$\text{Mask}_{i,j} = \mathbb{I}(|W_{i,j}| \ge \tau_s)$$

---

## 3. Knowledge Distillation (`KnowledgeDistiller`)

Trains a compact student model $S_\theta$ to match the softened probability distribution of a large teacher model $T_\phi$ with temperature scaling $\tau$:

$$\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(S(x), y) + (1 - \alpha) \tau^2 D_{\text{KL}}\left( \sigma\left(\frac{T(x)}{\tau}\right) \;\Bigg\|\; \sigma\left(\frac{S(x)}{\tau}\right) \right)$$

## Example

```python
import numpy as np
from chokkhu.compression import Quantizer

weights = np.random.randn(10, 10).astype(np.float32)
quantizer = Quantizer(bits=8)
quantized_weights, scale = quantizer.quantize(weights)
print("Quantized shape:", quantized_weights.shape)
```
