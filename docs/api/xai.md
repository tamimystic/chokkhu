# Explainable AI (XAI) & Interpretability API Reference

The `chokkhu.explainability` package provides gradient-based, perturbation-based, and mechanistic interpretability methods in pure NumPy.

---

## 1. Axiomatic Gradient Attribution

### Integrated Gradients (`IntegratedGradients`)
Satisfies Completeness and Implementation Invariance by integrating gradients along a straight interpolation path from baseline $x'$ to input $x$:

$$\text{IG}_i(x) = (x_i - x_i') \times \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} \, d\alpha$$

$$\text{Riemann Approximation: } \text{IG}_i(x) \approx \frac{x_i - x_i'}{m} \sum_{k=1}^m \frac{\partial F\left( x' + \frac{k}{m}(x - x') \right)}{\partial x_i}$$

```python
import numpy as np
from chokkhu.explainability import IntegratedGradients

ig = IntegratedGradients(model=my_model, steps=50)
x_input = np.random.randn(1, 10)
x_baseline = np.zeros_like(x_input)

attributions = ig.attribute(x_input, baseline=x_baseline, target_class=0)
print(f"Feature Attributions: {attributions}")
```

### SmoothGrad & DeepLIFT
- **SmoothGrad**: Averages saliency maps over Gaussian-perturbed inputs to eliminate noisy high-frequency artifacts: $\hat{S}(x) = \frac{1}{N} \sum_{i=1}^N \nabla_x F(x + \mathcal{N}(0, \sigma^2))$.
- **DeepLIFT**: Propagates activation differences from baseline references through non-linearities.

---

## 2. Game-Theoretic & Mechanistic Methods

### KernelSHAP (`KernelSHAP`)
Estimates Shapley values by fitting a weighted linear surrogate model over binary coalition masks:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} (f(S \cup \{i\}) - f(S))$$

### Causal Activation Patching
Replaces specific intermediate layer activations from a clean run with corrupted activations to isolate the exact computational circuit responsible for a model's prediction.

## Example

```python
import numpy as np
from chokkhu.xai import SHAPExplainer

# Mock model prediction function
predict_fn = lambda x: np.sum(x, axis=1)
X_background = np.random.rand(100, 5)
X_explain = np.random.rand(1, 5)

explainer = SHAPExplainer(predict_fn, X_background)
shap_values = explainer.explain(X_explain)
print("SHAP values:", shap_values)
```
