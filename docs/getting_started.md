# Getting Started with Chokkhu

Welcome to **Chokkhu (চক্ষু)**! This guide will walk you through installation, environment sanity checks, and 3 production-grade quickstart examples.

---

## 1. Installation

### From PyPI (Recommended)
```bash
pip install --upgrade chokkhu
```

### Using uv (Fast Package Manager)
```bash
uv pip install chokkhu
```

### From Source (Editable Mode)
```bash
git clone https://github.com/tamimystic/chokkhu.git
cd chokkhu
pip install -e .
```

---

## 2. Environment Verification

Run this quick one-liner to verify your installation, check startup latency, and confirm that zero heavy dependencies are loaded into memory:

```python
import time
t0 = time.perf_counter()
import chokkhu as ck
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"Chokkhu v{ck.__version__} initialized in {elapsed_ms:.2f} ms!")
```

Expected Output:
```
Chokkhu v1.6.0 initialized in ~12.4 ms!
```

---

## 3. Quickstart Recipes

### Recipe A: End-to-End Leak-Free AutoML Pipeline

```python
import chokkhu as ck
import pandas as pd
import numpy as np

# 1. Create a synthetic dirty dataset with NaNs and mixed types
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "income": np.random.randn(n) * 15000 + 50000,
    "debt_ratio": [np.nan if i % 10 == 0 else np.random.uniform(0.1, 0.8) for i in range(n)],
    "city": np.random.choice(["Dhaka", "Chittagong", "Sylhet"], size=n),
    "target": np.random.choice([0, 1], size=n, p=[0.75, 0.25])
})

# 2. Run the complete zero-leakage pipeline with conformal prediction intervals
result = ck.pipeline(
    data=data,
    target="target",
    task="classification",
    clean=True,
    preprocess={"scale": "standard", "encode": "onehot"},
    conformal_interval=0.95,
    test_size=0.2,
    random_state=42
)

# 3. View evaluation metrics & generate interactive SVG dashboard
print(result.summary())
ck.reports.create_diagnostic_dashboard(result, save_path="chokkhu_dashboard.html")

# 4. Predict on unseen inference data with zero data leakage
new_applicant = pd.DataFrame({
    "income": [62000.0],
    "debt_ratio": [0.25],
    "city": ["Dhaka"]
})
pred = result.predict(new_applicant)
print(f"Predicted Class: {pred[0]}")
```

---

### Recipe B: Frontier LLM Inference with Dynamic KV-Cache

```python
import numpy as np
import chokkhu as ck
from chokkhu.models.nlp.architectures.llama import LLaMA

# 1. Initialize a sovereign LLaMA architecture (pure NumPy)
vocab_size = 1000
model = LLaMA(
    vocab_size=vocab_size,
    dim=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,  # Grouped-Query Attention (GQA)
    max_seq_len=256
)

# 2. Input prompt token sequence
prompt_tokens = np.array([[12, 45, 88, 104]])

# 3. Autoregressive token generation with O(1) step KV-Cache & sampling
generated = model.generate(
    prompt_tokens,
    max_new_tokens=15,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2
)

print(f"Generated Token IDs: {generated.tolist()}")
```

---

### Recipe C: Solving Differential Equations with Scientific PINNs

```python
import numpy as np
import chokkhu as ck
from chokkhu.sciml.pinns import HeatPINN

# 1. Initialize 1D Heat Equation PINN (u_t = alpha * u_xx)
pinn = HeatPINN(hidden_dim=32, num_layers=3, alpha=0.01)

# 2. Generate spatiotemporal collocation domain points (x in [-1, 1], t in [0, 1])
x_colloc = np.random.uniform(-1, 1, size=(200, 1))
t_colloc = np.random.uniform(0, 1, size=(200, 1))

# Initial condition: u(x, 0) = -sin(pi * x)
x_init = np.random.uniform(-1, 1, size=(50, 1))
t_init = np.zeros_like(x_init)
u_init = -np.sin(np.pi * x_init)

# Boundary conditions: u(-1, t) = 0, u(1, t) = 0
t_bc = np.random.uniform(0, 1, size=(50, 1))
x_bc_left = -np.ones_like(t_bc)
x_bc_right = np.ones_like(t_bc)

# 3. Train PINN using physics loss + boundary condition loss
history = pinn.fit(
    x_colloc=x_colloc,
    t_colloc=t_colloc,
    x_init=x_init,
    t_init=t_init,
    u_init=u_init,
    x_bc_left=x_bc_left,
    x_bc_right=x_bc_right,
    t_bc=t_bc,
    epochs=100,
    lr=0.01
)

# 4. Predict temperature distribution u(x, t) at t = 0.5
x_eval = np.linspace(-1, 1, 20).reshape(-1, 1)
t_eval = np.full_like(x_eval, 0.5)
u_pred = pinn.predict(x_eval, t_eval)
print("Temperature distribution predicted successfully without PyTorch!")
```
