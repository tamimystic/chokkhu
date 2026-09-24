# Causal Treatment Effects & Patient Survival Curves Tutorial

In this tutorial, you will estimate causal treatment effects and forecast right-censored patient survival curves.

---

## 1. Causal Treatment Effect Estimation with Doubly Robust Learner

```python
import numpy as np
import chokkhu as ck
from chokkhu.models.causal import DoublyRobustLearner

np.random.seed(42)
n = 500

# Confounders (Age, Blood Pressure, BMI, Biomarkers)
X = np.random.randn(n, 4)

# Treatment assignment mechanism (Propensity)
propensity_true = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
treatment = np.random.binomial(1, propensity_true)

# True Average Treatment Effect = +3.2 mmHg reduction
outcome = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 3.2 * treatment + np.random.randn(n) * 0.5

# Fit Doubly Robust Learner
dr_learner = DoublyRobustLearner()
dr_learner.fit(X, treatment, outcome)

estimated_ate = dr_learner.estimate_ate(X)
print(f"True ATE: 3.200 | Estimated ATE: {estimated_ate:.3f}")
```

---

## 2. Patient Survival Analysis with Kaplan-Meier

```python
from chokkhu.models.survival import KaplanMeierFitter

# Survival times in months
durations = np.array([3, 6, 12, 18, 24, 30, 36, 48, 60])
# 1 = observed event (mortality), 0 = right-censored
events = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0])

km = KaplanMeierFitter()
km.fit(durations, events)

print("Survival Curve Timeline & Probabilities:")
for t, p in zip(*km.survival_function_):
    print(f"  Month {t:2d}: {p:.2%}")
```
