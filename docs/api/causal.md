# Causal Inference & Uplift Modeling API Reference

The `chokkhu.models.causal` module estimates Average Treatment Effects (ATE), Conditional Average Treatment Effects (CATE), and individual treatment uplift from observational data without randomized controlled trials.

---

## 1. Core Treatment Effect Estimators

### Doubly Robust Learner (`DoublyRobustLearner`)
Combines propensity scoring $e(X) = P(T=1|X)$ and outcome regression $\mu_{(t)}(X) = \mathbb{E}[Y|X, T=t]$. Consistent if **either** the propensity model OR the outcome model is correctly specified:

$$\hat{\tau}_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \left[ \mu_{(1)}(X_i) - \mu_{(0)}(X_i) + \frac{T_i (Y_i - \mu_{(1)}(X_i))}{e(X_i)} - \frac{(1 - T_i)(Y_i - \mu_{(0)}(X_i))}{1 - e(X_i)} \right]$$

```python
import numpy as np
from chokkhu.models.causal import DoublyRobustLearner

X = np.random.randn(200, 5)
treatment = np.random.binomial(1, 0.5, size=200)
# Real treatment effect = +2.5
outcome = 3.0 * X[:, 0] + 2.5 * treatment + np.random.randn(200) * 0.5

dr = DoublyRobustLearner()
dr.fit(X, treatment, outcome)
ate = dr.estimate_ate(X)
print(f"Estimated ATE: {ate:.3f}")
```

### Inverse Probability Weighting (`InverseProbabilityWeighting`)
Reweighs samples by the inverse of their estimated propensity score to create a pseudo-randomized population.

### Meta-Learners (`XLearner` / `TwoModelUplift`)
- **X-Learner**: Designed for unbalanced treatment vs. control group sizes.
- **TwoModelUplift**: Predicts incremental customer response for targeted marketing.

---

## 2. Evaluation Metrics

### Qini Score (`qini_score`)
Measures the area between the cumulative uplift curve and the random targeting diagonal:

$$Q = \int_0^1 (U(p) - p \cdot U(1)) \, dp$$
