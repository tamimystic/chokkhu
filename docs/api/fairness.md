# Algorithmic Fairness & Bias Mitigation API Reference

The `chokkhu.fairness` package detects, measures, and mitigates demographic disparity and disparate impact in machine learning models.

---

## 1. Fairness Metrics

### Demographic Parity Difference (`demographic_parity_difference`)
Measures the absolute difference in positive outcome selection rates across unprivileged ($A=0$) and privileged ($A=1$) sensitive attribute groups:

$$\text{DPD} = | P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1) |$$

### Equalized Odds Difference (`equalized_odds_difference`)
Measures the maximum disparity between False Positive Rates (FPR) and True Positive Rates (TPR) across groups:

$$\text{EOD} = \max\left( |\text{FPR}_{A=0} - \text{FPR}_{A=1}|, |\text{TPR}_{A=0} - \text{TPR}_{A=1}| \right)$$

---

## 2. Bias Mitigation Transformers

### Pre-Processing: Sample Reweighing (`ReweighingTransformer`)
Assigns statistical weights to training instances to decouple the sensitive attribute $A$ from the ground-truth label $Y$:

$$W(A=a, Y=y) = \frac{P(A=a) \cdot P(Y=y)}{P(A=a, Y=y)}$$

```python
import numpy as np
from chokkhu.fairness import ReweighingTransformer

X = np.random.randn(100, 4)
sensitive_attr = np.random.binomial(1, 0.4, size=100)
labels = np.random.binomial(1, 0.5, size=100)

reweighter = ReweighingTransformer()
sample_weights = reweighter.fit_transform(sensitive_attr, labels)
```

### Pre-Processing: Disparate Impact Remover (`DisparateImpactRemover`)
Edits continuous feature values via quantile alignment so that group conditional cumulative distributions become identical.

### Post-Processing: Threshold Optimizer (`ThresholdOptimizer`)
Finds group-specific decision classification thresholds $(\theta_{A=0}, \theta_{A=1})$ that maximize accuracy while satisfying equalized odds constraints.
