# Survival Analysis & Time-to-Event API Reference

The `chokkhu.models.survival` module models right-censored time-to-event outcomes for clinical trials, predictive maintenance, and customer churn.

---

## 1. Non-Parametric Estimators

### Kaplan-Meier Survival Fitter (`KaplanMeierFitter`)
Estimates the empirical survival function $S(t) = P(T > t)$:

$$\hat{S}(t) = \prod_{i: t_i \le t} \left( 1 - \frac{d_i}{n_i} \right)$$

```python
import numpy as np
from chokkhu.models.survival import KaplanMeierFitter

durations = np.array([5, 12, 18, 24, 30, 36, 42])
events = np.array([1, 1, 0, 1, 0, 1, 0])  # 0: right-censored, 1: event observed

km = KaplanMeierFitter()
km.fit(durations, events)
timeline, survival_probs = km.survival_function_
print(f"5-year Survival Probability: {km.predict(20):.3f}")
```

### Nelson-Aalen Cumulative Hazard (`NelsonAalenFitter`)
Estimates the cumulative hazard function $H(t) = \int_0^t h(u) \, du$.

---

## 2. Semi-Parametric & Deep Survival Models

### Cox Proportional Hazards (`CoxPHRegression`)
Models hazard as a function of covariates $X$:

$$h(t|X) = h_0(t) \exp(X \beta)$$

Optimized using the partial log-likelihood:

$$\ell(\beta) = \sum_{i: E_i=1} \left( X_i \beta - \ln \sum_{j \in R(T_i)} \exp(X_j \beta) \right)$$

### DeepSurv (`DeepSurv`)
Deep non-linear extension of the Cox model parameterized by a neural network.

---

## 3. Survival Metrics

### Concordance Index (`concordance_index`)
Evaluates model ranking discrimination over all comparable patient pairs:

$$C = \frac{\sum_{i,j} \mathbb{I}(T_i < T_j) \cdot \mathbb{I}(\hat{r}_i > \hat{r}_j) \cdot E_i}{\sum_{i,j} \mathbb{I}(T_i < T_j) \cdot E_i}$$

### Integrated Brier Score (`integrated_brier_score`)
Assesses both discrimination and calibration of predicted survival curves over time.

## Example

```python
import numpy as np
from chokkhu.models.survival.cox_ph import CoxPH

X = np.random.rand(50, 3)
durations = np.random.randint(1, 100, 50)
events = np.random.randint(0, 2, 50)

model = CoxPH()
model.fit(X, durations, events)
print("Cox Proportional Hazards model fitted.")
```
