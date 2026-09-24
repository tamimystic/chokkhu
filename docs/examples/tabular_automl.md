# End-to-End Tabular AutoML Tutorial

In this tutorial, you will build an end-to-end classification pipeline with automated data cleaning, missing value imputation, PCA transformation, SMOTE resampling, cross-validated model selection, conformal prediction intervals, and dark-mode HTML dashboard generation.

---

## 1. Prepare Synthetic Dirty Dataset

```python
import numpy as np
import pandas as pd
import chokkhu as ck

np.random.seed(42)
n_samples = 300

# Create realistic messy dataset with NaNs, infs, and categorical columns
data = pd.DataFrame({
    "age": [np.nan if i % 12 == 0 else np.random.randint(18, 70) for i in range(n_samples)],
    "annual_income": [np.inf if i == 7 else np.random.randn() * 20000 + 60000 for i in range(n_samples)],
    "credit_score": np.random.randint(300, 850, size=n_samples),
    "education": np.random.choice(["High School", "BSc", "MSc", "PhD", np.nan], size=n_samples),
    "loan_approved": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})

print("Raw Dataset Preview:")
print(data.head())
```

---

## 2. Execute Zero-Leakage Pipeline

```python
result = ck.pipeline(
    data=data,
    target="loan_approved",
    task="classification",
    clean=True,
    preprocess={"scale": "standard", "encode": "onehot"},
    transform={"pca": 3},
    resample="smote",
    conformal_interval=0.95,
    test_size=0.2,
    random_state=42
)

# Print comprehensive performance summary
print(result.summary())
```

---

## 3. Generate Diagnostic Dashboard & Run Inference

```python
# Export standalone dark-mode HTML dashboard
ck.reports.create_diagnostic_dashboard(result, save_path="loan_approval_dashboard.html")

# Predict on new unseen applicant
new_applicant = pd.DataFrame({
    "age": [32],
    "annual_income": [75000.0],
    "credit_score": [720],
    "education": ["MSc"]
})

prediction = result.predict(new_applicant)
probabilities = result.predict_proba(new_applicant)

print(f"Loan Approval Decision: {'Approved' if prediction[0] == 1 else 'Rejected'}")
print(f"Approval Probability: {probabilities[0, 1]:.2%}")
```
