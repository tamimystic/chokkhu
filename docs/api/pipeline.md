# AutoML & End-to-End Pipeline API Reference

The `chokkhu.pipeline` and `chokkhu.automl` modules provide automated model selection, hyperparameter optimization, and leak-free machine learning orchestration.

---

## 1. `ck.pipeline`

Executes the complete End-to-End Machine Learning Pipeline with ZERO Data Leakage.

```python
import chokkhu as ck
import pandas as pd

df = pd.read_csv("dataset.csv")

result = ck.pipeline(
    data=df,
    target="label",
    task="classification",
    clean=True,
    preprocess={"scale": "standard", "encode": "onehot"},
    transform={"pca": 5},
    resample="smote",
    conformal_interval=0.95,
    test_size=0.2,
    random_state=42
)

# Inference on new raw data
y_pred = result.predict(new_df)
y_pred, y_low, y_high = result.predict_interval(new_df)
```

### Parameter Breakdown
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `str` / `pd.DataFrame` | *Required* | Raw tabular dataset or file path. |
| `target` | `str` / `None` | `None` | Target prediction column name. |
| `task` | `str` | `"auto"` | Learning task (`"classification"`, `"regression"`, `"timeseries_forecast"`, `"causal_inference"`, `"survival"`, `"anomaly_detection"`, `"clustering"`). |
| `clean` | `bool` / `dict` | `"auto"` | Data sanitation flag or configuration dictionary (missing value strategy, outlier clipping, dtype fixing). |
| `preprocess` | `bool` / `dict` | `"auto"` | Scaling and encoding options (`scale="standard"`, `encode="onehot"`). |
| `transform` | `dict` / `None` | `None` | Transformation configuration (`pca=N`, `lda=N`, `polynomial=2`). |
| `resample` | `str` / `None` | `None` | Resampling strategy for training data (`"smote"`, `"adasyn"`, `"random_oversample"`, `"random_undersample"`). |
| `conformal_interval` | `float` / `None` | `None` | Conformal uncertainty confidence level (e.g. `0.95` for $95\%$ empirical coverage). |
| `test_size` | `float` | `0.2` | Proportion of dataset held out for untouched test evaluation. |
| `random_state` | `int` | `42` | Pseudorandom seed for reproducible splitting. |

---

## 2. `ck.auto_train`

Searches candidate model families with automated hyperparameter tuning and cross-validation ranking.

```python
result = ck.auto_train(
    X_train,
    y_train,
    task="classification",
    time_budget_secs=60,
    metric="accuracy",
    cv=5,
    random_state=42
)
```

### Supported Classical Model Candidates
- `random_forest`: Random Forest with Out-of-Bag estimation.
- `gradient_boosting`: Sovereign gradient-boosted decision trees.
- `decision_tree`: CART decision tree classifier / regressor.
- `logistic_regression` / `linear_regression`: Differentiable generalized linear models.
- `ridge` / `lasso`: L1/L2 penalized regularized regression.
- `knn`: $k$-Nearest Neighbors classifier / regressor with KD-Tree.
- `naive_bayes`: Gaussian & Multinomial Naive Bayes.
- `svm`: Support Vector Classifier with SMO solver.
