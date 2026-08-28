<div align="center">

<img src="https://raw.githubusercontent.com/tamimystic/chokkhu/main/profile.jpg" width="140" height="140" style="border-radius:50%;" alt="Author Profile">

# Chokkhu

**An End-to-End, Production-Grade ML & Data Science Pipeline Toolkit for Tabular and Computer Vision Datasets.**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/chokkhu/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/chokkhu/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/tamimystic/chokkhu/blob/main/LICENSE)

> "Minimalistic Code. Maximum Output. Zero Heavy Dependencies. Zero Data Leakage."

</div>

---

## Highlights & Core Philosophy

1. **Zero Heavy Dependencies**: Built from scratch using pure NumPy, Pandas, SciPy, Matplotlib, Seaborn, and OpenCV-headless. No Scikit-Learn, PyTorch, TensorFlow, or imbalanced-learn required.
2. **Guaranteed Zero Data Leakage**: The automated `chokkhu.pipeline()` engine guarantees strict isolation between training and validation/test folds during preprocessing, scaling, encoding, and resampling.
3. **Explainable AI (XAI)**: Native model interpretability with KernelSHAP (Shapley value estimation), Permutation Feature Importance, and Partial Dependence Plots (PDP).
4. **Deep Learning from Scratch**: Built-in autograd tensor engine and Multi-Layer Perceptron (`NeuralNetwork`) with classification and regression support.
5. **Hyperparameter Tuning**: Automatic cross-validated grid search and tuning directly inside `chokkhu.train(..., tune=True)`.
6. **Command Line Interface (CLI)**: Run EDA, dataset cleaning, or complete pipelines straight from your terminal (`chokkhu pipeline ...`).

---

## Installation

Install Chokkhu directly from PyPI:

```bash
pip install --upgrade chokkhu
```

---

## Quickstart: Leak-Free End-to-End Pipeline in 3 Lines

```python
import chokkhu as ck

# Load data and execute complete leak-free ML pipeline
res = ck.pipeline(data="dataset.csv", target="price", model="random_forest", task="regression")

# Inspect comprehensive metrics and export fitted pipeline
print(res.summary())
res.save("trained_pipeline.pkl")

# Predict on new unseen raw data
new_preds = res.predict("unseen_data.csv")
```

---

## Complete Feature Walkthrough

### 1. Data Ingestion (`chokkhu.load` / `chokkhu.save`)
Auto-detects format from extension:
```python
import chokkhu as ck

# Tabular (CSV, TSV, JSON, Parquet, Excel, Feather)
df = ck.load("dataset.parquet")

# Image directory loading
img_data = ck.load("image_folder/", type="image", img_size=(128, 128))

# Save anything
ck.save(df, "clean_dataset.parquet")
```

---

### 2. Exploratory Data Analysis (`chokkhu.eda`)
Generate statistical analysis and responsive HTML reports:
```python
# Tabular EDA (Univariate, Bivariate, Correlation, VIF, Mahalanobis, Drift PSI)
ck.eda.tabular("data.csv", target_col="target", save_reports=True)

# Image EDA (Resolution distribution, blur score, entropy, SNR, duplicates)
ck.eda.image("image_folder/", save_reports=True)
```

---

### 3. Data Cleaning (`chokkhu.clean`)
Automated missing value handling, outlier detection, and duplicate removal:
```python
df_cleaned = ck.clean(
    data="raw_data.csv",
    missing="knn",            # 'mean', 'median', 'mode', 'knn', 'iterative', 'drop'
    outliers="isolation",      # 'iqr', 'zscore', 'modified_zscore', 'isolation', 'winsorize'
    duplicates=True,
    fix_data_types=True
)
```

---

### 4. Preprocessing & Feature Selection (`chokkhu.preprocess`)
Leak-free, stateful transformations:
```python
df_processed, state = ck.preprocess(
    data=df_cleaned,
    target="target",
    scale="standard",          # 'standard', 'minmax', 'robust', 'power', 'quantile'
    encode="onehot",           # 'onehot', 'label', 'binary', 'ordinal', 'target'
    select_features="rfe",     # 'variance', 'correlation', 'mutual_info', 'anova', 'rfe'
    select_k=10
)
```

---

### 5. Feature Engineering & Resampling (`chokkhu.transform`)
```python
df_trans = ck.transform(
    data=df_processed,
    target="target",
    pca=5,
    resample="smote",          # 'smote', 'adasyn', 'random_oversample', 'random_undersample', 'smote_tomek'
    polynomial=2
)
```

---

### 6. Data Splitting (`chokkhu.split`)
```python
# 2-way split
X_train, X_test, y_train, y_test = ck.split(df_trans, target="target", test_size=0.2, stratify=True)

# 3-way split (Train / Val / Test)
X_train, X_val, X_test, y_train, y_val, y_test = ck.split(df_trans, target="target", test_size=0.2, val_size=0.1)
```

---

### 7. Model Training & Auto-Tuning (`chokkhu.train`)

```python
# Train Random Forest with automatic hyperparameter tuning
model = ck.train(
    model="random_forest",
    X_train=X_train,
    y_train=y_train,
    tune=True,
    cv=3,
    random_state=42
)
print("Best parameters found:", model.best_params_)

# Train Deep Learning Neural Network (MLP) from scratch
dl_model = ck.train(
    model="neural_network",
    X_train=X_train,
    y_train=y_train,
    task="classification",
    layers=[64, 32],
    epochs=50,
    learning_rate=0.01
)
```

---

### 8. Model Evaluation (`chokkhu.evaluate`)

```python
results = ck.evaluate(model, X_test, y_test, task="classification", save_reports=True)
print("Accuracy:", results["accuracy"])
print("ROC-AUC:", results.get("roc_auc"))
```

---

### 9. Explainable AI (`chokkhu.explain`)

```python
# 1. Permutation Feature Importance
exp_pfi = ck.explain(model, X_test, y_test, method="feature_importance")
print(exp_pfi.summary())

# 2. KernelSHAP Feature Attribution
exp_shap = ck.explain(model, X_test, method="shap", n_samples=50)
print(exp_shap.to_dataframe().head())

# 3. Partial Dependence Plot (PDP)
exp_pdp = ck.explain(model, X_test, method="pdp", pdp_feature=0)
```

---

### 10. Command Line Interface (CLI)

Run operations directly from your terminal:

```bash
# Automated EDA
chokkhu eda --data dataset.csv --target price --save-reports

# Clean dataset
chokkhu clean --data raw.csv --missing median --outliers iqr --output clean.csv

# Execute full ML pipeline
chokkhu pipeline --data dataset.csv --target price --model random_forest --save model_pipe.pkl
```

---

## Feature Matrix Summary

| Phase | Capabilities | Available Implementations |
| :--- | :--- | :--- |
| **I/O** | Multi-Format Ingestion | CSV, TSV, JSON, Parquet, Excel, Feather, NumPy (.npy, .npz), Image Directories |
| **EDA** | Statistical & Visual Insights | Univariate, Bivariate, Multivariate (VIF, Mahalanobis, Cramér's V, PSI), GLCM, Image Quality |
| **Cleaning** | Data Sanitation | KNN, MICE Iterative, Mode/Median/Mean, Tukey IQR, Isolation Forest, Z-Score, Winsorization, Auto Dtype |
| **Preprocessing**| Scalers, Encoders, Selectors | Standard, MinMax, Robust, Power (Yeo-Johnson), Quantile; One-Hot, Binary, Target, Ordinal; ANOVA, RFE |
| **Transformation**| Embeddings & Balancing | PCA, SVD, LDA, t-SNE, SMOTE, ADASYN, Tomek Links, Polynomial Features, Image Augmenter |
| **Splitting** | Leak-Free Partitioning | Train/Test, Train/Val/Test, Stratified, K-Fold, TimeSeriesSplit |
| **Models** | Machine & Deep Learning | Linear/Logistic Regression, Ridge, Lasso, ElasticNet, KNN, NaiveBayes, SVM, DecisionTree, RandomForest, GBM, NeuralNetwork (MLP), KMeans, DBSCAN, Hierarchical, Q-Learning |
| **Tuning** | Automated Optimization | K-Fold Cross-Validated Hyperparameter Search (`tune=True`) |
| **Evaluation** | Research-Grade Metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, MSE, RMSE, MAE, R2-Score, ROC-AUC, PR-AUC, Log-Loss |
| **Explainability**| Model Interpretability | KernelSHAP, Permutation Feature Importance, Partial Dependence Plots (PDP) |
| **Pipeline** | Unified Architecture | Leak-free `pipeline()` engine with `TransformationState`, `PipelineResult`, and serialization |
| **CLI** | Terminal Automation | `chokkhu eda`, `chokkhu clean`, `chokkhu pipeline`, `chokkhu --version` |

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
