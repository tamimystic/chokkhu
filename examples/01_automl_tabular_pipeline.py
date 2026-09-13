"""Example 1: Automated Machine Learning (AutoML) & Multi-Task Tabular Dispatcher."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chokkhu as ck
from chokkhu.pipeline.stacking import SuperLearner
from chokkhu.models.ml import LogisticRegression, DecisionTree, GradientBoosting


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 1: AUTOML & MULTI-TASK TABULAR PIPELINES")
    print("=" * 70)

    # 1. Synthetic dataset creation
    np.random.seed(42)
    n_samples = 400
    n_features = 8
    X_raw = np.random.randn(n_samples, n_features)
    y_raw = (X_raw[:, 0] * 1.5 - X_raw[:, 1] * 2.0 + X_raw[:, 2] ** 2 > 0.5).astype(int)

    feature_cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=feature_cols)
    df["target"] = y_raw

    # Introduce simulated missing values and noisy outliers
    df.loc[np.random.choice(n_samples, 15), "feature_1"] = np.nan
    df.loc[np.random.choice(n_samples, 5), "feature_2"] = 99.0

    print(
        f"Dataset summary: {df.shape[0]} rows, {df.shape[1]} columns (with simulated NaNs & Outliers)"
    )

    # 2. Universal Pipeline Dispatcher with Conformal Prediction
    print("\n[1] Running Sovereign End-to-End Classification Pipeline...")
    res = ck.pipeline(
        data=df,
        target="target",
        task="classification",
        time_budget=3.0,
        conformal_interval=0.95,
        save_report="reports/automl_report.html",
        verbose=False,
    )

    print(res.summary())
    sample_test = X_raw[:3]
    preds = res.predict(sample_test)
    probs = res.predict_proba(sample_test)
    print(f"  Test Predictions  : {preds}")
    print(f"  Class Probabilities:\n{np.round(probs, 3)}")

    # 3. SuperLearner Stacking Ensemble
    print("\n[2] Training SuperLearner Multi-Layer Stacking Classifier...")
    base_models = [
        LogisticRegression(learning_rate=0.05, epochs=50),
        DecisionTree(max_depth=4, task="classification"),
        GradientBoosting(n_estimators=10, max_depth=3),
    ]
    super_learner = SuperLearner(
        estimators=base_models,
        cv=3,
        task="classification",
    )
    X_clean = np.nan_to_num(X_raw, nan=0.0)
    super_learner.fit(X_clean, y_raw)
    stack_preds = super_learner.predict(X_clean[:5])
    print(f"  SuperLearner out-of-fold predictions on 5 samples: {stack_preds}")
    print("=" * 70)


if __name__ == "__main__":
    main()
