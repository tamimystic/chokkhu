import pandas as pd
import pytest
import numpy as np
from chokkhu.explainability import (
    explain,
    ExplanationResult,
)
from chokkhu.models.ml.random_forest import RandomForest


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 60
    X = np.random.randn(n, 4)
    # Feature 0 is dominant
    y = (2.0 * X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    feature_names = ["dominant", "weak", "noise1", "noise2"]
    return X, y, feature_names


def test_permutation_feature_importance(sample_dataset):
    X, y, feature_names = sample_dataset
    model = RandomForest(n_estimators=10, random_state=42)
    model.fit(X, y)

    res = explain(
        model=model,
        X=X,
        y=y,
        method="feature_importance",
        feature_names=feature_names,
        n_repeats=3,
        random_state=42,
        verbose=False,
    )
    assert isinstance(res, ExplanationResult)
    assert res.method == "feature_importance"
    assert len(res.importances) == 4
    assert res.importances[0] >= res.importances[2]

    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance" in df.columns

    summary = res.summary()
    assert "CHOKKHU EXPLAINABILITY REPORT" in summary


def test_kernel_shap(sample_dataset):
    X, y, feature_names = sample_dataset
    model = RandomForest(n_estimators=10, random_state=42)
    model.fit(X, y)

    res = explain(
        model=model,
        X=X,
        method="shap",
        feature_names=feature_names,
        n_samples=20,
        random_state=42,
        verbose=False,
    )
    assert isinstance(res, ExplanationResult)
    assert res.method == "shap"
    assert res.shap_values is not None
    assert res.shap_values.shape[1] == 4

    df = res.to_dataframe()
    assert "mean_abs_shap" in df.columns


def test_partial_dependence(sample_dataset):
    X, y, feature_names = sample_dataset
    model = RandomForest(n_estimators=10, random_state=42)
    model.fit(X, y)

    res = explain(
        model=model,
        X=X,
        method="pdp",
        feature_names=feature_names,
        pdp_feature="dominant",
        verbose=False,
    )
    assert isinstance(res, ExplanationResult)
    assert res.method == "pdp"
    assert "grid_values" in res.pdp_data
    assert len(res.pdp_data["grid_values"]) == 20
