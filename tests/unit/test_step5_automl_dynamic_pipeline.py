from __future__ import annotations

import tempfile
import numpy as np
import pandas as pd
from chokkhu.automl import auto_train, automl, AutoMLResult


def test_automl_raw_dataframe_with_nans_and_categories():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 4,
            "cat1": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B"] * 4,
            "target": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1] * 4,
        }
    )
    res = auto_train(
        df,
        target="target",
        candidate_models=["decision_tree", "logistic_regression"],
        cv=2,
        clean=True,
        preprocess=True,
        random_state=42,
    )
    assert isinstance(res, AutoMLResult)
    assert res.task == "classification"
    assert res.best_score > 0.5

    # Test prediction on raw DataFrame with NaNs and strings
    test_df = pd.DataFrame({"num1": [np.nan, 3.0], "cat1": ["A", "B"]})
    preds = res.predict(test_df)
    assert len(preds) == 2
    probs = res.predict_proba(test_df)
    assert probs.shape == (2, 2)


def test_automl_dynamic_parameter_overrides():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = (X[:, 0] > 0).astype(int)

    res = auto_train(
        X,
        y,
        task="classification",
        candidate_models=["random_forest", "decision_tree"],
        cv=2,
        model_params={
            "random_forest": {"n_estimators": 5, "max_depth": 3},
            "decision_tree": {"max_depth": 2},
        },
        random_state=42,
    )
    assert isinstance(res, AutoMLResult)
    assert res.best_score >= 0.0


def test_automl_ensemble():
    np.random.seed(42)
    X = np.random.randn(40, 4)
    y = 2.0 * X[:, 0] - X[:, 1] + 1.0

    res = auto_train(
        X,
        y,
        task="regression",
        candidate_models=["linear_regression", "ridge", "decision_tree"],
        ensemble=True,
        cv=2,
        random_state=42,
    )
    assert isinstance(res, AutoMLResult)
    assert res.best_model_name == "ensemble"
    preds = res.predict(X[:5])
    assert len(preds) == 5


def test_automl_save_and_load():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    res = automl(
        X,
        y,
        task="classification",
        candidate_models=["decision_tree"],
        cv=2,
        random_state=42,
    )
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    res.save(path)

    loaded = AutoMLResult.load(path)
    assert loaded.best_model_name == res.best_model_name
    np.testing.assert_allclose(res.predict(X[:5]), loaded.predict(X[:5]))
