"""Unit tests for Multi-Task Universal Pipeline Dispatcher and SuperLearner OOF Stacking."""

import numpy as np
import pandas as pd
from chokkhu.models.ml import DecisionTree, LogisticRegression
from chokkhu.models.ml.glm import RidgeRegression
from chokkhu.pipeline import (
    ChokkhuPipeline,
    MultiTaskPipelineResult,
    PipelineResult,
    SuperLearner,
    dispatch_pipeline,
    pipeline,
)
from chokkhu.preprocessing import StandardScaler
from chokkhu.transformation import PCA


def test_super_learner_classification():
    """Test SuperLearner multi-model OOF stacking for binary classification."""
    np.random.seed(42)
    X = np.random.randn(100, 6)
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0).astype(int)

    base_models = [
        LogisticRegression(learning_rate=0.05, epochs=50),
        DecisionTree(task="classification", max_depth=3),
    ]

    sl = SuperLearner(
        estimators=base_models, task="classification", cv=3, random_state=42
    )
    sl.fit(X, y)

    assert sl.is_fitted_
    assert sl.weights_ is not None or sl.fitted_meta_estimator_ is not None

    preds = sl.predict(X[:10])
    assert len(preds) == 10
    assert set(np.unique(preds)).issubset({0, 1})

    probs = sl.predict_proba(X[:10])
    assert probs.shape[0] == 10
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_super_learner_regression():
    """Test SuperLearner stacking for regression with passthrough features."""
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + np.random.randn(80) * 0.1

    base_models = [
        RidgeRegression(alpha=0.5),
        DecisionTree(task="regression", max_depth=4),
    ]

    sl = SuperLearner(
        estimators=base_models,
        task="regression",
        cv=3,
        passthrough=True,
        random_state=42,
    )
    sl.fit(X, y)

    preds = sl.predict(X[:10])
    assert len(preds) == 10
    assert isinstance(preds, np.ndarray)


def test_chokkhu_pipeline_chaining_with_stacking():
    """Test fluent ChokkhuPipeline composition with SuperLearner."""
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] > 0).astype(int)

    base_models = [
        LogisticRegression(learning_rate=0.05, epochs=30),
        DecisionTree(task="classification", max_depth=2),
    ]
    stack = SuperLearner(estimators=base_models, task="classification", cv=2)

    pipe = (
        ChokkhuPipeline()
        .add_preprocessor(StandardScaler())
        .add_transformer(PCA(n_components=3))
        .add_model(stack)
    )
    pipe.fit(X, y)

    preds = pipe.predict(X[:5])
    assert len(preds) == 5


def test_multi_task_pipeline_classification_conformal():
    """Test 1-line pipeline with conformal interval calibration."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
            "target": np.random.choice([0, 1], size=100),
        }
    )

    res = pipeline(
        data=df,
        target="target",
        task="classification",
        conformal_interval=0.90,
        random_state=42,
        verbose=False,
    )

    assert isinstance(res, (PipelineResult, MultiTaskPipelineResult))
    assert res.task == "classification"
    assert "accuracy" in res.metrics

    preds = res.predict(df.drop(columns=["target"]).iloc[:5])
    assert len(preds) == 5

    summary = res.summary()
    assert "SUMMARY" in summary


def test_multi_task_pipeline_timeseries():
    """Test time-series forecasting pipeline auto-dispatch."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    vals = np.sin(t) + np.random.randn(100) * 0.1
    df = pd.DataFrame({"value": vals})

    res = dispatch_pipeline(
        data=df,
        target="value",
        task="timeseries_forecast",
        lags=4,
        test_size=0.2,
        verbose=False,
    )

    assert res.task == "timeseries_forecast"
    assert "mae" in res.metrics
    assert "rmse" in res.metrics

    fc = res.forecast(steps=5)
    assert len(fc) == 5


def test_multi_task_pipeline_causal_inference():
    """Test causal inference pipeline auto-dispatch."""
    np.random.seed(42)
    N = 100
    w = np.random.randn(N, 2)
    t = (w[:, 0] > 0).astype(int)
    y = 2.5 * t + w[:, 0] + np.random.randn(N) * 0.1

    df = pd.DataFrame(
        {
            "w1": w[:, 0],
            "w2": w[:, 1],
            "treatment": t,
            "outcome": y,
        }
    )

    res = dispatch_pipeline(
        data=df,
        treatment="treatment",
        target="outcome",
        task="causal_inference",
        verbose=False,
    )

    assert res.task == "causal_inference"
    ate = res.estimate_ate()
    assert isinstance(ate, float)
    assert np.isfinite(ate)

    effects = res.predict_effect(df.iloc[:5])
    assert len(effects) == 5


def test_multi_task_pipeline_survival():
    """Test survival analysis pipeline auto-dispatch."""
    np.random.seed(42)
    N = 80
    df = pd.DataFrame(
        {
            "age": np.random.uniform(30, 70, size=N),
            "duration": np.random.exponential(10, size=N) + 1.0,
            "event": np.random.choice([0, 1], size=N),
        }
    )

    res = dispatch_pipeline(
        data=df,
        duration="duration",
        event="event",
        task="survival",
        verbose=False,
    )

    assert res.task == "survival"
    assert "c_index" in res.metrics
    assert 0.0 <= res.metrics["c_index"] <= 1.0


def test_multi_task_pipeline_anomaly_and_clustering():
    """Test unsupervised anomaly detection and clustering pipelines."""
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(60, 4), columns=[f"col_{i}" for i in range(4)])

    # Anomaly
    res_anom = dispatch_pipeline(
        data=df,
        task="anomaly_detection",
        verbose=False,
    )
    assert res_anom.task == "anomaly_detection"
    assert "anomaly_ratio" in res_anom.metrics

    # Clustering
    res_clust = dispatch_pipeline(
        data=df,
        task="clustering",
        n_clusters=3,
        verbose=False,
    )
    assert res_clust.task == "clustering"
    assert res_clust.metrics["n_clusters"] == 3
