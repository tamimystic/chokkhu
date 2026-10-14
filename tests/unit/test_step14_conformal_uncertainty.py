import numpy as np
import pytest

from chokkhu.uncertainty import (
    ConformalClassifier,
    ConformalizedQuantileRegression,
    MondrianConformalClassifier,
    SplitConformalRegressor,
)


def test_split_conformal_regressor() -> None:
    np.random.seed(42)
    y_true_cal = np.random.randn(200)
    y_pred_cal = y_true_cal + np.random.normal(0, 0.5, size=200)

    reg = SplitConformalRegressor(alpha=0.1)
    reg.calibrate(y_true_cal, y_pred_cal)
    assert reg.is_calibrated_
    assert reg.quantile_ > 0.0

    y_test_pred = np.array([0.0, 1.0, -1.0])
    lower, upper = reg.predict_interval(y_test_pred)
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert np.all(lower < upper)

    eval_stats = reg.evaluate_coverage(y_true_cal, y_pred_cal)
    assert eval_stats["empirical_coverage"] >= 0.88
    assert eval_stats["average_width"] > 0.0

    # Uncalibrated error
    with pytest.raises(ValueError):
        SplitConformalRegressor().predict_interval(y_test_pred)


def test_conformal_classifier_lac_and_aps() -> None:
    np.random.seed(42)
    n = 300
    # Simulate calibrated probabilities for 3 classes
    logits = np.random.randn(n, 3)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    y_true = np.array([np.random.choice(3, p=probs[i]) for i in range(n)])

    # 1. Test Least Ambiguous Classifier (LAC)
    clf_lac = ConformalClassifier(alpha=0.1, method="lac")
    clf_lac.calibrate(probs[:200], y_true[:200])
    assert clf_lac.is_calibrated_
    eval_lac = clf_lac.evaluate_coverage(probs[200:], y_true[200:])
    assert eval_lac["empirical_coverage"] >= 0.85
    assert eval_lac["average_set_size"] >= 1.0

    # 2. Test Adaptive Prediction Sets (APS)
    clf_aps = ConformalClassifier(alpha=0.1, method="aps", random_state=42)
    clf_aps.calibrate(probs[:200], y_true[:200])
    assert clf_aps.is_calibrated_
    eval_aps = clf_aps.evaluate_coverage(probs[200:], y_true[200:])
    assert eval_aps["empirical_coverage"] >= 0.85
    assert eval_aps["average_set_size"] >= 1.0


def test_mondrian_conformal_classifier() -> None:
    np.random.seed(42)
    n = 300
    logits = np.random.randn(n, 3)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    y_true = np.array([np.random.choice(3, p=probs[i]) for i in range(n)])

    mondrian = MondrianConformalClassifier(alpha=0.15)
    mondrian.calibrate(probs[:200], y_true[:200])
    assert mondrian.is_calibrated_
    assert len(mondrian.calibrators_) == 3

    eval_stats = mondrian.evaluate_coverage(probs[200:], y_true[200:])
    assert eval_stats["empirical_coverage"] >= 0.80
    assert len(eval_stats["per_class_coverage"]) == 3


def test_cqr_regression_smoke() -> None:
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.dot(X, np.array([1.5, -2.0, 0.5, -0.8])) + np.random.normal(0, 0.2, 50)

    cqr = ConformalizedQuantileRegression(alpha=0.1)
    cqr.fit(X[:30], y[:30])
    cqr.calibrate(X[30:40], y[30:40])
    stats = cqr.evaluate_coverage(X[40:], y[40:])
    assert stats["empirical_coverage"] >= 0.70
