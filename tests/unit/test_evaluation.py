from __future__ import annotations

import numpy as np
from chokkhu.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_f1,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from chokkhu.evaluation.engine import evaluate


class DummyClassifier:
    def __init__(self):
        self.task = "classification"

    def predict(self, X):
        return np.array([0, 1, 0, 1])


class DummyRegressor:
    def __init__(self):
        self.task = "regression"

    def predict(self, X):
        return np.array([1.0, 2.0, 3.0, 4.0])


def test_classification_metrics():
    y_true = np.array([0, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 1])

    assert accuracy_score(y_true, y_pred) == 0.75

    cm, classes = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 1] == 1  # False positive
    assert cm[1, 1] == 1  # True positive

    prec, rec, f1 = precision_recall_f1(y_true, y_pred, average="macro")
    assert prec > 0
    assert rec > 0
    assert f1 > 0


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    mse = mean_squared_error(y_true, y_pred)
    assert mse == 0.25  # (1^2) / 4 = 0.25

    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 0.25  # 1 / 4 = 0.25

    r2 = r2_score(y_true, y_pred)
    assert r2 > 0.8


def test_evaluate_engine():
    X_test = np.zeros((4, 2))

    # Classification
    y_cls = np.array([0, 1, 0, 0])
    model_cls = DummyClassifier()
    res_cls = evaluate(model_cls, X_test, y_cls)
    assert "accuracy" in res_cls
    assert "f1_score" in res_cls

    # Regression
    y_reg = np.array([1.0, 2.0, 3.0, 5.0])
    model_reg = DummyRegressor()
    res_reg = evaluate(model_reg, X_test, y_reg)
    assert "mse" in res_reg
    assert "r2_score" in res_reg
