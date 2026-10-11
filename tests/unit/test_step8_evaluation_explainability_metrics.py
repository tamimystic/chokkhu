import numpy as np

from chokkhu.evaluation import (
    balanced_accuracy_score,
    brier_score_loss,
    explained_variance_score,
    hinge_loss,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    median_absolute_error,
    permutation_importance,
    precision_recall_curve,
    roc_curve,
)
from chokkhu.models.ml import LogisticRegression


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    med_ae = median_absolute_error(y_true, y_pred)
    assert np.isclose(med_ae, 0.5)

    evs = explained_variance_score(y_true, y_pred)
    assert 0.0 <= evs <= 1.0

    y_pos_true = np.array([1.0, 4.0, 10.0])
    y_pos_pred = np.array([1.2, 3.8, 9.5])
    msle = mean_squared_log_error(y_pos_true, y_pos_pred)
    assert msle >= 0.0

    mape = mean_absolute_percentage_error(y_pos_true, y_pos_pred)
    assert mape >= 0.0


def test_classification_metrics_and_curves():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    y_pred = (y_prob >= 0.5).astype(int)

    brier = brier_score_loss(y_true, y_prob)
    assert 0.0 <= brier <= 1.0

    hinge = hinge_loss(y_true, np.array([-1.5, -0.2, 0.3, 1.8]))
    assert hinge >= 0.0

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    assert 0.0 <= bal_acc <= 1.0

    fpr, tpr, roc_th = roc_curve(y_true, y_prob)
    assert len(fpr) == len(tpr) == len(roc_th)

    p, r, pr_th = precision_recall_curve(y_true, y_prob)
    assert len(p) == len(r) == len(pr_th) + 1


def test_permutation_importance():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(epochs=50)
    model.fit(X, y)

    res = permutation_importance(model, X, y, n_repeats=3, random_state=42)
    assert "importances_mean" in res
    assert len(res["importances_mean"]) == 3
    assert res["importances_mean"][0] >= res["importances_mean"][1]
