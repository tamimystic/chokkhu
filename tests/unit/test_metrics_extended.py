import numpy as np
from chokkhu.evaluation.metrics import (
    roc_auc_score,
    pr_auc_score,
    log_loss,
    root_mean_squared_error,
)


def test_roc_auc_score():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = roc_auc_score(y_true, y_scores)
    assert 0.0 <= auc <= 1.0
    assert auc > 0.5


def test_pr_auc_score():
    y_true = np.array([0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    pr_auc = pr_auc_score(y_true, y_scores)
    assert 0.0 <= pr_auc <= 1.0


def test_log_loss():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    loss = log_loss(y_true, y_prob)
    assert loss > 0.0
    assert loss < 0.5


def test_root_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse == 0.0
