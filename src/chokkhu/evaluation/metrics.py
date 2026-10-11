from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    matrix: np.ndarray = np.zeros((n_classes, n_classes), dtype=int)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1

    return matrix, classes


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> tuple:
    matrix, classes = confusion_matrix(y_true, y_pred)
    n_classes = len(classes)

    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f1s = np.zeros(n_classes)

    for i in range(n_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[i] = (
            2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            if (precisions[i] + recalls[i]) > 0
            else 0.0
        )

    if average == "macro":
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
    elif average == "weighted":
        weights = np.sum(matrix, axis=1) / np.sum(matrix)
        return (
            float(np.average(precisions, weights=weights)),
            float(np.average(recalls, weights=weights)),
            float(np.average(f1s, weights=weights)),
        )
    else:
        raise ValueError("Unsupported average type. Use 'macro' or 'weighted'.")


def precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    p, _, _ = precision_recall_f1(y_true, y_pred, average=average)
    return float(p)


def recall_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    _, r, _ = precision_recall_f1(y_true, y_pred, average=average)
    return float(r)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    _, _, f1 = precision_recall_f1(y_true, y_pred, average=average)
    return float(f1)


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = [float(np.mean(y_pred[y_true == c] == c)) for c in classes]
    return float(np.mean(recalls))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.maximum(0.0, np.asarray(y_true, dtype=np.float64))
    y_p = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    return float(np.mean((np.log1p(y_t) - np.log1p(y_p)) ** 2))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_t), 1e-12)
    return float(np.mean(np.abs((y_t - y_p) / denom)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    var_diff = float(np.var(diff))
    var_true = float(np.var(y_true))
    return float(1.0 - (var_diff / max(var_true, 1e-12)))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_t = np.asarray(y_true)
    y_p: np.ndarray = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)

    if y_p.ndim == 1 or (y_p.ndim == 2 and y_p.shape[1] == 1):
        y_p_flat = y_p.flatten()
        return float(
            -np.mean(y_t * np.log(y_p_flat) + (1.0 - y_t) * np.log(1.0 - y_p_flat))
        )

    classes = np.unique(y_t)
    n_classes = len(classes)
    n_samples = len(y_t)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y_onehot: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, c in enumerate(y_t):
        Y_onehot[i, class_to_idx[c]] = 1.0

    loss_val = float(np.sum(Y_onehot * np.log(y_p)))
    return float(-loss_val / n_samples)


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    if y_p.ndim > 1 and y_p.shape[1] > 1:
        y_p = y_p[:, 1]
    return float(np.mean((y_t - y_p.flatten()) ** 2))


def hinge_loss(y_true: np.ndarray, pred_decision: np.ndarray) -> float:
    y_t = np.where(np.asarray(y_true) <= 0, -1.0, 1.0)
    pred_d = np.asarray(pred_decision, dtype=np.float64).flatten()
    return float(np.mean(np.maximum(0.0, 1.0 - y_t * pred_d)))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_s: np.ndarray = np.asarray(y_score, dtype=np.float64)
    if y_s.ndim > 1 and y_s.shape[1] > 1:
        y_s = y_s[:, 1]
    y_s = y_s.flatten()

    classes = np.unique(y_t)
    if len(classes) != 2:
        return 0.5

    pos_label = classes[1]
    y_bin = (y_t == pos_label).astype(int)

    n_pos: float = float(np.sum(y_bin == 1))
    n_neg: float = float(np.sum(y_bin == 0))
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5

    order = np.argsort(y_s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_s) + 1)

    sum_ranks_pos: float = float(np.sum(ranks[y_bin == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1.0)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_s: np.ndarray = np.asarray(y_score, dtype=np.float64)
    if y_s.ndim > 1 and y_s.shape[1] > 1:
        y_s = y_s[:, 1]
    y_s = y_s.flatten()

    classes = np.unique(y_t)
    if len(classes) != 2:
        return 0.5

    y_bin = (y_t == classes[1]).astype(int)
    thresholds = np.unique(y_s)
    thresholds = np.sort(thresholds)[::-1]

    precisions = []
    recalls = []
    n_pos: float = float(np.sum(y_bin == 1))
    if n_pos == 0.0:
        return 0.0

    for th in thresholds:
        y_pred = (y_s >= th).astype(int)
        tp: float = float(np.sum((y_pred == 1) & (y_bin == 1)))
        fp: float = float(np.sum((y_pred == 1) & (y_bin == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / n_pos
        precisions.append(prec)
        recalls.append(rec)

    p_arr = np.array([1.0] + precisions + [0.0])
    r_arr = np.array([0.0] + recalls + [1.0])
    return float(np.sum(0.5 * (p_arr[:-1] + p_arr[1:]) * np.diff(r_arr)))


def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_s.ndim > 1 and y_s.shape[1] > 1:
        y_s = y_s[:, 1]
    y_s = y_s.flatten()

    classes = np.unique(y_t)
    pos_label = classes[1] if len(classes) > 1 else classes[0]
    y_bin = (y_t == pos_label).astype(int)

    thresholds = np.unique(y_s)
    thresholds = np.sort(thresholds)[::-1]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]

    n_pos = float(np.sum(y_bin == 1))
    n_neg = float(np.sum(y_bin == 0))
    tpr_list, fpr_list = [0.0], [0.0]

    for th in thresholds[1:]:
        y_pred = (y_s >= th).astype(int)
        tp = float(np.sum((y_pred == 1) & (y_bin == 1)))
        fp = float(np.sum((y_pred == 1) & (y_bin == 0)))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_s.ndim > 1 and y_s.shape[1] > 1:
        y_s = y_s[:, 1]
    y_s = y_s.flatten()

    classes = np.unique(y_t)
    pos_label = classes[1] if len(classes) > 1 else classes[0]
    y_bin = (y_t == pos_label).astype(int)

    thresholds = np.sort(np.unique(y_s))
    n_pos = float(np.sum(y_bin == 1))
    p_list, r_list = [], []

    for th in thresholds:
        y_pred = (y_s >= th).astype(int)
        tp = float(np.sum((y_pred == 1) & (y_bin == 1)))
        fp = float(np.sum((y_pred == 1) & (y_bin == 0)))
        p_list.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        r_list.append(tp / n_pos if n_pos > 0 else 0.0)

    p_list.append(1.0)
    r_list.append(0.0)
    return np.array(p_list), np.array(r_list), thresholds


def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "accuracy",
    n_repeats: int = 5,
    random_state: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape

    def _score(preds: np.ndarray) -> float:
        if scoring == "r2":
            return r2_score(y_arr, preds)
        elif scoring in ("neg_mean_squared_error", "mse"):
            return -mean_squared_error(y_arr, preds)
        return accuracy_score(y_arr, preds)

    base_preds = model.predict(X_arr)
    base_score = _score(base_preds)

    importances = np.zeros((n_features, n_repeats), dtype=np.float64)
    for col in range(n_features):
        for rep in range(n_repeats):
            X_perm = np.copy(X_arr)
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            perm_preds = model.predict(X_perm)
            perm_score = _score(perm_preds)
            importances[col, rep] = base_score - perm_score

    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)
    return {
        "importances_mean": importances_mean,
        "importances_std": importances_std,
        "importances": importances,
    }
