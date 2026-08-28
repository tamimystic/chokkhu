from __future__ import annotations

import numpy as np
import pandas as pd


class _ITree:
    def __init__(self, X, current_height=0, max_height=10):
        self.size = len(X)
        self.split_feat = None
        self.split_val = None
        self.left = None
        self.right = None
        if current_height >= max_height or self.size <= 1:
            return
        n_features = X.shape[1]
        self.split_feat = np.random.randint(0, n_features)
        min_v, max_v = (X[:, self.split_feat].min(), X[:, self.split_feat].max())
        if min_v == max_v:
            return
        self.split_val = np.random.uniform(min_v, max_v)
        left_mask = X[:, self.split_feat] < self.split_val
        self.left = _ITree(X[left_mask], current_height + 1, max_height)
        self.right = _ITree(X[~left_mask], current_height + 1, max_height)


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in a Binary Search Tree (BST)."""
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    return float(2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n))


def _path_length(tree: _ITree, x: np.ndarray, current_depth: int = 0) -> float:
    if tree.left is None or tree.right is None:
        return current_depth + _c_factor(tree.size)
    if tree.split_feat is not None and tree.split_val is not None:
        if x[tree.split_feat] < tree.split_val:
            return _path_length(tree.left, x, current_depth + 1)
        return _path_length(tree.right, x, current_depth + 1)
    return float(current_depth)


def _isolation_scores(X: np.ndarray, n_trees: int = 50) -> np.ndarray:
    n_samples = len(X)
    subsample_size = min(256, n_samples)
    max_height = int(np.ceil(np.log2(max(2, subsample_size))))
    trees = [
        _ITree(
            X[np.random.choice(n_samples, subsample_size, replace=False)],
            max_height=max_height,
        )
        for _ in range(n_trees)
    ]
    scores = np.zeros(n_samples)
    # Use subsample size for c_factor normalization (Liu et al. 2008)
    c_psi = _c_factor(subsample_size)
    for i in range(n_samples):
        avg_path = float(np.mean([_path_length(t, X[i]) for t in trees]))
        scores[i] = 2.0 ** (-avg_path / c_psi)
    return scores


def handle_outliers(
    data: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
    zscore_threshold: float = 3.0,
    columns: list = None,
    percentile_low: float = 0.01,
    percentile_high: float = 0.99,
    action: str = "remove",
) -> pd.DataFrame:
    df = data.copy()
    if method is None:
        return df
    num_cols = (
        columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    )
    if not num_cols:
        return df
    if method == "log_transform":
        for col in num_cols:
            min_val = df[col].min()
            shift = abs(min_val) if min_val < 0 else 0.0
            df[col] = np.log1p(df[col] + shift)
        return df
    if method == "isolation":
        clean_df = df[num_cols].dropna()
        if len(clean_df) > 10:
            scores = _isolation_scores(clean_df.values)
            outlier_idx = clean_df.index[scores > 0.6]
            if action == "remove":
                return df.drop(index=outlier_idx)
            elif action == "nan":
                df.loc[outlier_idx, num_cols] = np.nan
        return df
    outlier_mask = pd.Series(False, index=df.index)
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        if method in ["iqr", "winsorize"]:
            q1, q3 = (s.quantile(0.25), s.quantile(0.75))
            iqr = q3 - q1
            low, high = (q1 - threshold * iqr, q3 + threshold * iqr)
        elif method == "zscore":
            mean, std = (s.mean(), s.std())
            if std == 0:
                continue
            low, high = (mean - zscore_threshold * std, mean + zscore_threshold * std)
        elif method == "modified_zscore":
            med = s.median()
            mad = np.median(np.abs(s - med))
            if mad == 0:
                continue
            diff = zscore_threshold * mad / 0.6745
            low, high = (med - diff, med + diff)
        elif method == "percentile":
            low, high = (s.quantile(percentile_low), s.quantile(percentile_high))
        else:
            continue
        col_outliers = (df[col] < low) | (df[col] > high)
        if method == "winsorize" or action == "cap":
            df[col] = df[col].clip(lower=low, upper=high)
        elif action == "nan":
            df.loc[col_outliers, col] = np.nan
        else:
            outlier_mask = outlier_mask | col_outliers
    if action == "remove" and method != "winsorize":
        df = df[~outlier_mask]
    return df
