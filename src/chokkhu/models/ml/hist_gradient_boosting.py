from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from chokkhu.models.base import ChokkhuModel


class HistBinMapper:
    def __init__(self, max_bins: int = 256) -> None:
        self.max_bins = max_bins
        self.bin_thresholds_: List[np.ndarray] = []

    def fit(self, X: np.ndarray) -> HistBinMapper:
        n_features = X.shape[1]
        self.bin_thresholds_ = []
        for f in range(n_features):
            vals = np.unique(X[:, f])
            if len(vals) <= self.max_bins:
                thresholds = vals[:-1]
            else:
                percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
                thresholds = np.unique(np.percentile(vals, percentiles))
            self.bin_thresholds_.append(thresholds)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        binned = np.zeros((n_samples, n_features), dtype=np.uint8)
        for f in range(n_features):
            thresholds = self.bin_thresholds_[f]
            binned[:, f] = np.digitize(X[:, f], thresholds).astype(np.uint8)
        return binned


class HistNode:
    def __init__(
        self,
        feature: Optional[int] = None,
        bin_idx: Optional[int] = None,
        value: Optional[float] = None,
        left: Optional[HistNode] = None,
        right: Optional[HistNode] = None,
        is_leaf: bool = False,
    ) -> None:
        self.feature = feature
        self.bin_idx = bin_idx
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf


class HistTree:
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        l2_regularization: float = 1.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.root: Optional[HistNode] = None

    def fit(
        self, X_binned: np.ndarray, gradients: np.ndarray, hessians: np.ndarray
    ) -> HistTree:
        sample_indices = np.arange(len(gradients))
        self.root = self._build_tree(
            X_binned, gradients, hessians, sample_indices, depth=0
        )
        return self

    def _build_tree(
        self,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_indices: np.ndarray,
        depth: int,
    ) -> HistNode:
        G = float(np.sum(gradients[sample_indices]))
        H = float(np.sum(hessians[sample_indices]))
        leaf_val = -G / (H + self.l2_regularization)

        if depth >= self.max_depth or len(sample_indices) < 2 * self.min_samples_leaf:
            return HistNode(value=leaf_val, is_leaf=True)

        n_features = X_binned.shape[1]
        best_gain = 0.0
        best_feat = -1
        best_bin = -1
        parent_score = (G**2) / (H + self.l2_regularization)

        for feat in range(n_features):
            feat_bins = X_binned[sample_indices, feat]
            max_b = int(np.max(feat_bins))
            if max_b == 0:
                continue

            g_hist = np.bincount(
                feat_bins, weights=gradients[sample_indices], minlength=max_b + 1
            )
            h_hist = np.bincount(
                feat_bins, weights=hessians[sample_indices], minlength=max_b + 1
            )
            cum_g = np.cumsum(g_hist)
            cum_h = np.cumsum(h_hist)
            cum_count = np.cumsum(np.bincount(feat_bins, minlength=max_b + 1))

            for b in range(max_b):
                n_left = cum_count[b]
                n_right = len(sample_indices) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                g_l = cum_g[b]
                h_l = cum_h[b]
                g_r = G - g_l
                h_r = H - h_l

                score = (
                    (g_l**2) / (h_l + self.l2_regularization)
                    + (g_r**2) / (h_r + self.l2_regularization)
                    - parent_score
                )
                if score > best_gain:
                    best_gain = score
                    best_feat = feat
                    best_bin = b

        if best_feat == -1:
            return HistNode(value=leaf_val, is_leaf=True)

        left_mask = X_binned[sample_indices, best_feat] <= best_bin
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]

        left_child = self._build_tree(
            X_binned, gradients, hessians, left_indices, depth + 1
        )
        right_child = self._build_tree(
            X_binned, gradients, hessians, right_indices, depth + 1
        )
        return HistNode(
            feature=best_feat,
            bin_idx=best_bin,
            value=leaf_val,
            left=left_child,
            right=right_child,
            is_leaf=False,
        )

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        n_samples = X_binned.shape[0]
        preds = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            curr = self.root
            while curr is not None and not curr.is_leaf:
                if curr.feature is not None and curr.bin_idx is not None:
                    curr = (
                        curr.left
                        if X_binned[i, curr.feature] <= curr.bin_idx
                        else curr.right
                    )
                else:
                    break
            preds[i] = (
                curr.value if curr is not None and curr.value is not None else 0.0
            )
        return preds


class HistGradientBoostingClassifier(ChokkhuModel):
    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 5,
        l2_regularization: float = 1.0,
        max_bins: int = 256,
        categorical_features: Any = None,
        monotonic_cst: Any = None,
        interaction_cst: Any = None,
        warm_start: bool = False,
        early_stopping: Any = "auto",
        scoring: Optional[str] = "loss",
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_depth = int(max_depth) if max_depth is not None else 6
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.bin_mapper = HistBinMapper(max_bins=self.max_bins)
        self.trees: List[HistTree] = []
        self.base_score: float = 0.0
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> HistGradientBoostingClassifier:
        if y is None:
            raise ValueError("y cannot be None for HistGradientBoostingClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        self.classes = np.unique(y_arr)

        y_binary = (
            (y_arr == self.classes[1]).astype(np.float64)
            if len(self.classes) == 2
            else y_arr.astype(np.float64)
        )
        p_mean = np.clip(np.mean(y_binary), 1e-6, 1.0 - 1e-6)
        self.base_score = float(np.log(p_mean / (1.0 - p_mean)))

        self.bin_mapper.fit(x_arr)
        x_binned = self.bin_mapper.transform(x_arr)
        raw_preds: np.ndarray = np.full(
            len(y_binary), self.base_score, dtype=np.float64
        )
        if not self.warm_start or not self.trees:
            self.trees = []

        for _ in range(len(self.trees), self.max_iter):
            p = 1.0 / (1.0 + np.exp(-raw_preds))
            gradients = p - y_binary
            hessians = np.maximum(p * (1.0 - p), 1e-6)

            tree = HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
            )
            tree.fit(x_binned, gradients, hessians)
            update = tree.predict(x_binned)
            raw_preds += self.learning_rate * update
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        x_binned = self.bin_mapper.transform(x_arr)
        raw_preds: np.ndarray = np.full(len(x_arr), self.base_score, dtype=np.float64)
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(x_binned)
        p1 = 1.0 / (1.0 + np.exp(-raw_preds))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class HistGradientBoostingRegressor(ChokkhuModel):
    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 5,
        l2_regularization: float = 1.0,
        max_bins: int = 256,
        categorical_features: Any = None,
        monotonic_cst: Any = None,
        interaction_cst: Any = None,
        warm_start: bool = False,
        early_stopping: Any = "auto",
        scoring: Optional[str] = "loss",
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_depth = int(max_depth) if max_depth is not None else 6
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.bin_mapper = HistBinMapper(max_bins=self.max_bins)
        self.trees: List[HistTree] = []
        self.base_score: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> HistGradientBoostingRegressor:
        if y is None:
            raise ValueError("y cannot be None for HistGradientBoostingRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        self.base_score = float(np.mean(y_arr))
        self.bin_mapper.fit(x_arr)
        x_binned = self.bin_mapper.transform(x_arr)
        raw_preds: np.ndarray = np.full(len(y_arr), self.base_score, dtype=np.float64)
        if not self.warm_start or not self.trees:
            self.trees = []

        for _ in range(len(self.trees), self.max_iter):
            gradients = raw_preds - y_arr
            hessians = np.ones_like(gradients, dtype=np.float64)

            tree = HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
            )
            tree.fit(x_binned, gradients, hessians)
            update = tree.predict(x_binned)
            raw_preds += self.learning_rate * update
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        x_binned = self.bin_mapper.transform(x_arr)
        raw_preds: np.ndarray = np.full(len(x_arr), self.base_score, dtype=np.float64)
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(x_binned)
        return raw_preds
