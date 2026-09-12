"""Isolation Forest for Unsupervised Anomaly Detection in Pure NumPy.

References:
- Liu, Ting, Zhou (2008): "Isolation Forest" (ICDM).
- Liu, Ting, Zhou (2012): "Isolation-Based Anomaly Detection" (ACM TKDD).
"""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np

from chokkhu.models.base import ChokkhuModel


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in Binary Search Tree (BST)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_mascheroni = 0.5772156649015328606
    return float(
        2.0 * (np.log(n - 1.0) + euler_mascheroni) - (2.0 * (n - 1.0) / float(n))
    )


class IsolationTreeNode:
    """Node in an Isolation Tree."""

    def __init__(
        self,
        feature: Optional[int] = None,
        split_value: Optional[float] = None,
        left: Optional[IsolationTreeNode] = None,
        right: Optional[IsolationTreeNode] = None,
        size: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size
        self.is_leaf = is_leaf


class IsolationTree:
    """Single Isolation Tree constructed via recursive random attribute partitioning."""

    def __init__(self, max_height: int, rng: np.random.RandomState) -> None:
        self.max_height = max_height
        self.rng = rng
        self.root: Optional[IsolationTreeNode] = None

    def fit(self, X: np.ndarray) -> IsolationTree:
        self.root = self._build_tree(X, current_height=0)
        return self

    def _build_tree(self, X: np.ndarray, current_height: int) -> IsolationTreeNode:
        n_samples, n_features = X.shape
        if current_height >= self.max_height or n_samples <= 1:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        # Find features that are non-constant
        valid_features: List[int] = []
        for feat in range(n_features):
            col_min = float(np.min(X[:, feat]))
            col_max = float(np.max(X[:, feat]))
            if col_min < col_max:
                valid_features.append(feat)

        if not valid_features:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        feat_idx = int(self.rng.choice(valid_features))
        col_min = float(np.min(X[:, feat_idx]))
        col_max = float(np.max(X[:, feat_idx]))
        split_val = float(self.rng.uniform(col_min, col_max))

        left_mask = X[:, feat_idx] < split_val
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], current_height + 1)
        right_node = self._build_tree(X[right_mask], current_height + 1)

        return IsolationTreeNode(
            feature=feat_idx,
            split_value=split_val,
            left=left_node,
            right=right_node,
            size=n_samples,
            is_leaf=False,
        )

    def path_length(self, x: np.ndarray) -> float:
        """Compute path length of sample x through this tree."""
        curr = self.root
        depth = 0.0
        while curr is not None and not curr.is_leaf:
            if curr.feature is None or curr.split_value is None:
                break
            if x[curr.feature] < curr.split_value:
                curr = curr.left
            else:
                curr = curr.right
            depth += 1.0

        if curr is not None and curr.size > 1:
            depth += _c_factor(curr.size)

        return depth


class IsolationForest(ChokkhuModel):
    """Ensemble of Isolation Trees for Outlier and Anomaly Detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = float(contamination)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.trees: List[IsolationTree] = []
        self.max_samples_val: int = 256
        self.threshold_: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> IsolationForest:
        """Fit Isolation Forest ensemble."""
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]

        if self.max_samples == "auto":
            self.max_samples_val = min(256, n_samples)
        else:
            self.max_samples_val = min(int(self.max_samples), n_samples)

        max_height = int(np.ceil(np.log2(max(self.max_samples_val, 2))))
        self.trees = []

        for _ in range(self.n_estimators):
            subsample_idx = self.rng.choice(
                n_samples, size=self.max_samples_val, replace=False
            )
            itree = IsolationTree(max_height=max_height, rng=self.rng)
            itree.fit(x_arr[subsample_idx])
            self.trees.append(itree)

        # Compute threshold based on contamination
        scores = self.score_samples(x_arr)
        self.threshold_ = float(
            np.percentile(scores, 100.0 * (1.0 - self.contamination))
        )
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score s(x, n) where higher score represents greater anomaly."""
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        c_n = _c_factor(self.max_samples_val)
        if c_n == 0:
            c_n = 1.0

        avg_path_lengths = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            x_i = x_arr[i]
            total_len = sum(tree.path_length(x_i) for tree in self.trees)
            avg_path_lengths[i] = total_len / float(self.n_estimators)

        # Anomaly score s = 2^(-E(h(x)) / c(n))
        return 2.0 ** (-avg_path_lengths / c_n)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Average path anomaly score offset by threshold (negative for anomalies)."""
        scores = self.score_samples(X)
        return self.threshold_ - scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: -1 for anomalies / outliers, 1 for inliers."""
        scores = self.score_samples(X)
        preds: np.ndarray = np.ones(len(scores), dtype=np.int64)
        preds[scores >= self.threshold_] = -1
        return preds
