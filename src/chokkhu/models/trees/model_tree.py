"""Piecewise Linear Model Trees (M5 Algorithm) and RuleFit Interpretable Ensembles.

Formulated from first principles using standard deviation reduction (SDR) partitioning,
leaf multivariate Ridge regressions, tree rule extraction, and Lasso coordinate descent in pure NumPy.
"""

from typing import List, Optional, Tuple

import numpy as np


class _M5Node:
    """Internal node for M5 Model Tree."""

    def __init__(
        self,
        depth: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.depth = depth
        self.is_leaf = is_leaf
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_M5Node"] = None
        self.right: Optional["_M5Node"] = None
        # Leaf linear model parameters: w (d,) and b (scalar)
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean_val: float = 0.0
        self.num_samples: int = 0


class M5ModelTree:
    r"""M5 Model Tree (Piecewise Linear Regression Tree).

        Combines decision tree partitioning with multivariate linear regression models at leaves:
            ext{SDR} = \sigma(T) - \sum_{i \in \{L, R\}}
    rac{|T_i|}{|T|} \sigma(T_i)

        Parameters
        ----------
        max_depth : int, default=5
            Maximum tree depth.
        min_samples_split : int, default=10
            Minimum number of samples required to split an internal node.
        min_sdr_gain : float, default=1e-4
            Minimum standard deviation reduction gain required for split.
        l2_reg : float, default=1e-3
            L2 ridge regularization penalty for leaf linear models.
        smoothing : float, default=15.0
            Smoothing constant k for boundary continuity: p' = (n p + k q) / (n + k).
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_sdr_gain: float = 1e-4,
        l2_reg: float = 1e-3,
        smoothing: float = 15.0,
    ) -> None:
        self.max_depth = max(1, int(max_depth))
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_sdr_gain = float(min_sdr_gain)
        self.l2_reg = float(l2_reg)
        self.smoothing = float(smoothing)
        self.root: Optional[_M5Node] = None

    def _fit_leaf_linear_model(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fit L2 regularized Ridge linear regression at a leaf node."""
        N, d = X.shape
        if N < 2:
            return np.zeros(d), float(np.mean(y)) if N == 1 else 0.0

        # Ridge normal equations: (X_c^T X_c + lambda I) w = X_c^T y_c
        X_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))

        X_c = X - X_mean
        y_c = y - y_mean

        A = np.dot(X_c.T, X_c) + self.l2_reg * np.eye(d)
        b_vec = np.dot(X_c.T, y_c)

        try:
            w = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A).dot(b_vec)

        bias = y_mean - float(np.dot(X_mean, w))
        return w, bias

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> _M5Node:
        """Recursively build M5 tree partitions."""
        N, d = X.shape
        node = _M5Node(depth=depth)
        node.num_samples = N
        node.mean_val = float(np.mean(y)) if N > 0 else 0.0

        # Fit leaf model for this node
        w, bias = self._fit_leaf_linear_model(X, y)
        node.weights = w
        node.bias = bias

        sd_parent = float(np.std(y)) if N > 1 else 0.0

        if depth >= self.max_depth or N < self.min_samples_split or sd_parent < 1e-8:
            node.is_leaf = True
            return node

        # Search best split maximizing Standard Deviation Reduction (SDR)
        best_sdr = -1.0
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_left_mask: Optional[np.ndarray] = None

        for feat in range(d):
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue

            # Candidate thresholds
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 20:
                # Subsample quantiles
                thresholds = np.quantile(vals, np.linspace(0.05, 0.95, 20))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                n_left = int(np.sum(left_mask))
                n_right = N - n_left

                if n_left < 2 or n_right < 2:
                    continue

                sd_left = float(np.std(y[left_mask]))
                sd_right = float(np.std(y[~left_mask]))

                sdr_gain = sd_parent - (
                    (n_left / N) * sd_left + (n_right / N) * sd_right
                )

                if sdr_gain > best_sdr and sdr_gain >= self.min_sdr_gain:
                    best_sdr = sdr_gain
                    best_feat = feat
                    best_thresh = float(thresh)
                    best_left_mask = left_mask

        if best_feat is None or best_thresh is None or best_left_mask is None:
            node.is_leaf = True
            return node

        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        node.right = self._build_tree(X[~best_left_mask], y[~best_left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M5ModelTree":
        """Fit M5 model tree on dataset."""
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float).flatten()
        self.root = self._build_tree(X_mat, y_vec, depth=0)
        return self

    def _predict_sample(self, node: _M5Node, x: np.ndarray) -> float:
        """Predict sample with recursive parent smoothing."""
        if (
            node.is_leaf
            or node.left is None
            or node.right is None
            or node.feature_idx is None
            or node.threshold is None
        ):
            assert node.weights is not None
            return float(np.dot(x, node.weights) + node.bias)

        if x[node.feature_idx] <= node.threshold:
            pred_child = self._predict_sample(node.left, x)
        else:
            pred_child = self._predict_sample(node.right, x)

        if self.smoothing <= 0.0:
            return pred_child

        # Parent model prediction
        assert node.weights is not None
        pred_parent = float(np.dot(x, node.weights) + node.bias)
        n = node.num_samples
        k = self.smoothing
        return float((n * pred_child + k * pred_parent) / (n + k))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for batch of samples."""
        if self.root is None:
            raise ValueError("M5ModelTree has not been fitted yet.")
        X_mat = np.asarray(X, dtype=float)
        preds = [self._predict_sample(self.root, row) for row in X_mat]
        return np.array(preds)


class RuleFitRegressor:
    """RuleFit Sparse Interpretable Rule Ensemble for Regression.

    Extracts binary decision rules from decision trees and fits sparse Lasso models:
    f(x) = a_0 + sum_j beta_j x_j + sum_m alpha_m r_m(x)

    Parameters
    ----------
    num_trees : int, default=10
        Number of decision trees to extract rules from.
    max_depth : int, default=3
        Maximum depth of rule trees.
    alpha : float, default=0.01
        L1 Lasso regularization penalty coefficient.
    max_iter : int, default=200
        Lasso coordinate descent iterations.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        num_trees: int = 10,
        max_depth: int = 3,
        alpha: float = 0.01,
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        self.num_trees = max(1, int(num_trees))
        self.max_depth = max(1, int(max_depth))
        self.alpha = float(alpha)
        self.max_iter = max(10, int(max_iter))
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.rules: List[List[Tuple[int, str, float]]] = []
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

    def _extract_rules_from_tree(
        self, tree: M5ModelTree
    ) -> List[List[Tuple[int, str, float]]]:
        """Extract path conditions to all leaf nodes."""
        extracted: List[List[Tuple[int, str, float]]] = []

        def traverse(node: _M5Node, current_rule: List[Tuple[int, str, float]]) -> None:
            if node.is_leaf or node.feature_idx is None or node.threshold is None:
                if current_rule:
                    extracted.append(list(current_rule))
                return

            feat = node.feature_idx
            thresh = node.threshold

            # Left branch: <= thresh
            if node.left:
                traverse(node.left, current_rule + [(feat, "<=", thresh)])
            # Right branch: > thresh
            if node.right:
                traverse(node.right, current_rule + [(feat, ">", thresh)])

        if tree.root:
            traverse(tree.root, [])
        return extracted

    def _eval_rule(
        self, X: np.ndarray, rule: List[Tuple[int, str, float]]
    ) -> np.ndarray:
        """Evaluate binary indicator vector for a rule condition."""
        mask = np.ones(X.shape[0], dtype=bool)
        for feat, op, thresh in rule:
            if op == "<=":
                mask &= X[:, feat] <= thresh
            else:
                mask &= X[:, feat] > thresh
        return mask.astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleFitRegressor":
        """Fit RuleFit ensemble with Lasso coordinate descent."""
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float).flatten()
        N, d = X_mat.shape

        self.x_mean = np.mean(X_mat, axis=0)
        self.x_std = np.std(X_mat, axis=0) + 1e-6
        X_norm = (X_mat - self.x_mean) / self.x_std

        # 1. Build ensemble of M5 trees and extract rules
        self.rules = []
        for t_idx in range(self.num_trees):
            # Bootstrap subsample
            indices = self.rng.choice(N, size=int(0.8 * N), replace=True)
            tree = M5ModelTree(max_depth=self.max_depth, smoothing=0.0)
            tree.fit(X_mat[indices], y_vec[indices])
            tree_rules = self._extract_rules_from_tree(tree)
            self.rules.extend(tree_rules)

        # 2. Construct design matrix: [X_norm, R]
        if self.rules:
            R_mat = np.column_stack([self._eval_rule(X_mat, r) for r in self.rules])
            # Standardize rule indicators
            r_std = np.std(R_mat, axis=0) + 1e-6
            R_norm = (R_mat - np.mean(R_mat, axis=0)) / r_std
            Z = np.hstack([X_norm, R_norm])
        else:
            Z = X_norm

        # 3. Lasso Coordinate Descent
        n_features = Z.shape[1]
        w = np.zeros(n_features, dtype=float)
        y_mean = float(np.mean(y_vec))
        r_res = y_vec - y_mean

        norm_sq = np.sum(Z**2, axis=0) + 1e-12

        for _ in range(self.max_iter):
            for j in range(n_features):
                # Partial residual
                rho = float(np.dot(Z[:, j], r_res + w[j] * Z[:, j]))
                # Soft thresholding
                if rho < -self.alpha * N:
                    w_new = (rho + self.alpha * N) / norm_sq[j]
                elif rho > self.alpha * N:
                    w_new = (rho - self.alpha * N) / norm_sq[j]
                else:
                    w_new = 0.0

                diff = w_new - w[j]
                if abs(diff) > 1e-12:
                    r_res -= diff * Z[:, j]
                    w[j] = w_new

        self.weights = w
        self.intercept = y_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for batch of samples."""
        if self.weights is None or self.x_mean is None or self.x_std is None:
            raise ValueError("RuleFitRegressor has not been fitted yet.")

        X_mat = np.asarray(X, dtype=float)
        X_norm = (X_mat - self.x_mean) / self.x_std

        if self.rules:
            R_mat = np.column_stack([self._eval_rule(X_mat, r) for r in self.rules])
            r_std = np.std(R_mat, axis=0) + 1e-6
            R_norm = (R_mat - np.mean(R_mat, axis=0)) / r_std
            Z = np.hstack([X_norm, R_norm])
        else:
            Z = X_norm

        return np.dot(Z, self.weights) + self.intercept


class RuleFitClassifier:
    """RuleFit Sparse Interpretable Rule Ensemble for Binary Classification.

    Parameters
    ----------
    num_trees : int, default=10
        Number of decision trees to extract rules from.
    max_depth : int, default=3
        Maximum depth of rule trees.
    alpha : float, default=0.01
        L1 Lasso regularization penalty.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        num_trees: int = 10,
        max_depth: int = 3,
        alpha: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.regressor = RuleFitRegressor(
            num_trees=num_trees,
            max_depth=max_depth,
            alpha=alpha,
            seed=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleFitClassifier":
        """Fit RuleFit classifier."""
        y_arr = np.asarray(y, dtype=float).flatten()
        self.regressor.fit(X, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raw_preds = self.regressor.predict(X)
        probs = 1.0 / (1.0 + np.exp(np.clip(-raw_preds, -50.0, 50.0)))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels in {0, 1}."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
