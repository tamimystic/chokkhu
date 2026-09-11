"""Learning-to-Rank Algorithms: LambdaMART and ListNet.

Formulated from first principles in pure NumPy and SciPy, implementing
Burges (2010) LambdaMART with pairwise NDCG@k lambda-gradients and Newton-Raphson
tree leaf updates, alongside Cao et al. (ICML 2007) ListNet top-1 listwise ranking.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


def _dcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    rel = np.asarray(relevance, dtype=np.float64)[:k]
    if len(rel) == 0:
        return 0.0
    gains = (2.0**rel) - 1.0
    discounts = np.log2(np.arange(len(rel)) + 2.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    actual_dcg = _dcg_at_k(relevance, k=k)
    ideal_relevance = np.sort(relevance)[::-1]
    ideal_dcg = _dcg_at_k(ideal_relevance, k=k)
    if ideal_dcg <= 1e-12:
        return 0.0
    return float(actual_dcg / ideal_dcg)


class _TreeNode:
    """Internal decision stump / tree node for LambdaMART."""

    def __init__(
        self,
        feature_idx: int = -1,
        threshold: float = 0.0,
        value: float = 0.0,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _RegressionTree:
    """Simple binary regression tree for gradient boosting in LambdaMART."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[_TreeNode] = None

    def fit(
        self, X: np.ndarray, lambdas: np.ndarray, weights: np.ndarray
    ) -> "_RegressionTree":
        self.root = self._build_tree(X, lambdas, weights, depth=0)
        return self

    def _build_tree(
        self,
        X: np.ndarray,
        lambdas: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        n_samples, n_features = X.shape
        leaf_val = float(np.sum(lambdas) / (np.sum(weights) + 1e-12))

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return _TreeNode(value=leaf_val)

        best_feat = -1
        best_thresh = 0.0
        best_gain = -1e12

        total_lambda: float = float(np.sum(lambdas))
        total_weight: float = float(np.sum(weights) + 1e-12)
        base_score = (total_lambda**2) / total_weight

        for feat in range(n_features):
            values = np.unique(X[:, feat])
            if len(values) <= 1:
                continue

            # Check percentile thresholds for speed
            if len(values) > 10:
                thresholds = np.percentile(values, np.linspace(10, 90, 9))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                n_left: int = int(np.sum(left_mask))
                n_right: int = int(np.sum(right_mask))
                if n_left < 1 or n_right < 1:
                    continue

                lambda_L: float = float(np.sum(lambdas[left_mask]))
                weight_L: float = float(np.sum(weights[left_mask]) + 1e-12)
                lambda_R: float = float(np.sum(lambdas[right_mask]))
                weight_R: float = float(np.sum(weights[right_mask]) + 1e-12)

                gain = (
                    ((lambda_L**2) / weight_L) + ((lambda_R**2) / weight_R) - base_score
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(thresh)

        if best_feat == -1:
            return _TreeNode(value=leaf_val)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(
            X[left_mask], lambdas[left_mask], weights[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], lambdas[right_mask], weights[right_mask], depth + 1
        )

        return _TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            value=leaf_val,
            left=left_child,
            right=right_child,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [self._predict_single(x, self.root) for x in X], dtype=np.float64
        )

    def _predict_single(self, x: np.ndarray, node: Optional[_TreeNode]) -> float:
        if node is None or node.is_leaf:
            return node.value if node else 0.0
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)


class LambdaMART:
    r"""LambdaMART: Gradient Boosted Trees for Direct NDCG Optimization.

    Optimizes listwise ranking metrics through pairwise lambda-gradient substitution
    and second-order Newton-Raphson leaf updates.

    Parameters
    ----------
    n_estimators : int, default=20
        Number of boosting trees.
    learning_rate : float, default=0.1
        Shrinkage parameter.
    max_depth : int, default=3
        Maximum tree depth for each weak learner.
    sigma : float, default=1.0
        Pairwise sigmoid temperature parameter.
    ndcg_k : int, default=10
        Truncation rank for Delta-NDCG calculation.
    seed : int, default=42
        Random state.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        sigma: float = 1.0,
        ndcg_k: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.sigma = float(sigma)
        self.ndcg_k = int(ndcg_k)
        self.seed = int(seed)

        self.trees: List[_RegressionTree] = []

    def _compute_lambdas_for_query(
        self,
        y: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute pairwise lambda gradients and diagonal Hessian weights for one query list."""
        n = len(y)
        lambdas: np.ndarray = np.zeros(n, dtype=np.float64)
        weights: np.ndarray = np.zeros(n, dtype=np.float64)

        if n <= 1:
            return lambdas, weights

        # Ranking permutation based on predicted scores
        ranked_indices = np.argsort(scores)[::-1]
        ranks: np.ndarray = np.zeros(n, dtype=np.int32)
        for r, idx in enumerate(ranked_indices):
            ranks[idx] = r + 1

        # Ideal DCG for this query
        ideal_relevance = np.sort(y)[::-1]
        idcg = _dcg_at_k(ideal_relevance, k=self.ndcg_k)
        if idcg <= 1e-12:
            return lambdas, weights

        # Pairwise computations
        for i in range(n):
            for j in range(n):
                if y[i] > y[j]:
                    # Delta NDCG when swapping doc i and doc j
                    ri = ranks[i]
                    rj = ranks[j]
                    disc_i = 1.0 / np.log2(ri + 1.0)
                    disc_j = 1.0 / np.log2(rj + 1.0)
                    gain_diff = (2.0 ** y[i]) - (2.0 ** y[j])
                    delta_ndcg = np.abs((gain_diff * (disc_i - disc_j)) / idcg)

                    # Logistic gradient
                    score_diff = self.sigma * (scores[i] - scores[j])
                    rho = 1.0 / (1.0 + np.exp(np.clip(score_diff, -30.0, 30.0)))
                    lambda_ij = self.sigma * rho * delta_ndcg
                    w_ij = (self.sigma**2) * rho * (1.0 - rho) * delta_ndcg

                    lambdas[i] += lambda_ij
                    lambdas[j] -= lambda_ij
                    weights[i] += w_ij
                    weights[j] += w_ij

        return lambdas, weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        query_ids: np.ndarray,
    ) -> "LambdaMART":
        r"""Fit the LambdaMART ensemble on query-grouped document features.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Document feature matrix.
        y : np.ndarray, shape (N,)
            Relevance labels.
        query_ids : np.ndarray, shape (N,)
            Query ID corresponding to each document instance.

        Returns
        -------
        self : LambdaMART
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        q_arr = np.asarray(query_ids)

        N, _ = X_arr.shape
        scores = np.zeros(N, dtype=np.float64)
        self.trees = []

        unique_queries = np.unique(q_arr)

        for _ in range(self.n_estimators):
            lambdas = np.zeros(N, dtype=np.float64)
            weights = np.zeros(N, dtype=np.float64)

            # Compute gradients per query group
            for q in unique_queries:
                q_mask = q_arr == q
                q_indices = np.where(q_mask)[0]
                q_y = y_arr[q_indices]
                q_scores = scores[q_indices]

                q_lambdas, q_weights = self._compute_lambdas_for_query(q_y, q_scores)
                lambdas[q_indices] = q_lambdas
                weights[q_indices] = q_weights

            # Fit new regression tree on gradient residuals
            tree = _RegressionTree(max_depth=self.max_depth)
            tree.fit(X_arr, lambdas, weights)
            self.trees.append(tree)

            # Update ensemble scores
            tree_pred = tree.predict(X_arr)
            scores += self.learning_rate * tree_pred

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute ranking score predictions for feature vectors.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        scores : np.ndarray, shape (N,)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        scores: np.ndarray = np.zeros(len(X_arr), dtype=np.float64)
        for tree in self.trees:
            scores += self.learning_rate * tree.predict(X_arr)
        return scores


class ListNet:
    r"""ListNet: Listwise Learning-to-Rank using Plackett-Luce Top-1 Probability Cross-Entropy.

    Directly optimizes the entire permutation ranking distribution across query documents.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    epochs : int, default=100
        Training epochs.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 100, seed: int = 42) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)

        self.W: np.ndarray = np.array([])
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, query_ids: np.ndarray) -> "ListNet":
        r"""Fit linear ListNet on query groups.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)
        query_ids : np.ndarray, shape (N,)

        Returns
        -------
        self : ListNet
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        q_arr = np.asarray(query_ids)

        N, D = X_arr.shape
        rng = np.random.RandomState(self.seed)

        self.W = rng.randn(D).astype(np.float64) * 0.01
        self.b = 0.0

        unique_queries = np.unique(q_arr)

        for _ in range(self.epochs):
            grad_W = np.zeros(D, dtype=np.float64)
            grad_b = 0.0

            for q in unique_queries:
                mask = q_arr == q
                q_X = X_arr[mask]
                q_y = y_arr[mask]

                if len(q_y) <= 1:
                    continue

                scores = np.dot(q_X, self.W) + self.b

                # Softmax top-1 probabilities
                exp_y = np.exp(q_y - np.max(q_y))
                p_y = exp_y / np.sum(exp_y)

                exp_s = np.exp(scores - np.max(scores))
                p_s = exp_s / np.sum(exp_s)

                # Cross-entropy gradient: \nabla_s L = p_s - p_y
                grad_s = p_s - p_y

                grad_W += np.dot(q_X.T, grad_s)
                grad_b += float(np.sum(grad_s))

            self.W -= self.lr * (grad_W / len(unique_queries))
            self.b -= self.lr * (grad_b / len(unique_queries))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.W) + self.b
