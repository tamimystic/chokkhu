from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..base import ChokkhuModel


class Node:
    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: Node | None = None,
        right: Node | None = None,
        *,
        value: Any | None = None,
        impurity: float = 0.0,
        n_node_samples: int = 0,
        probabilities: np.ndarray | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.probabilities = probabilities

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree(ChokkhuModel):
    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = 100,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        ccp_alpha: float = 0.0,
        task: str = "classification",
        n_features: Optional[int] = None,
        classes: Optional[np.ndarray] = None,
    ) -> None:
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = n_features if n_features is not None else max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.task = task
        self.n_features = self.max_features

        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = (
            np.asarray(classes) if classes is not None else None
        )
        self.n_classes_: int = len(self.classes_) if self.classes_ is not None else 0
        self.n_features_in_: int = 0
        self.feature_importances_: np.ndarray = np.array([])
        self._leaf_count: int = 0
        self._rng = np.random.RandomState(random_state)

    def _resolve_params(self, n_samples: int, n_features: int) -> Tuple[int, int, int]:
        min_split = (
            int(math.ceil(self.min_samples_split * n_samples))
            if isinstance(self.min_samples_split, float)
            and 0.0 < self.min_samples_split <= 1.0
            else max(2, int(self.min_samples_split))
        )
        min_leaf = (
            int(math.ceil(self.min_samples_leaf * n_samples))
            if isinstance(self.min_samples_leaf, float)
            and 0.0 < self.min_samples_leaf <= 1.0
            else max(1, int(self.min_samples_leaf))
        )
        if self.max_features is None:
            max_feats = n_features
        elif isinstance(self.max_features, str):
            if self.max_features in ("sqrt", "auto"):
                max_feats = max(1, int(math.sqrt(n_features)))
            elif self.max_features == "log2":
                max_feats = max(1, int(math.log2(n_features)))
            else:
                max_feats = n_features
        elif isinstance(self.max_features, float):
            max_feats = max(1, int(self.max_features * n_features))
        else:
            max_feats = min(n_features, max(1, int(self.max_features)))
        return min_split, min_leaf, max_feats

    def _compute_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None or self.task != "classification":
            return np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            weight_dict = {
                c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)
            }
        elif isinstance(self.class_weight, dict):
            weight_dict = self.class_weight
        else:
            return np.ones(len(y), dtype=np.float64)
        return np.array([weight_dict.get(val, 1.0) for val in y], dtype=np.float64)

    def _calculate_impurity(self, y: np.ndarray, weights: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return 0.0

        if self.task == "classification":
            if self.classes_ is None:
                self.classes_ = np.unique(y)
            probs = np.array(
                [float(np.sum(weights[y == c])) / total_weight for c in self.classes_],
                dtype=np.float64,
            )
            probs = probs[probs > 0.0]
            if self.criterion in ("entropy", "log_loss"):
                ent_val = float(np.sum(probs * np.log2(probs + 1e-15)))
                return -ent_val
            return float(1.0 - np.sum(probs**2))
        else:
            mean = np.average(y, weights=weights)
            if self.criterion in ("absolute_error", "mae"):
                return float(np.average(np.abs(y - np.median(y)), weights=weights))
            return float(np.average((y - mean) ** 2, weights=weights))

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DecisionTree:
        if y is None:
            raise ValueError("y cannot be None for Decision Tree")
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self._leaf_count = 0

        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(y_arr.dtype, np.floating)
                else "classification"
            )

        if self.task == "classification":
            if self.classes_ is None:
                self.classes_ = np.unique(y_arr)
                self.n_classes_ = len(self.classes_)
            if self.criterion not in ("gini", "entropy", "log_loss"):
                self.criterion = "gini"
        else:
            if self.criterion not in (
                "squared_error",
                "variance",
                "mse",
                "absolute_error",
                "mae",
                "friedman_mse",
            ):
                self.criterion = "squared_error"

        min_split, min_leaf, max_feats = self._resolve_params(n_samples, n_features)
        weights = self._compute_weights(y_arr)
        raw_importances = np.zeros(n_features, dtype=np.float64)

        self.root = self._grow_tree(
            X_arr,
            y_arr,
            weights,
            depth=0,
            min_split=min_split,
            min_leaf=min_leaf,
            max_feats=max_feats,
            total_samples=n_samples,
            raw_importances=raw_importances,
        )

        sum_imp = float(np.sum(raw_importances))
        self.feature_importances_ = (
            raw_importances / sum_imp if sum_imp > 0.0 else np.zeros(n_features)
        )
        return self

    def _leaf_node(self, y: np.ndarray, weights: np.ndarray) -> Node:
        self._leaf_count += 1
        impurity = self._calculate_impurity(y, weights)
        if self.task == "classification":
            if self.classes_ is None:
                self.classes_ = np.unique(y)
            counts = np.array([np.sum(weights[y == c]) for c in self.classes_])
            total = float(np.sum(counts))
            probs = (
                counts / total
                if total > 0.0
                else np.ones(len(self.classes_)) / len(self.classes_)
            )
            best_idx = int(np.argmax(probs))
            return Node(
                value=self.classes_[best_idx],
                impurity=impurity,
                n_node_samples=len(y),
                probabilities=probs,
            )
        else:
            val = float(np.average(y, weights=weights)) if len(y) > 0 else 0.0
            return Node(value=val, impurity=impurity, n_node_samples=len(y))

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        min_split: int,
        min_leaf: int,
        max_feats: int,
        total_samples: int,
        raw_importances: np.ndarray,
    ) -> Node:
        n_samples, n_feats = X.shape
        n_unique_y = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_unique_y == 1
            or n_samples < min_split
            or (
                self.max_leaf_nodes is not None
                and self._leaf_count >= self.max_leaf_nodes - 1
            )
        ):
            return self._leaf_node(y, weights)

        feat_idxs = (
            np.random.choice(n_feats, max_feats, replace=False)
            if self.random_state is None
            else self._rng.choice(n_feats, max_feats, replace=False)
        )
        best_feat, best_thr, best_gain = self._best_split(
            X, y, weights, feat_idxs, min_leaf
        )

        if best_feat is None or best_thr is None:
            return self._leaf_node(y, weights)

        left_idxs = X[:, best_feat] <= best_thr
        right_idxs = ~left_idxs

        norm_gain = (n_samples / float(total_samples)) * best_gain
        if norm_gain < self.min_impurity_decrease:
            return self._leaf_node(y, weights)

        raw_importances[best_feat] += norm_gain

        parent_impurity = self._calculate_impurity(y, weights)
        left = self._grow_tree(
            X[left_idxs],
            y[left_idxs],
            weights[left_idxs],
            depth + 1,
            min_split,
            min_leaf,
            max_feats,
            total_samples,
            raw_importances,
        )
        right = self._grow_tree(
            X[right_idxs],
            y[right_idxs],
            weights[right_idxs],
            depth + 1,
            min_split,
            min_leaf,
            max_feats,
            total_samples,
            raw_importances,
        )

        return Node(
            feature=best_feat,
            threshold=best_thr,
            left=left,
            right=right,
            impurity=parent_impurity,
            n_node_samples=n_samples,
        )

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feat_idxs: np.ndarray,
        min_leaf: int,
    ) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -1.0
        best_feat: Optional[int] = None
        best_thr: Optional[float] = None
        parent_impurity = self._calculate_impurity(y, weights)

        for f_idx in feat_idxs:
            col = X[:, f_idx]
            if self.splitter == "random":
                c_min, c_max = float(np.min(col)), float(np.max(col))
                thresholds = (
                    np.array([self._rng.uniform(c_min, c_max)])
                    if c_min < c_max
                    else np.array([])
                )
            else:
                thresholds = np.unique(col)
                if len(thresholds) > 50:
                    thresholds = np.percentile(col, np.linspace(2, 98, 50))

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                n_l, n_r = int(np.sum(left_mask)), int(np.sum(right_mask))
                if n_l < min_leaf or n_r < min_leaf:
                    continue

                w_l = float(np.sum(weights[left_mask]))
                w_r = float(np.sum(weights[right_mask]))
                total_w = w_l + w_r
                if total_w <= 0.0:
                    continue

                imp_l = self._calculate_impurity(y[left_mask], weights[left_mask])
                imp_r = self._calculate_impurity(y[right_mask], weights[right_mask])
                gain = parent_impurity - (
                    (w_l / total_w) * imp_l + (w_r / total_w) * imp_r
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thr = float(thr)

        return best_feat, best_thr, max(0.0, best_gain)

    def _traverse(self, x: np.ndarray, node: Node) -> Node:
        if node.is_leaf_node():
            return node
        if node.feature is not None and node.threshold is not None:
            if x[node.feature] <= node.threshold:
                if node.left:
                    return self._traverse(x, node.left)
            else:
                if node.right:
                    return self._traverse(x, node.right)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.root:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        preds = [self._traverse(x, self.root).value for x in X_arr]
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.root:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        X_arr = np.asarray(X, dtype=np.float64)
        n_classes = (
            len(self.classes_) if self.classes_ is not None else max(1, self.n_classes_)
        )
        probas = []
        for x in X_arr:
            leaf = self._traverse(x, self.root)
            if leaf.probabilities is not None and len(leaf.probabilities) == n_classes:
                probas.append(leaf.probabilities)
            else:
                p: np.ndarray = np.zeros(n_classes, dtype=np.float64)
                if self.classes_ is not None and leaf.value in self.classes_:
                    idx = int(np.where(self.classes_ == leaf.value)[0][0])
                    p[idx] = 1.0
                else:
                    p[0] = 1.0
                probas.append(p)
        return np.array(probas, dtype=np.float64)


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "classification")
        super().__init__(**kwargs)


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "regression")
        kwargs.setdefault("criterion", "squared_error")
        super().__init__(**kwargs)
