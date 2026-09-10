"""Matrix Factorization & Collaborative Filtering Recommender Systems.

Pure NumPy implementations of:
- SVDRecommender (FunkSVD with user/item biases & L2 regularization)
- SVDPlusPlus (SVD++ with implicit rating feedback history)
- NMFRecommender (Non-Negative Matrix Factorization with multiplicative updates)
- ImplicitALS (Alternating Least Squares for implicit feedback data)
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class SVDRecommender:
    r"""Funk-SVD Matrix Factorization with user/item biases.

    Formula:
        \hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i

    Parameters
    ----------
    n_factors : int, default=32
        Dimensionality of the latent factor space.
    n_epochs : int, default=20
        Number of SGD training epochs.
    lr : float, default=0.005
        Learning rate for SGD updates.
    reg : float, default=0.02
        L2 regularization parameter (lambda).
    seed : Optional[int], default=42
        Random seed for factor initialization.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        self.mu: float = 0.0
        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}

        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        self.user_interactions: Dict[int, Set[int]] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
        ratings: Union[List[float], np.ndarray],
    ) -> "SVDRecommender":
        """Train the SVD model on user-item interaction ratings."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)
        r_arr = np.asarray(ratings, dtype=np.float32)

        if len(u_arr) != len(i_arr) or len(u_arr) != len(r_arr):
            raise ValueError(
                "user_ids, item_ids, and ratings must have identical lengths."
            )

        # Build vocabulary mappings
        self.user_map = {}
        self.item_map = {}
        self.user_interactions = {}

        for u, i in zip(u_arr, i_arr):
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
                self.user_interactions[u_idx] = set()
            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

            self.user_interactions[self.user_map[u]].add(self.item_map[i])

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # Global mean
        self.mu = float(np.mean(r_arr))

        rng = np.random.RandomState(self.seed)
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        self.user_factors = (rng.randn(n_users, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.item_factors = (rng.randn(n_items, self.n_factors) * 0.1).astype(
            np.float32
        )

        u_indices = np.array([self.user_map[u] for u in u_arr], dtype=np.int32)
        i_indices = np.array([self.item_map[i] for i in i_arr], dtype=np.int32)
        n_samples = len(r_arr)

        for _ in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            for idx in perm:
                u = u_indices[idx]
                i = i_indices[idx]
                r = r_arr[idx]

                pred = (
                    self.mu
                    + self.user_bias[u]
                    + self.item_bias[i]
                    + float(np.dot(self.user_factors[u], self.item_factors[i]))
                )
                err = r - pred

                # Update biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # Update latent factors
                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()
                self.user_factors[u] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[i] += self.lr * (err * pu - self.reg * qi)

        self.is_fitted = True
        return self

    def predict(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> float:
        """Predict rating for a given user and item."""
        if not self.is_fitted or self.user_bias is None or self.item_bias is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(item_id)

        # Cold-start handling
        if u_idx is None and i_idx is None:
            return self.mu
        if u_idx is None:
            assert self.item_bias is not None
            return self.mu + float(self.item_bias[i_idx])
        if i_idx is None:
            assert self.user_bias is not None
            return self.mu + float(self.user_bias[u_idx])

        assert self.user_factors is not None
        assert self.item_factors is not None
        return float(
            self.mu
            + self.user_bias[u_idx]
            + self.item_bias[i_idx]
            + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
        filter_interacted: bool = True,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-K item recommendations for a user."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend.")

        u_idx = self.user_map.get(user_id)

        if u_idx is None:
            assert self.item_bias is not None
            scores = self.mu + self.item_bias
            all_indices = np.argsort(-scores)[:top_k]
            return [(self.reverse_item_map[i], float(scores[i])) for i in all_indices]

        pu = self.user_factors[u_idx]
        assert self.user_bias is not None
        assert self.item_bias is not None
        scores = (
            self.mu
            + self.user_bias[u_idx]
            + self.item_bias
            + np.dot(self.item_factors, pu)
        )

        if filter_interacted and u_idx in self.user_interactions:
            interacted = self.user_interactions[u_idx]
            for i in interacted:
                scores[i] = -float("inf")

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.reverse_item_map[i], float(scores[i]))
            for i in top_indices
            if scores[i] != -float("inf")
        ]

    def __repr__(self) -> str:
        return (
            f"SVDRecommender(n_factors={self.n_factors}, lr={self.lr}, "
            f"reg={self.reg}, users={len(self.user_map)}, items={len(self.item_map)})"
        )


class SVDPlusPlus:
    r"""SVD++ Recommender incorporating implicit rating feedback history.

        Formula:
            \hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left(p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j
    ight)
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        self.mu: float = 0.0
        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}

        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.implicit_factors: Optional[np.ndarray] = None

        self.user_history: Dict[int, List[int]] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
        ratings: Union[List[float], np.ndarray],
    ) -> "SVDPlusPlus":
        """Train SVD++ model on rating pairs."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)
        r_arr = np.asarray(ratings, dtype=np.float32)

        self.user_map = {}
        self.item_map = {}
        self.user_history = {}

        for u, i in zip(u_arr, i_arr):
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
                self.user_history[u_idx] = []
            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

            self.user_history[self.user_map[u]].append(self.item_map[i])

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.mu = float(np.mean(r_arr))

        rng = np.random.RandomState(self.seed)
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        self.user_factors = (rng.randn(n_users, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.item_factors = (rng.randn(n_items, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.implicit_factors = (rng.randn(n_items, self.n_factors) * 0.1).astype(
            np.float32
        )

        u_indices = np.array([self.user_map[u] for u in u_arr], dtype=np.int32)
        i_indices = np.array([self.item_map[i] for i in i_arr], dtype=np.int32)
        n_samples = len(r_arr)

        for _ in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            for idx in perm:
                u = u_indices[idx]
                i = i_indices[idx]
                r = r_arr[idx]

                history = self.user_history[u]
                sqrt_nu = max(1.0, np.sqrt(len(history)))
                y_sum = np.sum(self.implicit_factors[history], axis=0) / sqrt_nu
                u_impl = self.user_factors[u] + y_sum

                pred = (
                    self.mu
                    + self.user_bias[u]
                    + self.item_bias[i]
                    + float(np.dot(self.item_factors[i], u_impl))
                )
                err = r - pred

                # Updates
                qi = self.item_factors[i].copy()
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                self.user_factors[u] += self.lr * (
                    err * qi - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (err * u_impl - self.reg * qi)

                # Update implicit factor terms for history items
                grad_y = (err * qi / sqrt_nu) - self.reg * self.implicit_factors[
                    history
                ]
                self.implicit_factors[history] += self.lr * grad_y

        self.is_fitted = True
        return self

    def predict(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> float:
        """Predict rating for user and item."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(item_id)

        if u_idx is None and i_idx is None:
            return self.mu
        if u_idx is None:
            assert self.item_bias is not None
            return self.mu + float(self.item_bias[i_idx])
        if i_idx is None:
            assert self.user_bias is not None
            return self.mu + float(self.user_bias[u_idx])

        history = self.user_history.get(u_idx, [])
        sqrt_nu = max(1.0, np.sqrt(len(history))) if history else 1.0
        assert self.implicit_factors is not None
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        y_sum = (
            np.sum(self.implicit_factors[history], axis=0) / sqrt_nu if history else 0.0
        )
        u_impl = self.user_factors[u_idx] + y_sum

        return float(
            self.mu
            + self.user_bias[u_idx]
            + self.item_bias[i_idx]
            + np.dot(self.item_factors[i_idx], u_impl)
        )


class ImplicitALS:
    """Alternating Least Squares (iALS) for Implicit Feedback Datasets.

    Parameters
    ----------
    n_factors : int, default=32
        Latent dimension.
    n_epochs : int, default=15
        Number of alternating iterations.
    alpha : float, default=40.0
        Confidence scaling factor (c_ui = 1 + alpha * r_ui).
    reg : float, default=0.01
        L2 regularization weight (lambda).
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 15,
        alpha: float = 40.0,
        reg: float = 0.01,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.reg = reg
        self.seed = seed

        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
        confidences: Optional[Union[List[float], np.ndarray]] = None,
    ) -> "ImplicitALS":
        """Fit iALS model using alternating least squares solver."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)
        c_arr = (
            np.ones(len(u_arr), dtype=np.float32)
            if confidences is None
            else np.asarray(confidences, dtype=np.float32)
        )

        for u in u_arr:
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
        for i in i_arr:
            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        user_item_conf: Dict[int, Dict[int, float]] = {u: {} for u in range(n_users)}
        item_user_conf: Dict[int, Dict[int, float]] = {i: {} for i in range(n_items)}

        for u, i, c in zip(u_arr, i_arr, c_arr):
            u_i = self.user_map[u]
            i_i = self.item_map[i]
            user_item_conf[u_i][i_i] = float(c)
            item_user_conf[i_i][u_i] = float(c)

        rng = np.random.RandomState(self.seed)
        self.user_factors = (rng.randn(n_users, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.item_factors = (rng.randn(n_items, self.n_factors) * 0.1).astype(
            np.float32
        )

        reg_I = self.reg * np.eye(self.n_factors, dtype=np.float32)

        for _ in range(self.n_epochs):
            # 1. Update User Factors
            YTY = np.dot(self.item_factors.T, self.item_factors)
            for u in range(n_users):
                items = list(user_item_conf[u].keys())
                if not items:
                    continue
                confs = np.array(list(user_item_conf[u].values()), dtype=np.float32)
                c_u = 1.0 + self.alpha * confs
                c_u_minus_1 = c_u - 1.0

                Y_u = self.item_factors[items]
                A = YTY + np.dot(Y_u.T * c_u_minus_1, Y_u) + reg_I
                b = np.dot(Y_u.T, c_u)
                self.user_factors[u] = np.linalg.solve(A, b)

            # 2. Update Item Factors
            XTX = np.dot(self.user_factors.T, self.user_factors)
            for i in range(n_items):
                users = list(item_user_conf[i].keys())
                if not users:
                    continue
                confs = np.array(list(item_user_conf[i].values()), dtype=np.float32)
                c_i = 1.0 + self.alpha * confs
                c_i_minus_1 = c_i - 1.0

                X_i = self.user_factors[users]
                A = XTX + np.dot(X_i.T * c_i_minus_1, X_i) + reg_I
                b = np.dot(X_i.T, c_i)
                self.item_factors[i] = np.linalg.solve(A, b)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-K item recommendations based on dot-product preference score."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend.")

        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            return []

        pu = self.user_factors[u_idx]
        scores = np.dot(self.item_factors, pu)
        top_indices = np.argsort(-scores)[:top_k]

        return [(self.reverse_item_map[i], float(scores[i])) for i in top_indices]


class NMFRecommender:
    r"""Non-Negative Matrix Factorization (NMF) Recommender with Multiplicative Updates.

        Decomposes the non-negative user-item rating matrix :math:`V pprox W H`
        where :math:`W \in \mathbb{R}_{+}^{M 	imes K}` (user components) and
        :math:`H \in \mathbb{R}_{+}^{K 	imes N}` (item components) using Lee & Seung (2001) updates:

        .. math::
            H \leftarrow H \odot
    rac{W^T V}{W^T W H + \epsilon}
            W \leftarrow W \odot
    rac{V H^T}{W H H^T + \epsilon}

        Parameters
        ----------
        n_factors : int, default=32
            Number of non-negative latent factors.
        n_epochs : int, default=50
            Number of multiplicative update iterations.
        eps : float, default=1e-9
            Small constant to avoid division by zero.
        seed : Optional[int], default=42
            Random seed for factor initialization.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 50,
        eps: float = 1e-9,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.eps = eps
        self.seed = seed

        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_interactions: Dict[int, Set[int]] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
        ratings: Union[List[float], np.ndarray],
    ) -> "NMFRecommender":
        """Fit NMF model on user-item non-negative ratings."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)
        r_arr = np.asarray(ratings, dtype=np.float32)

        if len(u_arr) != len(i_arr) or len(u_arr) != len(r_arr):
            raise ValueError(
                "user_ids, item_ids, and ratings must have identical lengths."
            )
        if np.any(r_arr < 0):
            raise ValueError("NMF requires non-negative rating values.")

        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_interactions = {}

        for u, i in zip(u_arr, i_arr):
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
                self.user_interactions[u_idx] = set()
            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

            self.user_interactions[self.user_map[u]].add(self.item_map[i])

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.global_mean = float(np.mean(r_arr)) if len(r_arr) > 0 else 0.0

        # Construct dense non-negative matrix V
        V: np.ndarray = np.zeros((n_users, n_items), dtype=np.float32)
        for u, i, r in zip(u_arr, i_arr, r_arr):
            V[self.user_map[u], self.item_map[i]] = float(r)

        rng = np.random.RandomState(self.seed)
        W = np.abs(rng.randn(n_users, self.n_factors).astype(np.float32)) + 0.1
        H = np.abs(rng.randn(self.n_factors, n_items).astype(np.float32)) + 0.1

        for _ in range(self.n_epochs):
            # Update H: H <- H * (W^T V) / (W^T W H + eps)
            WT_V = np.dot(W.T, V)
            WT_W_H = np.dot(np.dot(W.T, W), H) + self.eps
            H = H * (WT_V / WT_W_H)

            # Update W: W <- W * (V H^T) / (W H H^T + eps)
            V_HT = np.dot(V, H.T)
            W_H_HT = np.dot(W, np.dot(H, H.T)) + self.eps
            W = W * (V_HT / W_H_HT)

        self.user_factors = W
        self.item_factors = H
        self.is_fitted = True
        return self

    def predict(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> float:
        """Predict reconstructed rating for a given user and item."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(item_id)

        if u_idx is None or i_idx is None:
            return self.global_mean

        pred = float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))
        return pred

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
        filter_interacted: bool = True,
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend top-K items for a given user."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend.")

        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            return []

        w_u = self.user_factors[u_idx]
        scores = np.dot(w_u, self.item_factors)

        if filter_interacted and u_idx in self.user_interactions:
            for i in self.user_interactions[u_idx]:
                scores[i] = -float("inf")

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.reverse_item_map[i], float(scores[i]))
            for i in top_indices
            if scores[i] != -float("inf")
        ]
