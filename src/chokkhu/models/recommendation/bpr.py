"""Bayesian Personalized Ranking (BPR-MF) Recommender System.

Pure NumPy implementation of Rendle et al. (2009) pairwise ranking optimization
for implicit feedback recommendation.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class BayesianPersonalizedRanking:
    r"""Bayesian Personalized Ranking Matrix Factorization (BPR-MF).

        Optimizes a pairwise ranking objective function via stochastic gradient descent
        with bootstrap sampling of negative item interactions:

        .. math::
            \mathcal{L}_{	ext{BPR}} = \sum_{(u, i, j) \in D_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})
            -
    rac{\lambda}{2} (\|p_u\|^2 + \|q_i\|^2 + \|q_j\|^2 + b_i^2 + b_j^2)

        where :math:`\hat{x}_{ui} = p_u^T q_i + b_i`.

        Parameters
        ----------
        n_factors : int, default=32
            Dimensionality of user and item latent embedding vectors.
        n_epochs : int, default=30
            Number of full training epochs (each epoch samples n_interactions triplets).
        lr : float, default=0.05
            Learning rate for SGD parameter updates.
        reg : float, default=0.01
            L2 regularization coefficient (:math:`\lambda`).
        seed : Optional[int], default=42
            Random seed for reproducibility and weight initialization.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 30,
        lr: float = 0.05,
        reg: float = 0.01,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None

        self.user_pos_items: Dict[int, List[int]] = {}
        self.user_pos_sets: Dict[int, Set[int]] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
    ) -> "BayesianPersonalizedRanking":
        """Train BPR-MF on implicit user-item positive interactions."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)

        if len(u_arr) != len(i_arr):
            raise ValueError("user_ids and item_ids must have identical lengths.")
        if len(u_arr) == 0:
            raise ValueError("Interaction dataset cannot be empty.")

        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_pos_items = {}
        self.user_pos_sets = {}

        for u, i in zip(u_arr, i_arr):
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
                self.user_pos_items[u_idx] = []
                self.user_pos_sets[u_idx] = set()

            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

            u_idx = self.user_map[u]
            i_idx = self.item_map[i]
            if i_idx not in self.user_pos_sets[u_idx]:
                self.user_pos_items[u_idx].append(i_idx)
                self.user_pos_sets[u_idx].add(i_idx)

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        rng = np.random.RandomState(self.seed)
        self.user_factors = (rng.randn(n_users, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.item_factors = (rng.randn(n_items, self.n_factors) * 0.1).astype(
            np.float32
        )
        self.item_bias = np.zeros(n_items, dtype=np.float32)

        active_users = [u for u in range(n_users) if len(self.user_pos_items[u]) > 0]
        n_samples = len(u_arr)

        for _ in range(self.n_epochs):
            for _ in range(n_samples):
                u = active_users[rng.randint(0, len(active_users))]
                pos_list = self.user_pos_items[u]
                pos_set = self.user_pos_sets[u]

                i = pos_list[rng.randint(0, len(pos_list))]

                j = rng.randint(0, n_items)
                while j in pos_set and len(pos_set) < n_items:
                    j = rng.randint(0, n_items)

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                qj = self.item_factors[j]

                x_ui = np.dot(pu, qi) + self.item_bias[i]
                x_uj = np.dot(pu, qj) + self.item_bias[j]
                x_uij = x_ui - x_uj

                x_uij_clipped = np.clip(x_uij, -30.0, 30.0)
                coeff = 1.0 / (1.0 + np.exp(x_uij_clipped))

                self.user_factors[u] += self.lr * (coeff * (qi - qj) - self.reg * pu)
                self.item_factors[i] += self.lr * (coeff * pu - self.reg * qi)
                self.item_factors[j] += self.lr * (-coeff * pu - self.reg * qj)
                self.item_bias[i] += self.lr * (coeff - self.reg * self.item_bias[i])
                self.item_bias[j] += self.lr * (-coeff - self.reg * self.item_bias[j])

        self.is_fitted = True
        return self

    def predict_score(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> float:
        """Compute predicted ranking score for a user-item pair."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling predict_score.")

        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(item_id)

        if u_idx is None and i_idx is None:
            return 0.0
        if u_idx is None:
            return float(self.item_bias[i_idx]) if self.item_bias is not None else 0.0
        if i_idx is None:
            return 0.0

        score = float(
            np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
            + (self.item_bias[i_idx] if self.item_bias is not None else 0.0)
        )
        return score

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
        filter_interacted: bool = True,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-K recommended items for a user."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend.")

        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            return []

        pu = self.user_factors[u_idx]
        scores = np.dot(self.item_factors, pu)
        if self.item_bias is not None:
            scores += self.item_bias

        if filter_interacted and u_idx in self.user_pos_sets:
            for i in self.user_pos_sets[u_idx]:
                scores[i] = -float("inf")

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.reverse_item_map[i], float(scores[i]))
            for i in top_indices
            if scores[i] != -float("inf")
        ]

    def __repr__(self) -> str:
        return (
            f"BayesianPersonalizedRanking(n_factors={self.n_factors}, "
            f"lr={self.lr}, reg={self.reg}, users={len(self.user_map)}, "
            f"items={len(self.item_map)})"
        )
