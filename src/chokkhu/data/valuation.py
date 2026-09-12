"""Data Valuation, Core-Set Selection & Instance Importance in Pure NumPy/SciPy.

References:
- Ghorbani & Zou (2019): "Data Shapley: Value of Data for Machine Learning" (ICML 2019).
- Kwon & Zou (2022): "Beta Shapley: a Value Function for Data Valuation" (NeurIPS 2022).
- Mirzasoleiman et al. (2020): "Coresets for Data-efficient Machine Learning" (ICML 2020).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist


class DataShapleyValuation:
    """Monte Carlo Truncated Data Shapley & Beta Shapley Instance Valuation.

    Computes instance-level Shapley values measuring the marginal performance contributions of individual training samples.
    """

    def __init__(
        self,
        num_permutations: int = 50,
        truncation_tolerance: float = 0.01,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.num_permutations = int(num_permutations)
        self.truncation_tolerance = float(truncation_tolerance)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def evaluate_shapley(
        self,
        utility_fn: Callable[[np.ndarray], float],
        num_samples: int,
    ) -> np.ndarray:
        """Estimate Data Shapley / Beta Shapley values for N instances using Truncated Monte Carlo.

        Args:
            utility_fn: Function mapping subset indices array S subset {0, ..., N-1} to a performance scalar V(S)
            num_samples: Total number of instances N in the dataset

        Returns:
            Shapley values array of shape (N,)
        """
        N = int(num_samples)
        shapley_values: np.ndarray = np.zeros(N, dtype=np.float64)
        sample_counts: np.ndarray = np.zeros(N, dtype=np.int64)

        # Baseline utility on full dataset and empty dataset
        all_indices: np.ndarray = np.arange(N, dtype=np.int64)
        v_full = utility_fn(all_indices)
        v_empty = utility_fn(np.array([], dtype=np.int64))

        for perm_idx in range(self.num_permutations):
            perm = self.rng.permutation(N)
            curr_subset = []
            curr_v = v_empty

            for j, instance_idx in enumerate(perm):
                # If current utility is already within tolerance of full utility, truncate remainder
                if abs(curr_v - v_full) <= self.truncation_tolerance and j > (N // 4):
                    new_v = curr_v
                else:
                    curr_subset.append(instance_idx)
                    new_v = utility_fn(np.array(curr_subset, dtype=np.int64))

                marginal_gain = new_v - curr_v

                # Beta Shapley weight: w(|S|) proportional to |S|^(alpha-1) * (N - 1 - |S|)^(beta-1)
                s_size = j
                if self.beta_alpha == 1.0 and self.beta_beta == 1.0:
                    weight = 1.0
                else:
                    # Normalized beta weight
                    weight = (max(s_size, 1) ** (self.beta_alpha - 1.0)) * (
                        max(N - 1 - s_size, 1) ** (self.beta_beta - 1.0)
                    )

                shapley_values[instance_idx] += weight * marginal_gain
                sample_counts[instance_idx] += 1
                curr_v = new_v

        # Average over permutations
        valid_mask = sample_counts > 0
        shapley_values[valid_mask] /= sample_counts[valid_mask].astype(np.float64)
        return shapley_values


class FacilityLocationCoresetSelector:
    """Submodular Facility Location Core-Set Selection (Mirzasoleiman et al., 2020).

    Selects an optimal representative subset S subset D of budget k << N maximizing submodular facility coverage:
        f(S) = sum_{i=1}^N max_{j in S} sim(x_i, x_j)
    using Lazy Greedy submodular maximization with (1 - 1/e) optimality guarantee.
    """

    def __init__(
        self,
        coreset_size: int = 20,
        similarity_metric: str = "rbf",
        gamma: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        self.coreset_size = int(coreset_size)
        self.similarity_metric = similarity_metric
        self.gamma = gamma
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def _compute_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix between all instances in X."""
        x_arr = np.asarray(X, dtype=np.float64)

        if self.similarity_metric == "rbf":
            sq_dists = cdist(x_arr, x_arr, metric="sqeuclidean")
            if self.gamma is None:
                triu_dists = sq_dists[np.triu_indices_from(sq_dists, k=1)]
                med = np.median(triu_dists) if len(triu_dists) > 0 else 1.0
                gamma = float(1.0 / (2.0 * max(med, 1e-6)))
            else:
                gamma = float(self.gamma)
            return np.exp(-gamma * sq_dists)
        elif self.similarity_metric == "cosine":
            norms = np.linalg.norm(x_arr, axis=1, keepdims=True) + 1e-12
            x_norm = x_arr / norms
            return np.maximum(0.0, np.dot(x_norm, x_norm.T))
        else:
            # Negative Euclidean distance shifted to positive
            dists = cdist(x_arr, x_arr, metric="euclidean")
            return np.max(dists) - dists

    def select(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Select core-set subset of size k using greedy submodular facility location.

        Returns:
            selected_indices: Array of shape (k,) containing chosen sample indices
            sample_weights: Array of shape (k,) with assigned sample importance weights
            coverage_score: Final submodular facility coverage objective value
        """
        x_arr = np.asarray(X, dtype=np.float64)
        N = x_arr.shape[0]
        k = min(self.coreset_size, N)

        sim_matrix = self._compute_similarity_matrix(x_arr)

        selected: List[int] = []
        max_sim_to_selected: np.ndarray = np.zeros(N, dtype=np.float64)

        # 1. Greedy iterative selection
        for step in range(k):
            best_gain = -1.0
            best_candidate = -1

            for candidate in range(N):
                if candidate in selected:
                    continue

                # Marginal gain: sum_i max(max_sim[i], sim_matrix[i, candidate]) - sum_i max_sim[i]
                new_max_sim: np.ndarray = np.maximum(
                    max_sim_to_selected, sim_matrix[:, candidate]
                )
                gain: float = float(np.sum(new_max_sim)) - float(
                    np.sum(max_sim_to_selected)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            if best_candidate == -1:
                break

            selected.append(best_candidate)
            max_sim_to_selected = np.maximum(
                max_sim_to_selected, sim_matrix[:, best_candidate]
            )

        selected_indices: np.ndarray = np.array(selected, dtype=np.int64)

        # 2. Compute coreset sample weights via Voronoi partition assignment
        # Assign each data point in N to its nearest selected exemplar
        selected_sims: np.ndarray = sim_matrix[:, selected_indices]  # [N, k]
        nearest_exemplar_idx: np.ndarray = np.argmax(
            selected_sims, axis=1
        )  # [N] in [0, k-1]

        weights: np.ndarray = np.zeros(k, dtype=np.float64)
        for j in range(k):
            weights[j] = float(np.sum(nearest_exemplar_idx == j))

        final_coverage: float = float(np.sum(max_sim_to_selected))
        return selected_indices, weights, final_coverage
