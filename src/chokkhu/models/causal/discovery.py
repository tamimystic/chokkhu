"""Causal Structure Discovery and Directed Acyclic Graph (DAG) Learning.

Formulated from first principles using continuous non-combinatorial optimization
(NOTEARS matrix exponential trace) and constraint-based PC algorithm in pure NumPy and SciPy.
"""

import numpy as np
from scipy import linalg, optimize, stats
from typing import Optional, Tuple, List, Dict, Set


class NOTEARSCausalDiscovery:
    """Non-combinatorial Optimization via Trace Exponential for Structure Learning (NOTEARS).

    Discovers weighted causal DAG adjacency matrix W from observational continuous data
    by formulating the combinatorial acyclicity constraint as a smooth algebraic equality:
    h(W) = tr(exp(W * W)) - d = 0.

    Parameters
    ----------
    lambda1 : float, default=0.1
        L1 regularization penalty coefficient for sparsity.
    loss_type : str, default="l2"
        Loss formulation: "l2" (least squares for linear SEM) or "logistic".
    max_iter : int, default=100
        Maximum Augmented Lagrangian outer iterations.
    h_tol : float, default=1e-8
        Acyclicity constraint tolerance for convergence.
    rho_max : float, default=1e16
        Maximum penalty parameter rho in Augmented Lagrangian.
    w_threshold : float, default=0.3
        Edge weight threshold for post-processing pruning.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        loss_type: str = "l2",
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ) -> None:
        self.lambda1 = float(lambda1)
        self.loss_type = loss_type
        self.max_iter = int(max_iter)
        self.h_tol = float(h_tol)
        self.rho_max = float(rho_max)
        self.w_threshold = float(w_threshold)
        self.W_est_: Optional[np.ndarray] = None
        self.adjacency_matrix_: Optional[np.ndarray] = None

    @staticmethod
    def _h(W: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute acyclicity constraint h(W) = tr(expm(W * W)) - d and its gradient."""
        d = W.shape[0]
        M = W * W
        E = linalg.expm(M)
        h = float(np.trace(E) - d)
        grad_h = 2.0 * E.T * W
        return h, grad_h

    def _loss(self, W: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute least squares loss f(W) = 1/(2n) * ||X - X W||_F^2 and gradient."""
        n, d = X.shape
        R = X - np.dot(X, W)
        loss = 0.5 / n * float(np.sum(R**2))
        grad = -1.0 / n * np.dot(X.T, R)
        return loss, grad

    def fit(self, X: np.ndarray) -> "NOTEARSCausalDiscovery":
        """Fit causal DAG structure from observational data matrix X."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        w_est = np.zeros((d, d), dtype=float)
        rho = 1.0
        alpha = 0.0
        h_curr = np.inf

        bnds = [
            (0.0, 0.0) if i == j else (None, None) for i in range(d) for j in range(d)
        ]

        for it in range(self.max_iter):
            while rho < self.rho_max:

                def objective(w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
                    W = w_flat.reshape(d, d)
                    loss_val, loss_grad = self._loss(W, X)
                    h_val, h_grad = self._h(W)

                    l1_penalty = self.lambda1 * np.sum(np.abs(W))
                    l1_grad = self.lambda1 * np.sign(W)

                    obj = loss_val + l1_penalty + alpha * h_val + 0.5 * rho * (h_val**2)
                    grad = loss_grad + l1_grad + (alpha + rho * h_val) * h_grad
                    return float(obj), grad.flatten()

                res = optimize.minimize(
                    fun=lambda w: objective(w)[0],
                    x0=w_est.flatten(),
                    jac=lambda w: objective(w)[1],
                    bounds=bnds,
                    method="L-BFGS-B",
                    options={"maxiter": 150, "ftol": 1e-8},
                )

                w_new = res.x.reshape(d, d)
                h_new, _ = self._h(w_new)

                if h_new > 0.25 * h_curr:
                    rho = min(10.0 * rho, self.rho_max)
                else:
                    break

            w_est = w_new
            h_curr = h_new
            alpha += rho * h_curr

            if h_curr <= self.h_tol or rho >= self.rho_max:
                break

        # Thresholding
        w_est[np.abs(w_est) < self.w_threshold] = 0.0
        np.fill_diagonal(w_est, 0.0)

        self.W_est_ = w_est
        self.adjacency_matrix_ = (np.abs(w_est) > 0).astype(int)
        return self

    def is_dag(self, W: Optional[np.ndarray] = None) -> bool:
        """Verify whether adjacency matrix W is strictly a Directed Acyclic Graph."""
        mat = self.adjacency_matrix_ if W is None else (np.abs(W) > 0).astype(int)
        if mat is None:
            return False

        in_degree = np.sum(mat, axis=0)
        zero_in = [i for i, deg in enumerate(in_degree) if deg == 0]
        visited_count = 0

        while zero_in:
            node = zero_in.pop(0)
            visited_count += 1
            for neighbor in range(mat.shape[1]):
                if mat[node, neighbor] > 0:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        zero_in.append(neighbor)

        return visited_count == mat.shape[0]


class PCAlgorithm:
    """Peter-Clark (PC) Constraint-Based Causal Discovery Algorithm.

    Reconstructs the Completed Partially Directed Acyclic Graph (CPDAG) equivalence class
    using conditional independence tests and orientation rules.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.max_cond_set_size = max_cond_set_size
        self.adjacency_matrix_: Optional[np.ndarray] = None
        self.sepset_: Dict[Tuple[int, int], Set[int]] = {}

    @staticmethod
    def _partial_corr(X: np.ndarray, i: int, j: int, cond: List[int]) -> float:
        """Compute sample partial correlation between X_i and X_j given conditioning set."""
        if len(cond) == 0:
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            return float(0.0 if np.isnan(r) else r)

        idx = [i, j] + list(cond)
        sub_cov = np.cov(X[:, idx], rowvar=False)
        try:
            sub_prec = linalg.pinv(sub_cov)
            r = -sub_prec[0, 1] / np.sqrt(
                np.abs(sub_prec[0, 0] * sub_prec[1, 1]) + 1e-12
            )
            return float(np.clip(r, -0.999999, 0.999999))
        except Exception:
            return 0.0

    def _is_conditionally_independent(
        self, X: np.ndarray, i: int, j: int, cond: List[int]
    ) -> bool:
        """Test conditional independence via Fisher's z-transformation."""
        n = X.shape[0]
        k = len(cond)
        if n - k - 3 <= 0:
            return False

        r = self._partial_corr(X, i, j, cond)
        z = 0.5 * np.log((1.0 + r) / max(1e-12, (1.0 - r)))
        stat = np.sqrt(n - k - 3) * np.abs(z)
        crit = stats.norm.ppf(1.0 - self.alpha / 2.0)
        return bool(stat <= crit)

    def fit(self, X: np.ndarray) -> "PCAlgorithm":
        """Run PC skeleton discovery and orientation on continuous observational data matrix."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        adj = np.ones((d, d), dtype=int) - np.eye(d, dtype=int)
        self.sepset_ = {}

        max_k = (
            d - 2
            if self.max_cond_set_size is None
            else min(d - 2, self.max_cond_set_size)
        )

        from itertools import combinations

        for k in range(max_k + 1):
            changed = False
            for i in range(d):
                for j in range(i + 1, d):
                    if adj[i, j] == 1:
                        neighbors = [
                            nb for nb in range(d) if nb != j and adj[i, nb] == 1
                        ]
                        if len(neighbors) >= k:
                            for cond_set in combinations(neighbors, k):
                                if self._is_conditionally_independent(
                                    X, i, j, list(cond_set)
                                ):
                                    adj[i, j] = 0
                                    adj[j, i] = 0
                                    self.sepset_[(i, j)] = set(cond_set)
                                    self.sepset_[(j, i)] = set(cond_set)
                                    changed = True
                                    break
            if not changed and k > 0:
                break

        # V-Structure / Collider Orientation
        for i in range(d):
            for j in range(i + 1, d):
                if adj[i, j] == 0:
                    sep = self.sepset_.get((i, j), set())
                    for k in range(d):
                        if k != i and k != j and adj[i, k] == 1 and adj[j, k] == 1:
                            if k not in sep:
                                adj[k, i] = 0
                                adj[k, j] = 0

        self.adjacency_matrix_ = adj
        return self

    def get_edges(self) -> List[Tuple[int, int, str]]:
        """Extract edges as (u, v, direction_str) where direction is '->' or '--'."""
        if self.adjacency_matrix_ is None:
            return []
        d = self.adjacency_matrix_.shape[0]
        edges = []
        for i in range(d):
            for j in range(d):
                if self.adjacency_matrix_[i, j] == 1:
                    if self.adjacency_matrix_[j, i] == 1:
                        if i < j:
                            edges.append((i, j, "--"))
                    else:
                        edges.append((i, j, "->"))
        return edges
