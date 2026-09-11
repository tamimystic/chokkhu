"""Frank-Wolfe Convex Ensemble Optimization for Model Predictions & Weights.

Reference:
    Frank & Wolfe, "An algorithm for quadratic programming", Naval Research Logistics 1956.
"""

from typing import Dict, List, Union, Any, Optional, Tuple
import numpy as np


class FrankWolfeEnsemble:
    """Frank-Wolfe algorithm for finding optimal convex combination weights on the probability simplex."""

    def __init__(self, max_iters: int = 50, tol: float = 1e-6, loss: str = "mse") -> None:
        """Initialize FrankWolfeEnsemble.

        Args:
            max_iters: Maximum optimization iterations.
            tol: Convergence tolerance on Frank-Wolfe duality gap.
            loss: Loss function ('mse' or 'log_loss').
        """
        if loss not in ("mse", "log_loss"):
            raise ValueError(f"Unsupported loss '{loss}', choose 'mse' or 'log_loss'")
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.loss = loss
        self.weights_: Optional[np.ndarray] = None

    def fit(self, predictions: List[np.ndarray], y_true: np.ndarray) -> "FrankWolfeEnsemble":
        """Fit optimal mixture weights alpha on the probability simplex.

        Args:
            predictions: List of model predictions [P_1, ..., P_M], each of shape (N, ...)
            y_true: Ground truth target array (N, ...)
        """
        M = len(predictions)
        if M == 0:
            raise ValueError("predictions list cannot be empty")
        if M == 1:
            self.weights_ = np.array([1.0], dtype=np.float64)
            return self

        # Stack predictions: shape (M, N) or (M, N, C)
        P = np.stack([p.astype(np.float64) for p in predictions], axis=0)
        y = y_true.astype(np.float64)

        # Initialize uniform weights on simplex
        alpha = np.full(M, 1.0 / M, dtype=np.float64)

        for t in range(self.max_iters):
            # Current ensemble prediction: y_pred = sum_m alpha_m * P_m
            y_pred = np.tensordot(alpha, P, axes=(0, 0))

            # Compute gradient of loss with respect to ensemble prediction
            if self.loss == "mse":
                grad_ypred = 2.0 * (y_pred - y) / y.size
                # Gradient wrt alpha_m: <grad_ypred, P_m>
                grad_alpha = np.array([np.sum(grad_ypred * P[m]) for m in range(M)])
            elif self.loss == "log_loss":
                eps = 1e-12
                y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
                grad_ypred = - (y / y_pred_clipped - (1.0 - y) / (1.0 - y_pred_clipped)) / y.size
                grad_alpha = np.array([np.sum(grad_ypred * P[m]) for m in range(M)])

            # Linear subproblem: find corner i* = argmin_i grad_alpha[i]
            i_star = int(np.argmin(grad_alpha))

            # Direction: d = e_{i*} - alpha
            d = -alpha.copy()
            d[i_star] += 1.0

            # Duality gap: - <grad_alpha, d>
            gap = -np.dot(grad_alpha, d)
            if gap <= self.tol:
                break

            # Step size gamma_t = 2 / (t + 2) or exact line search for MSE
            if self.loss == "mse":
                # Exact line search for quadratic objective
                diff = P[i_star] - y_pred
                denom = np.sum(diff ** 2)
                if denom > 1e-12:
                    gamma = np.clip(np.sum((y - y_pred) * diff) / denom, 0.0, 1.0)
                else:
                    gamma = 2.0 / (t + 2.0)
            else:
                gamma = 2.0 / (t + 2.0)

            # Update alpha
            alpha = (1.0 - gamma) * alpha
            alpha[i_star] += gamma

        # Normalize to guarantee exact sum to 1
        alpha = np.maximum(alpha, 0.0)
        self.weights_ = alpha / np.sum(alpha)
        return self

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions using fitted mixture weights."""
        if self.weights_ is None:
            raise ValueError("FrankWolfeEnsemble must be fitted before predict()")
        P = np.stack([p.astype(np.float64) for p in predictions], axis=0)
        return np.tensordot(self.weights_, P, axes=(0, 0))

    def merge_weight_dicts(
        self,
        weights_list: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Convex combination of model parameter dictionaries using fitted weights."""
        if self.weights_ is None:
            raise ValueError("FrankWolfeEnsemble must be fitted before merging weights")
        merged: Dict[str, np.ndarray] = {}
        for key in weights_list[0]:
            stacked = np.stack([w[key].astype(np.float64) for w in weights_list], axis=0)
            merged[key] = np.tensordot(self.weights_, stacked, axes=(0, 0)).astype(
                weights_list[0][key].dtype
            )
        return merged
