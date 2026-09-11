"""Online Optimization, FTRL-Proximal Regret Minimization, and Hedge Algorithm.

Formulated from first principles using coordinate-adaptive L1/L2 soft-thresholded
proximal gradient descent and exponential weighting minimax regret minimization in pure NumPy.
"""

from typing import Optional, Tuple

import numpy as np


class FollowTheRegularizedLeader:
    r"""Follow-The-Regularized-Leader (FTRL-Proximal) Sparse Online Optimizer.

        Implements the McMahan et al. (Google Ad Click-Through-Rate Prediction) algorithm:
        w_{t+1, i} = -	ext{sign}(z_i)
    rac{\max(0, |z_i| - \lambda_1)}{
    rac{eta + \sqrt{n_i}}{lpha} + \lambda_2}

        Parameters
        ----------
        alpha : float, default=0.1
            Per-coordinate learning rate scale alpha.
        beta : float, default=1.0
            Smoothing parameter for learning rate denominator.
        lambda1 : float, default=0.1
            L1 regularization penalty coefficient (sparsity inducing).
        lambda2 : float, default=1.0
            L2 regularization penalty coefficient.
        dim : Optional[int], default=None
            Feature dimension. If None, dynamically initialized on first sample.
        loss : str, default="logistic"
            Loss function: "logistic" (binary classification) or "squared" (regression).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        lambda1: float = 0.1,
        lambda2: float = 1.0,
        dim: Optional[int] = None,
        loss: str = "logistic",
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.dim = dim
        self.loss = loss

        self.n: Optional[np.ndarray] = None  # Sum of squared gradients
        self.z: Optional[np.ndarray] = None  # Accumulated proximal gradient state
        self.w: Optional[np.ndarray] = None  # Active weights

        if dim is not None:
            self._init_dim(dim)

    def _init_dim(self, d: int) -> None:
        self.dim = int(d)
        self.n = np.zeros(self.dim, dtype=float)
        self.z = np.zeros(self.dim, dtype=float)
        self.w = np.zeros(self.dim, dtype=float)

    def _update_weight_coordinate(self, i: int) -> float:
        """Compute coordinate weight w_i from current state (z_i, n_i)."""
        if self.z is None or self.n is None:
            return 0.0

        z_i = self.z[i]
        n_i = self.n[i]

        if abs(z_i) <= self.lambda1:
            return 0.0

        sign_z = 1.0 if z_i > 0 else -1.0
        denom = (self.beta + np.sqrt(n_i)) / self.alpha + self.lambda2
        num = abs(z_i) - self.lambda1
        return float(-sign_z * (num / denom))

    def get_weights(self) -> np.ndarray:
        """Return current dense weight vector."""
        if self.w is None:
            return np.array([])
        for i in range(len(self.w)):
            self.w[i] = self._update_weight_coordinate(i)
        return self.w.copy()

    def predict_one(self, x: np.ndarray) -> float:
        """Predict scalar output for a single feature vector x."""
        x_arr = np.asarray(x, dtype=float).flatten()
        if self.dim is None:
            self._init_dim(len(x_arr))

        assert self.w is not None
        # Lazy coordinate weight update for active features
        w_dot_x: float = 0.0
        for i, xi in enumerate(x_arr):
            if abs(xi) > 1e-12:
                w_i = self._update_weight_coordinate(i)
                self.w[i] = w_i
                w_dot_x += float(w_i * xi)

        if self.loss == "logistic":
            # Sigmoid with numerical clip
            prob: float = float(1.0 / (1.0 + np.exp(np.clip(-w_dot_x, -50.0, 50.0))))
            return prob
        return float(w_dot_x)

    def update_one(self, x: np.ndarray, y: float) -> float:
        """Update optimizer state given a single sample (x, y).

        Returns
        -------
        loss : float
            Instantaneous sample loss before update.
        """
        x_arr = np.asarray(x, dtype=float).flatten()
        if self.dim is None:
            self._init_dim(len(x_arr))

        assert self.n is not None and self.z is not None and self.w is not None

        p = self.predict_one(x_arr)
        target = float(y)

        if self.loss == "logistic":
            grad_scalar = p - target  # d/dw log-loss
            loss_val = float(
                -target * np.log(max(p, 1e-12))
                - (1.0 - target) * np.log(max(1.0 - p, 1e-12))
            )
        else:
            grad_scalar = p - target
            loss_val = float(0.5 * (p - target) ** 2)

        # Update (n_i, z_i) for active features
        for i, xi in enumerate(x_arr):
            if abs(xi) > 1e-12:
                g_i = grad_scalar * xi
                n_old = self.n[i]
                n_new = n_old + g_i**2
                self.n[i] = n_new

                sigma_i = (np.sqrt(n_new) - np.sqrt(n_old)) / self.alpha
                w_i = self.w[i]
                self.z[i] += g_i - sigma_i * w_i

        return loss_val

    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 1
    ) -> "FollowTheRegularizedLeader":
        """Stream through dataset for specified number of epochs."""
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float)
        if self.dim is None:
            self._init_dim(X_mat.shape[1])

        for _ in range(epochs):
            for idx in range(X_mat.shape[0]):
                self.update_one(X_mat[idx], y_vec[idx])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for a batch of features."""
        X_mat = np.asarray(X, dtype=float)
        preds = [self.predict_one(row) for row in X_mat]
        return np.array(preds)


class HedgeAlgorithm:
    r"""Hedge Multi-Expert Online Decision Engine (Freund & Schapire).

        Maintains randomized strategy over K experts minimizing cumulative regret:
        w_{t+1, i} =
    rac{w_{t, i} \cdot e^{-\eta l_{t, i}}}{\sum_j w_{t, j} \cdot e^{-\eta l_{t, j}}}

        Parameters
        ----------
        num_experts : int
            Number of decision experts K.
        eta : Optional[float], default=None
            Learning rate parameter. If None, optimal minimax rate eta = sqrt(8 ln(K) / T) is used.
        time_horizon : int, default=100
            Anticipated decision rounds T for automatic eta calibration.
    """

    def __init__(
        self,
        num_experts: int,
        eta: Optional[float] = None,
        time_horizon: int = 100,
    ) -> None:
        self.num_experts = max(2, int(num_experts))
        self.time_horizon = max(1, int(time_horizon))

        if eta is None:
            self.eta = float(
                np.sqrt(8.0 * np.log(self.num_experts) / self.time_horizon)
            )
        else:
            self.eta = float(eta)

        self.weights: np.ndarray = (
            np.ones(self.num_experts, dtype=float) / self.num_experts
        )
        self.cumulative_losses: np.ndarray = np.zeros(self.num_experts, dtype=float)
        self.algorithm_cumulative_loss: float = 0.0

    def predict_distribution(self) -> np.ndarray:
        """Return current probability distribution over experts."""
        return self.weights.copy()

    def aggregate_prediction(self, expert_predictions: np.ndarray) -> float:
        """Compute convex combination of expert advice."""
        preds = np.asarray(expert_predictions, dtype=float)
        return float(np.dot(self.weights, preds))

    def update(self, expert_losses: np.ndarray) -> Tuple[float, float]:
        """Update expert weights given observed round losses l_t in [0, 1].

        Parameters
        ----------
        expert_losses : np.ndarray of shape (num_experts,)
            Instantaneous losses suffered by each expert in current round.

        Returns
        -------
        round_algorithm_loss : float
            Expected loss suffered by the Hedge mixture.
        cumulative_regret : float
            Difference between algorithm cumulative loss and best single expert.
        """
        losses = np.asarray(expert_losses, dtype=float)
        round_loss = float(np.dot(self.weights, losses))

        self.algorithm_cumulative_loss += round_loss
        self.cumulative_losses += losses

        # Multiplicative weight update with numerical stabilization
        log_weights = np.log(np.maximum(self.weights, 1e-12)) - self.eta * losses
        log_weights -= np.max(log_weights)
        new_weights = np.exp(log_weights)
        self.weights = new_weights / np.sum(new_weights)

        best_expert_loss = float(np.min(self.cumulative_losses))
        regret = self.algorithm_cumulative_loss - best_expert_loss
        return round_loss, regret
