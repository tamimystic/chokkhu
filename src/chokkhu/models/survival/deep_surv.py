"""DeepSurv: Deep Cox Proportional Hazards Neural Network.

Pure NumPy implementation of Faraggi-Simon / Katzman DeepSurv architecture:
- Multi-layer perceptron parameterizing non-linear log hazard h_theta(X)
- Negative log partial likelihood loss with analytical backprop
- Baseline cumulative hazard and baseline survival function estimation
- Pure sovereign implementation with zero external deep learning frameworks
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class DeepSurv:
    r"""DeepSurv: Non-Linear Deep Feedforward Cox Model.

    Maximizes Cox partial log-likelihood using a neural network parameterizing :math:`h_\theta(X)`:

    .. math::
        L(\theta) = -\frac{1}{N_E} \sum_{i: E_i = 1} \left( h_\theta(X_i) -
        \ln \sum_{j: T_j \ge T_i} \exp(h_\theta(X_j)) \right) + \frac{\lambda}{2} \|\Theta\|_2^2

    Parameters
    ----------
    hidden_layers : List[int], default=[32, 16]
        Sizes of intermediate dense hidden layers.
    activation : str, default="relu"
        Activation function ("relu", "tanh", "gelu").
    lr : float, default=0.01
        Learning rate.
    weight_decay : float, default=1e-4
        L2 regularization penalty.
    n_epochs : int, default=100
        Number of training epochs.
    seed : Optional[int], default=42
        Random seed for reproducible parameter initialization.
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        self.hidden_layers = hidden_layers or [32, 16]
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.seed = seed

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.m_w: List[np.ndarray] = []
        self.v_w: List[np.ndarray] = []
        self.m_b: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []

        self.baseline_timeline_: Optional[np.ndarray] = None
        self.baseline_cumulative_hazard_: Optional[np.ndarray] = None
        self.baseline_survival_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.is_fitted: bool = False

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, z)
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "gelu":
            return (
                0.5
                * z
                * (
                    1.0
                    + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * np.power(z, 3)))
                )
            )
        return z

    def _act_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0.0).astype(np.float64)
        elif self.activation == "tanh":
            return 1.0 - np.square(a)
        elif self.activation == "gelu":
            # Numerical approximation
            eps = 1e-5
            return (self._act(z + eps) - self._act(z - eps)) / (2.0 * eps)
        return np.ones_like(z)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        current = X
        n_layers = len(self.weights)

        for l_idx in range(n_layers):
            z = np.dot(current, self.weights[l_idx]) + self.biases[l_idx]
            zs.append(z)
            if l_idx == n_layers - 1:
                # Linear output for risk score
                current = z
            else:
                current = self._act(z)
            activations.append(current)

        return activations, zs

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> "DeepSurv":
        """Train DeepSurv model via Adam optimization."""
        X_arr = np.asarray(X, dtype=np.float64)
        T_arr = np.asarray(durations, dtype=np.float64).ravel()
        if events is None:
            E_arr = np.ones_like(T_arr, dtype=np.int32)
        else:
            E_arr = np.asarray(events, dtype=np.int32).ravel()

        if len(X_arr) != len(T_arr):
            raise ValueError("X and durations must have identical length.")

        n_samples, n_features = X_arr.shape
        n_events = float(np.sum(E_arr == 1))
        if n_events == 0:
            raise ValueError("At least one observed event required.")

        # Sort dataset by duration descending for efficient cumulative sum risk set computation
        sort_idx = np.argsort(-T_arr)
        X_s = X_arr[sort_idx]
        E_s = E_arr[sort_idx]

        # Initialize network layers
        rng = np.random.RandomState(self.seed)
        layer_dims = [n_features] + self.hidden_layers + [1]

        self.weights = []
        self.biases = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)
            b = np.zeros(fan_out, dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.loss_history_ = []

        for epoch in range(1, self.n_epochs + 1):
            activations, zs = self._forward(X_s)
            h = activations[-1].ravel()  # (N,)

            # Stable risk computation
            exp_h = np.exp(np.clip(h, -30.0, 30.0))
            # Cumulative risk sum from longest duration to shortest (since sorted descending)
            cum_risk = np.cumsum(exp_h)  # (N,)

            # Loss: - 1/N_E * sum_{i: E_i=1} (h_i - log(cum_risk_i))
            event_mask = E_s == 1
            loss_val: float = float(
                np.sum(h[event_mask] - np.log(np.maximum(cum_risk[event_mask], 1e-12)))
            )
            loss_events: float = float(-loss_val / n_events)
            l2_reg = (
                0.5
                * self.weight_decay
                * sum(float(np.sum(w * w)) for w in self.weights)
            )
            total_loss = float(loss_events + l2_reg)
            self.loss_history_.append(total_loss)

            # Exact analytical gradient dL/dh:
            # For each sample k:
            # dL/dh_k = -1/N_E * (E_k - exp(h_k) * sum_{i: E_i=1, T_i <= T_k} 1 / cum_risk_i)
            # Since sorted descending: T_i <= T_k means i >= k in sorted order!
            inv_cum_risk = np.zeros(n_samples, dtype=np.float64)
            inv_cum_risk[event_mask] = 1.0 / np.maximum(cum_risk[event_mask], 1e-12)
            # Reverse cumulative sum gives sum_{i >= k, E_i=1} 1 / cum_risk_i
            sum_inv_risk = np.cumsum(inv_cum_risk[::-1])[::-1]

            grad_h = -(E_s - exp_h * sum_inv_risk) / n_events
            delta = grad_h[:, None]  # (N, 1)

            # Backprop through layers
            n_layers = len(self.weights)
            for l_idx in reversed(range(n_layers)):
                a_prev = activations[l_idx]
                grad_w = (
                    np.dot(a_prev.T, delta) + self.weight_decay * self.weights[l_idx]
                )
                grad_b = np.sum(delta, axis=0)

                if l_idx > 0:
                    delta = np.dot(delta, self.weights[l_idx].T) * self._act_grad(
                        zs[l_idx - 1], activations[l_idx]
                    )

                # Adam optimizer update
                self.m_w[l_idx] = beta1 * self.m_w[l_idx] + (1.0 - beta1) * grad_w
                self.v_w[l_idx] = beta2 * self.v_w[l_idx] + (1.0 - beta2) * np.square(
                    grad_w
                )
                m_w_hat = self.m_w[l_idx] / (1.0 - beta1**epoch)
                v_w_hat = self.v_w[l_idx] / (1.0 - beta2**epoch)
                self.weights[l_idx] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

                self.m_b[l_idx] = beta1 * self.m_b[l_idx] + (1.0 - beta1) * grad_b
                self.v_b[l_idx] = beta2 * self.v_b[l_idx] + (1.0 - beta2) * np.square(
                    grad_b
                )
                m_b_hat = self.m_b[l_idx] / (1.0 - beta1**epoch)
                v_b_hat = self.v_b[l_idx] / (1.0 - beta2**epoch)
                self.biases[l_idx] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        self.is_fitted = True

        # Compute baseline cumulative hazard (Breslow)
        h_all = self.predict_risk(X_arr)
        exp_h_all = np.exp(np.clip(h_all, -30.0, 30.0))

        unique_times = np.sort(np.unique(T_arr[E_arr == 1]))
        timeline_list = (
            [0.0] + list(unique_times) if unique_times[0] > 0.0 else list(unique_times)
        )
        timeline = np.array(timeline_list, dtype=np.float64)

        base_cum_hazard: np.ndarray = np.zeros(len(timeline), dtype=np.float64)
        cum_h = 0.0

        for i, t in enumerate(timeline):
            if t == 0.0 and (len(unique_times) == 0 or unique_times[0] > 0.0):
                continue
            d_m = float(np.sum((T_arr == t) & (E_arr == 1)))
            denom = float(np.sum(exp_h_all[T_arr >= t]))
            h_m = (d_m / denom) if denom > 0 else 0.0
            base_hazard = h_m
            cum_h += base_hazard
            base_cum_hazard[i] = cum_h

        self.baseline_timeline_ = timeline
        self.baseline_cumulative_hazard_ = base_cum_hazard
        self.baseline_survival_ = np.exp(-base_cum_hazard)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict linear risk score h_theta(X)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_risk.")
        X_arr = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward(X_arr)
        return activations[-1].ravel()

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        """Predict exponentiated hazard ratio exp(h_theta(X))."""
        risk = self.predict_risk(X)
        return np.exp(np.clip(risk, -30.0, 30.0))

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[Union[float, np.ndarray, List[float]]] = None,
    ) -> np.ndarray:
        """Predict survival curve S(t | X) = S_0(t)^exp(h_theta(X))."""
        if (
            not self.is_fitted
            or self.baseline_survival_ is None
            or self.baseline_timeline_ is None
        ):
            raise RuntimeError("Model must be fitted before predict_survival_function.")

        X_arr = np.asarray(X, dtype=np.float64)
        hazard_mult = self.predict_partial_hazard(X_arr)

        if times is None:
            base_s = self.baseline_survival_
            return np.power(base_s[None, :], hazard_mult[:, None])
        else:
            t_query = np.asarray(times, dtype=np.float64)
            scalar_time = t_query.ndim == 0
            t_query = np.atleast_1d(t_query)

            indices = (
                np.searchsorted(self.baseline_timeline_, t_query, side="right") - 1
            )
            indices = np.clip(indices, 0, len(self.baseline_survival_) - 1)
            base_s = self.baseline_survival_[indices]
            base_s[t_query < 0.0] = 1.0

            res = np.power(base_s[None, :], hazard_mult[:, None])
            return res[:, 0] if scalar_time and res.shape[1] == 1 else res
