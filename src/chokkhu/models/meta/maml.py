"""Model-Agnostic Meta-Learning (MAML) in Pure NumPy.

References:
- Finn, Abbeel, Levine (2017): "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class MAML:
    """Model-Agnostic Meta-Learning (MAML) for Few-Shot Classification and Regression.

    Optimizes initial parameters theta such that a small number of gradient steps
    on a new task leads to rapid generalization:
        min_theta sum_{T_i} L_{T_i}( theta'_i ) where theta'_i = theta - alpha * grad_theta L_{T_i}( theta )
    """

    def __init__(
        self,
        layer_sizes: List[int],
        task_type: str = "classification",
        inner_lr: float = 0.05,
        meta_lr: float = 0.01,
        inner_steps: int = 3,
        random_state: int = 42,
    ) -> None:
        self.layer_sizes = [int(s) for s in layer_sizes]
        self.task_type = task_type
        self.inner_lr = float(inner_lr)
        self.meta_lr = float(meta_lr)
        self.inner_steps = int(inner_steps)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.num_layers = len(self.layer_sizes) - 1
        self.params: Dict[str, np.ndarray] = self._init_params()

        # Meta Adam optimizer state
        self.m: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.v: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.t: int = 0

    def _init_params(self) -> Dict[str, np.ndarray]:
        """Initialize neural network weights and biases using He initialization."""
        params: Dict[str, np.ndarray] = {}
        for layer_idx in range(self.num_layers):
            in_d = self.layer_sizes[layer_idx]
            out_d = self.layer_sizes[layer_idx + 1]
            scale = np.sqrt(2.0 / in_d)
            params[f"W{layer_idx}"] = self.rng.randn(in_d, out_d) * scale
            params[f"b{layer_idx}"] = np.zeros(out_d, dtype=np.float64)
        return params

    def _forward(
        self, X: np.ndarray, params: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass through network returning output and intermediate activations."""
        x_arr = np.asarray(X, dtype=np.float64)
        activations: List[np.ndarray] = [x_arr]
        pre_activations: List[np.ndarray] = []

        curr = x_arr
        for layer_idx in range(self.num_layers):
            W = params[f"W{layer_idx}"]
            b = params[f"b{layer_idx}"]
            z = np.dot(curr, W) + b
            pre_activations.append(z)

            if layer_idx < self.num_layers - 1:
                curr = np.maximum(0.0, z)  # ReLU
                activations.append(curr)
            else:
                curr = z  # Linear output
                activations.append(curr)

        return curr, activations, pre_activations

    def compute_loss_and_gradients(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute loss and exact analytical gradients with respect to params."""
        x_arr = np.asarray(X, dtype=np.float64)
        n = x_arr.shape[0]
        if n == 0:
            return 0.0, {k: np.zeros_like(v) for k, v in params.items()}

        out, activations, _ = self._forward(x_arr, params)
        grads: Dict[str, np.ndarray] = {}

        if self.task_type == "classification":
            # Softmax cross-entropy
            shifted = out - np.max(out, axis=-1, keepdims=True)
            exp_vals = np.exp(shifted)
            probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

            y_arr = np.asarray(y, dtype=np.int64)
            log_probs = np.log(np.maximum(probs, 1e-12))
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), y_arr] = 1.0
            loss = float(-np.mean(np.sum(one_hot * log_probs, axis=1)))

            d_out = (probs - one_hot) / float(n)
        else:
            # Regression MSE loss
            y_arr = np.asarray(y, dtype=np.float64)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, np.newaxis]
            diff = out - y_arr
            loss = float(0.5 * np.mean(diff**2))
            d_out = diff / float(n)

        # Backpropagation through layers
        d_curr = d_out
        for layer_idx in reversed(range(self.num_layers)):
            h_prev = activations[layer_idx]
            W = params[f"W{layer_idx}"]

            grads[f"W{layer_idx}"] = np.dot(h_prev.T, d_curr)
            grads[f"b{layer_idx}"] = np.sum(d_curr, axis=0)

            if layer_idx > 0:
                d_prev = np.dot(d_curr, W.T)
                # ReLU derivative
                d_curr = d_prev * (h_prev > 0.0)

        return loss, grads

    def adapt(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None,
        inner_steps: Optional[int] = None,
        inner_lr: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Perform inner-loop task adaptation from meta-parameters using SGD."""
        adapted: Dict[str, np.ndarray] = (
            {k: np.copy(v) for k, v in params.items()}
            if params is not None
            else {k: np.copy(v) for k, v in self.params.items()}
        )
        steps = int(inner_steps) if inner_steps is not None else self.inner_steps
        lr = float(inner_lr) if inner_lr is not None else self.inner_lr

        for _ in range(steps):
            _, grads = self.compute_loss_and_gradients(X_support, y_support, adapted)
            for k in adapted:
                adapted[k] = adapted[k] - lr * grads[k]

        return adapted

    def meta_fit(
        self,
        tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        epochs: int = 50,
        batch_size: int = 4,
        inner_steps: Optional[int] = None,
        inner_lr: Optional[float] = None,
        meta_lr: Optional[float] = None,
        verbose: bool = False,
    ) -> List[float]:
        """Train meta-parameters theta across a collection of tasks using First-Order MAML.

        Each task is a 4-tuple: (X_support, y_support, X_query, y_query).
        """
        n_tasks = len(tasks)
        steps = int(inner_steps) if inner_steps is not None else self.inner_steps
        i_lr = float(inner_lr) if inner_lr is not None else self.inner_lr
        m_lr = float(meta_lr) if meta_lr is not None else self.meta_lr

        beta_1, beta_2, eps = 0.9, 0.999, 1e-8
        loss_history: List[float] = []

        for epoch in range(epochs):
            task_indices = self.rng.permutation(n_tasks)
            epoch_losses: List[float] = []

            for start_idx in range(0, n_tasks, batch_size):
                batch_task_idx = task_indices[start_idx : start_idx + batch_size]
                meta_grads: Dict[str, np.ndarray] = {
                    k: np.zeros_like(v) for k, v in self.params.items()
                }
                batch_loss = 0.0

                for t_idx in batch_task_idx:
                    x_supp, y_supp, x_query, y_query = tasks[t_idx]

                    # 1. Inner adaptation
                    adapted_params = self.adapt(
                        x_supp, y_supp, self.params, inner_steps=steps, inner_lr=i_lr
                    )

                    # 2. Evaluate query loss and gradients on adapted parameters (FOMAML)
                    q_loss, q_grads = self.compute_loss_and_gradients(
                        x_query, y_query, adapted_params
                    )
                    batch_loss += q_loss

                    for k in meta_grads:
                        meta_grads[k] += q_grads[k] / float(len(batch_task_idx))

                # 3. Meta-update with Adam
                self.t += 1
                for k in self.params:
                    g = np.clip(meta_grads[k], -5.0, 5.0)
                    self.m[k] = beta_1 * self.m[k] + (1.0 - beta_1) * g
                    self.v[k] = beta_2 * self.v[k] + (1.0 - beta_2) * (g**2)

                    m_hat = self.m[k] / (1.0 - beta_1**self.t)
                    v_hat = self.v[k] / (1.0 - beta_2**self.t)

                    self.params[k] -= m_lr * m_hat / (np.sqrt(v_hat) + eps)

                epoch_losses.append(batch_loss / float(len(batch_task_idx)))

            avg_epoch_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"MAML Meta-Epoch {epoch+1}/{epochs} - Query Loss: {avg_epoch_loss:.4f}"
                )

        return loss_history

    def predict(
        self, X: np.ndarray, params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Predict target labels or continuous regression values."""
        p = params if params is not None else self.params
        out, _, _ = self._forward(X, p)
        if self.task_type == "classification":
            return np.argmax(out, axis=-1)
        return out.squeeze()

    def predict_proba(
        self, X: np.ndarray, params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Predict class probability distribution."""
        p = params if params is not None else self.params
        out, _, _ = self._forward(X, p)
        shifted = out - np.max(out, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
