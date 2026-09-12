"""Deep Equilibrium Models (DEQ) & Implicit Layer Networks in Pure NumPy.

References:
- Bai, Kolter, Koltun (2019): "Deep Equilibrium Models" (NeurIPS 2019).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


class DeepEquilibriumModel:
    """Deep Equilibrium Model (DEQ) finding fixed points z* = f_theta(z*, x) via root solving.

    Uses the Implicit Function Theorem (IFT) to compute exact analytical backward gradients
    directly at the equilibrium state z* without unrolling forward computation graphs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        max_iter: int = 40,
        tol: float = 1e-4,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Equilibrium transformation parameters: z* = tanh(W_z z* + W_x x + b)
        scale_x = np.sqrt(2.0 / self.input_dim)
        scale_z = 0.95 / np.sqrt(self.hidden_dim)  # Spectral norm contraction < 1

        self.W_x: np.ndarray = self.rng.randn(self.input_dim, self.hidden_dim) * scale_x
        self.W_z: np.ndarray = (
            self.rng.randn(self.hidden_dim, self.hidden_dim) * scale_z
        )
        self.b_z: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        # Output classifier head
        scale_out = np.sqrt(2.0 / self.hidden_dim)
        self.W_out: np.ndarray = (
            self.rng.randn(self.hidden_dim, self.num_classes) * scale_out
        )
        self.b_out: np.ndarray = np.zeros(self.num_classes, dtype=np.float64)

    def _fixed_point_forward(self, X: np.ndarray) -> np.ndarray:
        """Find equilibrium state z* such that z* = tanh(z* W_z + X W_x + b_z)."""
        x_arr = np.asarray(X, dtype=np.float64)
        n = x_arr.shape[0]
        z: np.ndarray = np.zeros((n, self.hidden_dim), dtype=np.float64)
        wx = np.dot(x_arr, self.W_x) + self.b_z

        for _ in range(self.max_iter):
            z_next = np.tanh(np.dot(z, self.W_z) + wx)
            diff: float = float(np.max(np.abs(z_next - z)))
            z = z_next
            if diff < self.tol:
                break

        return z

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute equilibrium state z* and output class probabilities."""
        z_star = self._fixed_point_forward(X)
        logits = np.dot(z_star, self.W_out) + self.b_out
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return probs, z_star

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> List[float]:
        """Train DEQ using Implicit Function Theorem backward passes."""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]
        history: List[float] = []

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            epoch_loss: List[float] = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                xb = x_arr[batch_idx]
                yb = y_arr[batch_idx]
                bs = len(xb)
                if bs == 0:
                    continue

                probs, z_star = self.forward(xb)

                # Cross-entropy loss
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(bs), yb] = 1.0
                loss = float(
                    -np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-12)), axis=1))
                )
                epoch_loss.append(loss)

                # Output layer gradients
                d_logits = (probs - one_hot) / float(bs)
                d_W_out = np.dot(z_star.T, d_logits)
                d_b_out = np.sum(d_logits, axis=0)

                # Loss gradient with respect to equilibrium state z*
                d_z_star = np.dot(d_logits, self.W_out.T)

                # IFT: (I - J_f^T) lambda = d_z_star
                # Jacobian J_f = diag(1 - z*^2) W_z^T
                # Solve adjoint via fixed-point iterations: lambda_{k+1} = d_z_star + lambda_k W_z^T diag(1 - z*^2)
                dtanh = 1.0 - z_star**2
                lam = np.copy(d_z_star)
                for _ in range(20):
                    lam = d_z_star + np.dot(lam * dtanh, self.W_z.T)

                # Backprop using adjoint lambda
                d_preact = lam * dtanh
                d_W_z = np.dot(z_star.T, d_preact)
                d_W_x = np.dot(xb.T, d_preact)
                d_b_z = np.sum(d_preact, axis=0)

                # Parameter updates
                self.W_out -= self.learning_rate * np.clip(d_W_out, -5.0, 5.0)
                self.b_out -= self.learning_rate * np.clip(d_b_out, -5.0, 5.0)
                self.W_z -= self.learning_rate * np.clip(d_W_z, -5.0, 5.0)
                self.W_x -= self.learning_rate * np.clip(d_W_x, -5.0, 5.0)
                self.b_z -= self.learning_rate * np.clip(d_b_z, -5.0, 5.0)

            avg_loss = float(np.mean(epoch_loss))
            history.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"DEQ Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=-1)
