"""Variational Quantum Classifier (VQC) and Quantum Kernel Methods."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from .circuit import QuantumCircuit


class VariationalQuantumClassifier:
    """Variational Quantum Classifier (VQC / Quantum Neural Network).

    Features:
    - Angle Embedding feature encoding.
    - Parameterized Hardware-Efficient Ansatz (Ry/Rz rotations + entangling CNOT layers).
    - Analytical exact gradients via the Parameter Shift Rule:
      d<Z>/d(theta) = [<Z>(theta + pi/2) - <Z>(theta - pi/2)] / 2

    Parameters
    ----------
    n_qubits : int
        Number of qubits (matches feature dimension or compressed features).
    n_layers : int
        Number of variational ansatz layers.
    lr : float
        Learning rate for gradient descent.
    random_state : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        lr: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.lr = float(lr)
        self.rng = np.random.default_rng(random_state)

        # Number of parameters: n_layers * n_qubits * 2 (Ry and Rz per qubit per layer)
        self.num_params = self.n_layers * self.n_qubits * 2
        self.weights = self.rng.uniform(-np.pi, np.pi, size=self.num_params)
        self.bias = 0.0

    def _execute_circuit(self, x: np.ndarray, params: np.ndarray) -> float:
        """Run circuit for a single input vector x with given weights."""
        qc = QuantumCircuit(n_qubits=self.n_qubits)

        # 1. Feature encoding via Angle Embedding (Ry)
        for i in range(min(self.n_qubits, len(x))):
            qc.ry(float(x[i]), i)

        # 2. Variational layers
        p_idx = 0
        for _ in range(self.n_layers):
            # Rotations
            for q in range(self.n_qubits):
                qc.ry(float(params[p_idx]), q)
                p_idx += 1
                qc.rz(float(params[p_idx]), q)
                p_idx += 1

            # Entanglement (ring of CNOTs)
            if self.n_qubits > 1:
                for q in range(self.n_qubits):
                    next_q = (q + 1) % self.n_qubits
                    qc.cnot(q, next_q)

        # 3. Measure expectation value of Pauli-Z on qubit 0
        return qc.expectation_z(0)

    def forward_single(self, x: np.ndarray) -> float:
        """Predict scalar logit for single sample x: f(x) = <Z> + bias."""
        exp_z = self._execute_circuit(x, self.weights)
        return exp_z + self.bias

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Predict logits for a batch of samples."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array([self.forward_single(x) for x in X_arr])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        logits = self.forward(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        logits = self.forward(X)
        return (logits >= 0.0).astype(int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 16,
    ) -> VariationalQuantumClassifier:
        """Train variational parameters using Parameter Shift Rule gradients."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).flatten()
        # Ensure binary labels in {-1, +1} or {0, 1}
        # Standardize target to {0, 1}
        unique_y = np.unique(y_arr)
        if set(unique_y).issubset({-1.0, 1.0}):
            y_target = (y_arr + 1.0) / 2.0
        else:
            y_target = y_arr

        n = len(X_arr)
        shift = np.pi / 2.0

        for _ in range(epochs):
            indices = self.rng.permutation(n)

            for i in range(0, n, batch_size):
                b_idx = indices[i : min(i + batch_size, n)]
                X_b = X_arr[b_idx]
                y_b = y_target[b_idx]

                grad_weights = np.zeros_like(self.weights)
                grad_bias = 0.0

                for x_sample, y_sample in zip(X_b, y_b):
                    pred_prob = 1.0 / (
                        1.0
                        + math.exp(-np.clip(self.forward_single(x_sample), -20.0, 20.0))
                    )
                    error = pred_prob - float(y_sample)

                    # Parameter Shift Rule for each parameter
                    for p in range(self.num_params):
                        # Theta + pi/2
                        w_plus = self.weights.copy()
                        w_plus[p] += shift
                        f_plus = self._execute_circuit(x_sample, w_plus)

                        # Theta - pi/2
                        w_minus = self.weights.copy()
                        w_minus[p] -= shift
                        f_minus = self._execute_circuit(x_sample, w_minus)

                        df_dp = (f_plus - f_minus) / 2.0
                        grad_weights[p] += error * df_dp

                    grad_bias += error

                # Gradient descent step
                b_len = float(len(X_b))
                self.weights -= self.lr * (grad_weights / b_len)
                self.bias -= self.lr * (grad_bias / b_len)

        return self


class QuantumKernel:
    """Quantum Kernel Matrix Estimator for Quantum SVM / Kernel Ridge.

    Computes transition probability fidelity K(x, x') = |<psi(x)|psi(x')>|^2.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for state embedding.
    """

    def __init__(self, n_qubits: int = 2) -> None:
        self.n_qubits = int(n_qubits)

    def _state(self, x: np.ndarray) -> np.ndarray:
        """Create encoded quantum state for feature vector x."""
        qc = QuantumCircuit(n_qubits=self.n_qubits)
        for i in range(min(self.n_qubits, len(x))):
            qc.h(i)
            qc.rz(float(x[i]), i)
        return qc.state

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute scalar quantum kernel similarity between two vectors."""
        psi1 = self._state(x1)
        psi2 = self._state(x2)
        # Inner product fidelity |<psi1|psi2>|^2
        inner_prod = np.vdot(psi1, psi2)
        return float(np.abs(inner_prod) ** 2)

    def compute_matrix(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute full quantum Gram matrix K(X1, X2)."""
        X1_arr = np.asarray(X1, dtype=np.float64)
        if X2 is None:
            n = len(X1_arr)
            K_self: np.ndarray = np.zeros((n, n), dtype=np.float64)
            states = [self._state(x) for x in X1_arr]
            for i in range(n):
                K_self[i, i] = 1.0
                for j in range(i + 1, n):
                    fidelity = float(np.abs(np.vdot(states[i], states[j])) ** 2)
                    K_self[i, j] = fidelity
                    K_self[j, i] = fidelity
            return K_self
        else:
            X2_arr = np.asarray(X2, dtype=np.float64)
            n1, n2 = len(X1_arr), len(X2_arr)
            K_cross: np.ndarray = np.zeros((n1, n2), dtype=np.float64)
            states1 = [self._state(x) for x in X1_arr]
            states2 = [self._state(x) for x in X2_arr]
            for i in range(n1):
                for j in range(n2):
                    K_cross[i, j] = float(np.abs(np.vdot(states1[i], states2[j])) ** 2)
            return K_cross
