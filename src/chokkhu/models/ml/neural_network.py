from __future__ import annotations

from typing import List, Optional
import numpy as np
from chokkhu.models.base import ChokkhuModel


class NeuralNetwork(ChokkhuModel):
    """Multi-Layer Perceptron (MLP) Deep Learning Model implemented from scratch.

    Supports Classification & Regression with arbitrary hidden layers and optimizers.
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        activation: str = "relu",
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        task: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.layers_config = layers or [64, 32]
        self.activation = activation
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task
        self.random_state = random_state

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 1

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, Z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -20.0, 20.0)))
        elif self.activation == "tanh":
            return np.tanh(Z)
        return Z

    def _activate_derivative(self, A: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (A > 0.0).astype(np.float64)
        elif self.activation == "sigmoid":
            return A * (1.0 - A)
        elif self.activation == "tanh":
            return 1.0 - A**2
        return np.ones_like(A)

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True) + 1e-15)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NeuralNetwork:
        if y is None:
            raise ValueError("Target y cannot be None for NeuralNetwork.")
        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        n_samples, in_features = X_arr.shape

        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(y_arr.dtype, np.floating)
                else "classification"
            )

        if self.task == "classification":
            self.classes_ = np.unique(y_arr)
            self.n_classes = len(self.classes_)
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_indices = np.array([class_to_idx[val] for val in y_arr])
            if self.n_classes == 2:
                Y = y_indices.reshape(-1, 1).astype(np.float64)
                out_dim = 1
            else:
                Y = np.zeros((n_samples, self.n_classes), dtype=np.float64)
                Y[np.arange(n_samples), y_indices] = 1.0
                out_dim = self.n_classes
        else:
            Y = y_arr.reshape(-1, 1).astype(np.float64)
            out_dim = 1

        layer_dims = [in_features] + self.layers_config + [out_dim]
        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(2.0 / fan_in)  # He initialization
            W = rng.normal(0.0, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

        # Training loop (SGD / Mini-batch)
        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            X_shuffled = X_arr[perm]
            Y_shuffled = Y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                b_size = end - start

                # Forward pass
                activations = [X_batch]
                for idx in range(len(self.weights) - 1):
                    Z = activations[-1] @ self.weights[idx] + self.biases[idx]
                    A = self._activate(Z)
                    activations.append(A)

                # Output layer
                Z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
                if self.task == "classification":
                    if self.n_classes == 2:
                        A_out = 1.0 / (1.0 + np.exp(-np.clip(Z_out, -20.0, 20.0)))
                        dZ = (A_out - Y_batch) / b_size
                    else:
                        A_out = self._softmax(Z_out)
                        dZ = (A_out - Y_batch) / b_size
                else:
                    A_out = Z_out
                    dZ = (A_out - Y_batch) / b_size

                activations.append(A_out)

                # Backward pass
                for idx in reversed(range(len(self.weights))):
                    A_prev = activations[idx]
                    dW = A_prev.T @ dZ
                    db = np.sum(dZ, axis=0, keepdims=True)

                    if idx > 0:
                        dA_prev = dZ @ self.weights[idx].T
                        dZ = dA_prev * self._activate_derivative(A_prev)

                    self.weights[idx] -= self.lr * dW
                    self.biases[idx] -= self.lr * db

        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        for idx in range(len(self.weights) - 1):
            Z = A @ self.weights[idx] + self.biases[idx]
            A = self._activate(Z)

        Z_out = A @ self.weights[-1] + self.biases[-1]
        if self.task == "classification":
            if self.n_classes == 2:
                return 1.0 / (1.0 + np.exp(-np.clip(Z_out, -20.0, 20.0)))
            else:
                return self._softmax(Z_out)
        return Z_out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        probs = self._forward(X)
        if self.n_classes == 2:
            return np.hstack([1.0 - probs, probs])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward(X)
        if self.task == "classification":
            if self.classes_ is None:
                return (out >= 0.5).astype(int).flatten()
            if self.n_classes == 2:
                preds = (out.flatten() >= 0.5).astype(int)
                return self.classes_[preds]
            else:
                idx = np.argmax(out, axis=1)
                return self.classes_[idx]
        return out.flatten()
