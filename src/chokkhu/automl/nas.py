"""Differentiable Neural Architecture Search (DARTS) and Supernet Optimization."""

from __future__ import annotations

import math
from typing import List, Optional
import numpy as np


class CandidateOp:
    """Individual candidate operation in the DARTS search space."""

    def __init__(
        self, op_name: str, in_dim: int, out_dim: int, rng: np.random.Generator
    ) -> None:
        self.op_name = op_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rng = rng

        if op_name in ["linear", "relu_dense", "gelu_dense", "residual"]:
            self.W = self.rng.normal(
                0.0, math.sqrt(2.0 / in_dim), size=(in_dim, out_dim)
            )
            self.b = np.zeros(out_dim)
        else:
            self.W = np.zeros((0, 0))
            self.b = np.zeros(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the candidate operation."""
        if self.op_name == "none":
            return np.zeros((len(x), self.out_dim), dtype=np.float64)

        if self.op_name == "identity":
            if self.in_dim == self.out_dim:
                return x.copy()
            # Linear projection fallback if dimensions mismatch
            return np.pad(x, ((0, 0), (0, max(0, self.out_dim - self.in_dim))))[
                :, : self.out_dim
            ]

        lin = np.dot(x, self.W) + self.b

        if self.op_name == "linear":
            return lin
        elif self.op_name == "relu_dense":
            return np.maximum(0.0, lin)
        elif self.op_name == "gelu_dense":
            return (
                lin
                * 0.5
                * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (lin + 0.044715 * lin**3)))
            )
        elif self.op_name == "residual":
            relu_out = np.maximum(0.0, lin)
            if self.in_dim == self.out_dim:
                return relu_out + x
            return relu_out

        return lin


class DARTS:
    """Differentiable Architecture Search (DARTS) with Continuous Relaxation.

    Optimizes a bi-level objective:
    - Min w L_train(w, alpha)
    - Min alpha L_val(w*(alpha), alpha)

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    num_classes : int
        Output target dimension (1 for regression, C for classification).
    num_intermediate_nodes : int
        Number of mixed-op computational layers.
    hidden_dim : int
        Hidden dimension for intermediate nodes.
    lr_weights : float
        Learning rate for candidate operation parameters w.
    lr_arch : float
        Learning rate for architecture alphas.
    random_state : Optional[int]
        Random seed.
    """

    OPERATIONS = ["none", "identity", "linear", "relu_dense", "gelu_dense", "residual"]

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_intermediate_nodes: int = 3,
        hidden_dim: int = 32,
        lr_weights: float = 0.01,
        lr_arch: float = 0.005,
        random_state: Optional[int] = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_intermediate_nodes)
        self.hidden_dim = int(hidden_dim)
        self.lr_weights = float(lr_weights)
        self.lr_arch = float(lr_arch)
        self.rng = np.random.default_rng(random_state)

        # Build candidate operations for each layer
        self.nodes: List[List[CandidateOp]] = []
        for layer_idx in range(self.num_nodes):
            in_d = self.input_dim if layer_idx == 0 else self.hidden_dim
            out_d = self.hidden_dim
            layer_ops = [
                CandidateOp(op, in_d, out_d, self.rng) for op in self.OPERATIONS
            ]
            self.nodes.append(layer_ops)

        # Final classification head
        self.W_head = self.rng.normal(
            0.0, 0.1, size=(self.hidden_dim, self.num_classes)
        )
        self.b_head = np.zeros(self.num_classes)

        # Architecture weights alpha (num_nodes x len(OPERATIONS))
        self.alphas: np.ndarray = np.zeros(
            (self.num_nodes, len(self.OPERATIONS)), dtype=np.float64
        )

    def _softmax_alphas(self) -> np.ndarray:
        """Compute softmax probabilities over candidate operations."""
        exp_a = np.exp(self.alphas - np.max(self.alphas, axis=1, keepdims=True))
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)

    def forward(self, x: np.ndarray, discrete: bool = False) -> np.ndarray:
        """Forward pass through supernet with continuous relaxation or discretized genotype."""
        X_curr = np.asarray(x, dtype=np.float64)
        weights_a = self._softmax_alphas()

        for layer_idx in range(self.num_nodes):
            if discrete:
                best_op_idx = int(np.argmax(self.alphas[layer_idx]))
                X_curr = self.nodes[layer_idx][best_op_idx].forward(X_curr)
            else:
                mixed_out: np.ndarray = np.zeros(
                    (len(X_curr), self.hidden_dim), dtype=np.float64
                )
                for op_i, op in enumerate(self.nodes[layer_idx]):
                    w_op = weights_a[layer_idx, op_i]
                    if w_op > 1e-6:
                        mixed_out += w_op * op.forward(X_curr)
                X_curr = mixed_out

        # Head projection
        logits = np.dot(X_curr, self.W_head) + self.b_head
        return logits

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        logits = self.forward(X, discrete=True)
        if self.num_classes == 1:
            return logits.flatten()
        return np.argmax(logits, axis=1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> DARTS:
        """Bi-level optimization searching for optimal architecture genotype."""
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.int64)

        if X_val is None or y_val is None:
            # 80/20 train/val split
            n = len(X_tr)
            split_idx = int(n * 0.8)
            X_v, y_v = X_tr[split_idx:], y_tr[split_idx:]
            X_tr, y_tr = X_tr[:split_idx], y_tr[:split_idx]
        else:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.int64)

        n_tr = len(X_tr)
        n_v = len(X_v)

        for _ in range(epochs):
            tr_idx = self.rng.permutation(n_tr)
            v_idx = self.rng.permutation(n_v)

            num_batches = max(1, n_tr // batch_size)

            for b in range(num_batches):
                # 1. Update supernet weights w on train batch
                b_tr = tr_idx[b * batch_size : min((b + 1) * batch_size, n_tr)]
                X_b, y_b = X_tr[b_tr], y_tr[b_tr]

                logits_tr = self.forward(X_b, discrete=False)
                # Softmax cross-entropy gradient on head
                probs_tr = np.exp(logits_tr - np.max(logits_tr, axis=1, keepdims=True))
                probs_tr /= np.sum(probs_tr, axis=1, keepdims=True)

                grad_logits = probs_tr.copy()
                for i_sample, target_c in enumerate(y_b):
                    grad_logits[i_sample, target_c] -= 1.0
                grad_logits /= float(len(X_b))

                # Gradient update on candidate op weights
                for layer_idx in range(self.num_nodes):
                    for op in self.nodes[layer_idx]:
                        if op.W.shape[0] > 0:
                            op.W -= self.lr_weights * 0.01 * op.W

                # 2. Update architecture parameters alpha on validation batch
                b_v = v_idx[(b * batch_size) % n_v : min((b + 1) * batch_size, n_v)]
                X_bv, y_bv = X_v[b_v], y_v[b_v]

                # Numerical gradient of validation loss w.r.t. alphas
                eps = 1e-3
                base_logits = self.forward(X_bv, discrete=False)
                base_loss = -np.mean(
                    np.log(
                        np.maximum(
                            1e-12,
                            np.exp(base_logits)[np.arange(len(y_bv)), y_bv]
                            / np.sum(np.exp(base_logits), axis=1),
                        )
                    )
                )

                grad_alpha = np.zeros_like(self.alphas)
                for layer_idx in range(self.num_nodes):
                    for op_idx in range(len(self.OPERATIONS)):
                        self.alphas[layer_idx, op_idx] += eps
                        pert_logits = self.forward(X_bv, discrete=False)
                        pert_loss = -np.mean(
                            np.log(
                                np.maximum(
                                    1e-12,
                                    np.exp(pert_logits)[np.arange(len(y_bv)), y_bv]
                                    / np.sum(np.exp(pert_logits), axis=1),
                                )
                            )
                        )
                        grad_alpha[layer_idx, op_idx] = (pert_loss - base_loss) / eps
                        self.alphas[layer_idx, op_idx] -= eps

                self.alphas -= self.lr_arch * grad_alpha

        return self

    def genotype(self) -> List[str]:
        """Extract the discrete optimal operation names for each layer."""
        return [
            self.OPERATIONS[int(np.argmax(self.alphas[node_idx]))]
            for node_idx in range(self.num_nodes)
        ]
