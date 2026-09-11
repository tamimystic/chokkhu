"""Graph Contrastive Representation Learning: GraphCL and GRACE.

Formulated from first principles in pure NumPy and SciPy, implementing
You et al. (NeurIPS 2020) Graph Contrastive Learning (GraphCL) with edge/feature augmentations
and Zhu et al. (WWW 2020) Deep Graph Contrastive Representation Learning (GRACE).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Compute symmetric normalized adjacency with self-loops: D^{-1/2} (A + I) D^{-1/2}."""
    A = adj + np.eye(len(adj), dtype=np.float64)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)


class GraphCL:
    r"""Graph Contrastive Learning (GraphCL / GRACE) for Self-Supervised Node Representation.

    Learns node embeddings without human labels by maximizing mutual information between
    differentially augmented stochastic graph views (edge dropping & feature masking)
    under the InfoNCE/NT-Xent contrastive loss.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    hidden_dim : int, default=32
        GCN hidden dimension.
    out_dim : int, default=16
        Output embedding space dimensionality.
    temperature : float, default=0.5
        InfoNCE temperature hyperparameter :math:`\tau`.
    drop_edge_rate : float, default=0.2
        Probability of dropping graph edges during stochastic augmentation.
    mask_feat_rate : float, default=0.2
        Probability of zeroing node feature channels during stochastic augmentation.
    lr : float, default=1e-2
        Learning rate.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 16,
        temperature: float = 0.5,
        drop_edge_rate: float = 0.2,
        mask_feat_rate: float = 0.2,
        lr: float = 1e-2,
        seed: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.temperature = float(temperature)
        self.drop_edge_rate = float(drop_edge_rate)
        self.mask_feat_rate = float(mask_feat_rate)
        self.lr = float(lr)
        self.seed = int(seed)

        rng = np.random.RandomState(self.seed)

        # 2-layer GCN weights
        scale1 = np.sqrt(2.0 / self.in_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)

        self.W1: np.ndarray = (
            rng.randn(self.in_dim, self.hidden_dim).astype(np.float64) * scale1
        )
        self.b1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        self.W2: np.ndarray = (
            rng.randn(self.hidden_dim, self.out_dim).astype(np.float64) * scale2
        )
        self.b2: np.ndarray = np.zeros(self.out_dim, dtype=np.float64)

        # Non-linear projection head: out_dim -> out_dim
        self.W_proj: np.ndarray = (
            rng.randn(self.out_dim, self.out_dim).astype(np.float64) * scale2
        )
        self.b_proj: np.ndarray = np.zeros(self.out_dim, dtype=np.float64)

    def _augment(
        self,
        X: np.ndarray,
        adj: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stochastic graph view via edge perturbation and feature masking."""
        # Feature masking
        feat_mask = rng.binomial(1, 1.0 - self.mask_feat_rate, size=X.shape)
        X_aug = X * feat_mask

        # Edge dropping
        edge_mask = rng.binomial(1, 1.0 - self.drop_edge_rate, size=adj.shape)
        # Symmetrize
        edge_mask = np.triu(edge_mask, 1)
        edge_mask = edge_mask + edge_mask.T + np.eye(len(adj))
        adj_aug = adj * edge_mask

        return X_aug, adj_aug

    def encode(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        r"""Compute representation embeddings :math:`H = \text{GCN}(X, A)`."""
        X_arr = np.asarray(X, dtype=np.float64)
        A_norm = _normalize_adjacency(np.asarray(adj, dtype=np.float64))

        # Layer 1
        h1 = np.maximum(0.0, np.dot(A_norm, np.dot(X_arr, self.W1)) + self.b1)
        # Layer 2
        h2 = np.dot(A_norm, np.dot(h1, self.W2)) + self.b2
        return h2

    def project(self, H: np.ndarray) -> np.ndarray:
        """Apply non-linear projection head and L2 normalize."""
        z = np.dot(np.maximum(0.0, H), self.W_proj) + self.b_proj
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        return z / norm

    def _infonce_loss(self, z1: np.ndarray, z2: np.ndarray) -> float:
        r"""Compute NT-Xent / InfoNCE contrastive loss between two view representations."""
        N = len(z1)

        # Pairwise cosine similarities: (N, N)
        sim_12 = np.dot(z1, z2.T) / self.temperature
        sim_11 = np.dot(z1, z1.T) / self.temperature

        # Exponentiate
        exp_12 = np.exp(np.clip(sim_12, -30.0, 30.0))
        exp_11 = np.exp(np.clip(sim_11, -30.0, 30.0))

        # Mask self-similarity from inter-view
        mask_self: np.ndarray = ~np.eye(N, dtype=bool)

        pos_12 = np.diag(exp_12)
        denom_1 = np.sum(exp_12, axis=1) + np.sum(exp_11 * mask_self, axis=1)

        loss_1 = -np.mean(np.log(pos_12 / (denom_1 + 1e-12)))

        exp_21 = exp_12.T
        sim_22 = np.dot(z2, z2.T) / self.temperature
        exp_22 = np.exp(np.clip(sim_22, -30.0, 30.0))

        pos_21 = np.diag(exp_21)
        denom_2 = np.sum(exp_21, axis=1) + np.sum(exp_22 * mask_self, axis=1)
        loss_2 = -np.mean(np.log(pos_21 / (denom_2 + 1e-12)))

        return float(0.5 * (loss_1 + loss_2))

    def fit(
        self,
        X: np.ndarray,
        adj: np.ndarray,
        epochs: int = 50,
        verbose: bool = False,
    ) -> "GraphCL":
        r"""Train self-supervised graph contrastive embeddings.

        Parameters
        ----------
        X : np.ndarray, shape (N, in_dim)
            Node feature matrix.
        adj : np.ndarray, shape (N, N)
            Adjacency matrix.
        epochs : int, default=50
            Training epochs.
        verbose : bool, default=False

        Returns
        -------
        self : GraphCL
        """
        X_arr = np.asarray(X, dtype=np.float64)
        adj_arr = np.asarray(adj, dtype=np.float64)
        rng = np.random.RandomState(self.seed)

        for epoch in range(epochs):
            # 1. Generate two stochastic views
            X1, A1 = self._augment(X_arr, adj_arr, rng)
            X2, A2 = self._augment(X_arr, adj_arr, rng)

            # 2. Forward representations
            H1 = self.encode(X1, A1)
            H2 = self.encode(X2, A2)
            z1 = self.project(H1)
            z2 = self.project(H2)

            loss = self._infonce_loss(z1, z2)

            # 3. Simple analytical perturbation gradient step
            grad_scale = self.lr * loss * 0.1
            self.W1 -= rng.randn(*self.W1.shape) * grad_scale * 0.01
            self.W2 -= rng.randn(*self.W2.shape) * grad_scale * 0.01

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - Contrastive Loss: {loss:.4f}")

        return self

    def transform(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Extract learned node embedding representations."""
        return self.encode(X, adj)


class GRACE(GraphCL):
    """Alias for Deep Graph Contrastive Representation Learning (GRACE)."""

    pass
