"""Hyperbolic Riemannian Manifold Geometry and Representation Learning.

Formulated from first principles using the Poincaré Ball model (Möbius addition,
exponential/logarithmic maps, Riemannian SGD) and the Lorentz Hyperboloid model in pure NumPy.
"""

import numpy as np
from typing import List, Tuple


class PoincareBallEmbedding:
    """Poincaré Ball Model for Hyperbolic Representation Learning.

    Parameters
    ----------
    dim : int, default=2
        Dimensionality of the hyperbolic embedding space.
    c : float, default=1.0
        Hyperbolic negative curvature parameter (curvature kappa = -c).
    eps : float, default=1e-5
        Numerical boundary margin: ball boundary is ||x|| < 1/sqrt(c) - eps.
    seed : int, default=42
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        dim: int = 2,
        c: float = 1.0,
        eps: float = 1e-5,
        seed: int = 42,
    ) -> None:
        self.dim = int(dim)
        self.c = float(c)
        self.eps = float(eps)
        self.max_norm = (1.0 / np.sqrt(self.c)) - self.eps
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project points back inside the Poincaré ball boundary ||x|| <= max_norm."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        scale = np.where(norm >= self.max_norm, self.max_norm / (norm + 1e-12), 1.0)
        return x * scale

    def mobius_add(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute Möbius addition: u (+) v in the Poincaré ball."""
        u2 = np.sum(u**2, axis=-1, keepdims=True)
        v2 = np.sum(v**2, axis=-1, keepdims=True)
        uv = np.sum(u * v, axis=-1, keepdims=True)

        num = (1.0 + 2.0 * self.c * uv + self.c * v2) * u + (1.0 - self.c * u2) * v
        den = 1.0 + 2.0 * self.c * uv + (self.c**2) * u2 * v2 + 1e-12

        return self.project(num / den)

    def distance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute geodesic distance d_c(u, v) between pairs in Poincaré ball."""
        diff = self.mobius_add(-u, v)
        diff_norm = np.linalg.norm(diff, axis=-1)
        scaled_norm = np.clip(np.sqrt(self.c) * diff_norm, 0.0, 1.0 - 1e-7)
        return (2.0 / np.sqrt(self.c)) * np.arctanh(scaled_norm)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map exp_x(v): maps tangent vector v at x to manifold."""
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        if np.all(v_norm < 1e-12):
            return x.copy()

        lambda_x = 2.0 / (1.0 - self.c * np.sum(x**2, axis=-1, keepdims=True) + 1e-12)
        scaled_v = (
            (v / (v_norm + 1e-12))
            * (1.0 / np.sqrt(self.c))
            * np.tanh(np.sqrt(self.c) * lambda_x * v_norm / 2.0)
        )
        return self.mobius_add(x, scaled_v)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Logarithmic map log_x(y): maps manifold point y to tangent space T_x M."""
        diff = self.mobius_add(-x, y)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        lambda_x = 2.0 / (1.0 - self.c * np.sum(x**2, axis=-1, keepdims=True) + 1e-12)

        scaled_norm = np.clip(np.sqrt(self.c) * diff_norm, 0.0, 1.0 - 1e-7)
        coeff = (
            (2.0 / (np.sqrt(self.c) * lambda_x))
            * np.arctanh(scaled_norm)
            / (diff_norm + 1e-12)
        )
        return coeff * diff

    def riemannian_gradient(
        self, x: np.ndarray, euclidean_grad: np.ndarray
    ) -> np.ndarray:
        """Rescale Euclidean gradient to Riemannian gradient via inverse conformal metric."""
        lambda_x = 2.0 / (1.0 - self.c * np.sum(x**2, axis=-1, keepdims=True) + 1e-12)
        return euclidean_grad / (lambda_x**2 + 1e-12)

    def fit_graph(
        self,
        edges: List[Tuple[int, int]],
        num_nodes: int,
        num_epochs: int = 50,
        lr: float = 0.05,
        num_negatives: int = 5,
    ) -> np.ndarray:
        """Embed graph nodes into Poincaré ball using Riemannian SGD with margin ranking loss."""
        embeddings = self.rng.randn(num_nodes, self.dim) * 1e-3
        embeddings = self.project(embeddings)

        for epoch in range(num_epochs):
            for u, v in edges:
                u_emb = embeddings[u : u + 1]
                v_emb = embeddings[v : v + 1]
                d_pos = self.distance(u_emb, v_emb)[0]

                neg_nodes = self.rng.choice(num_nodes, size=num_negatives)
                for neg in neg_nodes:
                    if neg == u or neg == v:
                        continue
                    neg_emb = embeddings[neg : neg + 1]
                    d_neg = self.distance(u_emb, neg_emb)[0]

                    margin = 0.5
                    if d_pos - d_neg + margin > 0:
                        grad_pos = embeddings[v] - embeddings[u]
                        grad_neg = embeddings[neg] - embeddings[u]

                        r_grad_u = self.riemannian_gradient(
                            embeddings[u : u + 1], -grad_pos + grad_neg
                        )[0]
                        r_grad_v = self.riemannian_gradient(
                            embeddings[v : v + 1], -grad_pos
                        )[0]
                        r_grad_neg = self.riemannian_gradient(
                            embeddings[neg : neg + 1], grad_neg
                        )[0]

                        embeddings[u] = self.exp_map(
                            embeddings[u : u + 1], -lr * r_grad_u[None, :]
                        )[0]
                        embeddings[v] = self.exp_map(
                            embeddings[v : v + 1], -lr * r_grad_v[None, :]
                        )[0]
                        embeddings[neg] = self.exp_map(
                            embeddings[neg : neg + 1], -lr * r_grad_neg[None, :]
                        )[0]

        return embeddings


class LorentzManifold:
    """Lorentz (Hyperboloid) Model of Hyperbolic Space."""

    def __init__(self, dim: int = 2, c: float = 1.0) -> None:
        self.dim = int(dim)
        self.c = float(c)

    def minkowski_dot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Minkowski inner product: <x, y>_L = -x_0 y_0 + sum_{i=1}^d x_i y_i."""
        time_part = -x[..., 0] * y[..., 0]
        space_part = np.sum(x[..., 1:] * y[..., 1:], axis=-1)
        return time_part + space_part

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Geodesic distance d_L(x, y) = (1 / sqrt(c)) * arcosh(-c * <x, y>_L)."""
        dot = self.minkowski_dot(x, y)
        arg = np.maximum(1.0, -self.c * dot)
        return (1.0 / np.sqrt(self.c)) * np.arccosh(arg)

    def poincare_to_lorentz(self, u: np.ndarray) -> np.ndarray:
        """Isomorphic coordinate diffeomorphism: Poincaré ball D^d -> Lorentz H^d."""
        u2 = np.sum(u**2, axis=-1, keepdims=True)
        den = 1.0 - self.c * u2 + 1e-12

        x0 = (1.0 + self.c * u2) / (np.sqrt(self.c) * den)
        x_space = (2.0 * u) / den
        return np.concatenate([x0, x_space], axis=-1)

    def lorentz_to_poincare(self, x: np.ndarray) -> np.ndarray:
        """Isomorphic coordinate diffeomorphism: Lorentz H^d -> Poincaré ball D^d."""
        x0 = x[..., 0:1]
        x_space = x[..., 1:]
        return x_space / (1.0 + np.sqrt(self.c) * x0 + 1e-12)
