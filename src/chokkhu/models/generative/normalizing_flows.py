"""Sovereign Invertible Normalizing Flows (RealNVP & Affine Coupling) in Pure NumPy."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class AffineCouplingLayer:
    """Invertible Affine Coupling Layer with Exact Jacobian Determinant."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        mask: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Binary split mask
        if mask is None:
            default_mask: np.ndarray = np.zeros(dim, dtype=np.float32)
            default_mask[: dim // 2] = 1.0
            self.mask = default_mask
        else:
            self.mask = np.asarray(mask, dtype=np.float32)

        # Scale network: predicts log-scale s
        self.w_s1: np.ndarray = (
            self.rng.randn(dim, hidden_dim) * np.sqrt(2.0 / dim)
        ).astype(np.float32)
        self.b_s1: np.ndarray = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w_s2: np.ndarray = (self.rng.randn(hidden_dim, dim) * 0.01).astype(
            np.float32
        )
        self.b_s2: np.ndarray = np.zeros((1, dim), dtype=np.float32)

        # Translation network: predicts shift t
        self.w_t1: np.ndarray = (
            self.rng.randn(dim, hidden_dim) * np.sqrt(2.0 / dim)
        ).astype(np.float32)
        self.b_t1: np.ndarray = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w_t2: np.ndarray = (self.rng.randn(hidden_dim, dim) * 0.01).astype(
            np.float32
        )
        self.b_t2: np.ndarray = np.zeros((1, dim), dtype=np.float32)

    def _scale_net(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.dot(x, self.w_s1) + self.b_s1)
        # Use tanh to stabilize scale factor in [-2, 2]
        s = np.tanh(np.dot(h, self.w_s2) + self.b_s2) * 2.0
        return s

    def _trans_net(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.dot(x, self.w_t1) + self.b_t1)
        t = np.dot(h, self.w_t2) + self.b_t2
        return t

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward mapping x -> y and log|det J|."""
        x = np.asarray(x, dtype=np.float32)
        x_masked = x * self.mask
        s = self._scale_net(x_masked)
        t = self._trans_net(x_masked)

        y = x_masked + (1.0 - self.mask) * (x * np.exp(s) + t)
        log_det = np.sum((1.0 - self.mask) * s, axis=-1)
        return y, log_det

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse mapping y -> x."""
        y = np.asarray(y, dtype=np.float32)
        y_masked = y * self.mask
        s = self._scale_net(y_masked)
        t = self._trans_net(y_masked)

        x = y_masked + (1.0 - self.mask) * ((y - t) * np.exp(-s))
        return x


class RealNVP:
    """Sovereign Real Non-Volume Preserving (RealNVP) Normalizing Flow."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        hidden_dim: int = 64,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Alternating mask pattern
        self.layers: List[AffineCouplingLayer] = []
        for i in range(num_layers):
            layer_mask: np.ndarray = np.zeros(dim, dtype=np.float32)
            if i % 2 == 0:
                layer_mask[::2] = 1.0
            else:
                layer_mask[1::2] = 1.0
            layer = AffineCouplingLayer(
                dim=dim,
                hidden_dim=hidden_dim,
                mask=layer_mask,
                random_state=random_state + i * 7,
            )
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data x to base Gaussian latent z and compute total log-det Jacobian."""
        z = np.asarray(x, dtype=np.float32)
        total_log_det = np.zeros(x.shape[0], dtype=np.float32)

        for layer in self.layers:
            z, log_det = layer.forward(z)
            total_log_det += log_det

        return z, total_log_det

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Transform base latent z back to data space x."""
        x = np.asarray(z, dtype=np.float32)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute exact log-likelihood log p(x) = log p_z(z) + sum log|det J|."""
        z, log_det = self.forward(x)
        # Log-likelihood under standard standard Gaussian N(0, I)
        log_pz = -0.5 * (self.dim * np.log(2.0 * np.pi) + np.sum(z**2, axis=-1))
        return log_pz + log_det

    def compute_loss(self, x: np.ndarray) -> float:
        """Compute Negative Log-Likelihood (NLL) loss."""
        return float(-np.mean(self.log_prob(x)))

    def sample(self, num_samples: int = 16) -> np.ndarray:
        """Sample new data instances from prior z ~ N(0, I)."""
        z = self.rng.randn(num_samples, self.dim).astype(np.float32)
        return self.inverse(z)
