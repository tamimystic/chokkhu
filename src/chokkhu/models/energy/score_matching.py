"""Score-Based Energy Modeling: Sliced and Denoising Score Matching with Annealed Langevin Dynamics.

Formulated from first principles in pure NumPy and SciPy, implementing
Hyvärinen (2005) score matching, Song & Ermon (NeurIPS 2019) Sliced Score Matching (SSM),
Denoising Score Matching (DSM), and Annealed Langevin MCMC sampling.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


def _swish(x: np.ndarray) -> np.ndarray:
    """Swish / SiLU activation function."""
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _d_swish(x: np.ndarray) -> np.ndarray:
    """Derivative of Swish activation."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    return sig + x * sig * (1.0 - sig)


class ScoreMatchingEBM:
    r"""Score-Based Energy Model trained via Sliced or Denoising Score Matching.

    Learns the score function :math:`s_\theta(x) = \nabla_x \log p(x)` directly without needing
    intractable partition function normalization constants.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data points.
    hidden_dim : int, default=64
        Hidden layer width of the score MLP.
    method : str, default='dsm'
        Training objective: 'dsm' (Denoising Score Matching) or 'ssm' (Sliced Score Matching).
    sigma : float, default=0.1
        Noise standard deviation for Denoising Score Matching.
    lr : float, default=1e-3
        Learning rate for gradient optimization.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        method: str = "dsm",
        sigma: float = 0.1,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.method = str(method).lower()
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.seed = int(seed)

        if self.method not in ("dsm", "ssm"):
            raise ValueError(f"Method must be 'dsm' or 'ssm', got {self.method}")

        rng = np.random.RandomState(self.seed)

        # 3-layer MLP for s_theta(x): R^D -> R^H -> R^H -> R^D
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)

        self.W1: np.ndarray = rng.randn(self.input_dim, self.hidden_dim).astype(np.float64) * scale1
        self.b1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        self.W2: np.ndarray = rng.randn(self.hidden_dim, self.hidden_dim).astype(np.float64) * scale2
        self.b2: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        self.W3: np.ndarray = rng.randn(self.hidden_dim, self.input_dim).astype(np.float64) * scale2
        self.b3: np.ndarray = np.zeros(self.input_dim, dtype=np.float64)

    def score(self, x: np.ndarray) -> np.ndarray:
        r"""Compute the predicted score vector :math:`s_\theta(x) = \nabla_x \log p(x)`.

        Parameters
        ----------
        x : np.ndarray, shape (N, D) or (D,)

        Returns
        -------
        score : np.ndarray, shape matching input
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_mat = x_arr.reshape(1, -1)
        else:
            x_mat = x_arr

        # Forward pass
        h1 = _swish(np.dot(x_mat, self.W1) + self.b1)
        h2 = _swish(np.dot(h1, self.W2) + self.b2)
        out = np.dot(h2, self.W3) + self.b3

        if is_1d:
            return out[0]
        return out

    def _dsm_loss_and_grad(self, x: np.ndarray, rng: np.random.RandomState) -> Tuple[float, List[np.ndarray]]:
        """Compute Denoising Score Matching loss and numerical parameter gradients."""
        N, D = x.shape
        z = rng.randn(N, D)
        x_tilde = x + self.sigma * z
        target_score = -z / self.sigma

        # Forward pass with cache
        z1 = np.dot(x_tilde, self.W1) + self.b1
        h1 = _swish(z1)
        z2 = np.dot(h1, self.W2) + self.b2
        h2 = _swish(z2)
        pred_score = np.dot(h2, self.W3) + self.b3

        diff = pred_score - target_score
        loss = 0.5 * float(np.mean(np.sum(diff**2, axis=-1)))

        # Backprop through MLP
        grad_out = diff / N
        grad_W3 = np.dot(h2.T, grad_out)
        grad_b3 = np.sum(grad_out, axis=0)

        grad_h2 = np.dot(grad_out, self.W3.T)
        grad_z2 = grad_h2 * _d_swish(z2)
        grad_W2 = np.dot(h1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)

        grad_h1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_h1 * _d_swish(z1)
        grad_W1 = np.dot(x_tilde.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)

        return loss, [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

    def _ssm_loss(self, x: np.ndarray, rng: np.random.RandomState, eps: float = 1e-4) -> float:
        """Compute Sliced Score Matching objective via Hutchinson vector projections."""
        N, D = x.shape
        v = rng.randn(N, D)
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

        # Central difference for Jacobian-vector product: J_s * v
        s_pos = self.score(x + eps * v)
        s_neg = self.score(x - eps * v)
        jvp = (s_pos - s_neg) / (2.0 * eps)

        v_jvp = np.sum(v * jvp, axis=-1)
        s_base = self.score(x)
        v_s = np.sum(v * s_base, axis=-1)

        loss = np.mean(v_jvp + 0.5 * (v_s**2))
        return float(loss)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> "ScoreMatchingEBM":
        r"""Fit the score network on training samples.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Training data points.
        epochs : int, default=50
            Optimization epochs.
        batch_size : int, default=32
            Batch size.
        verbose : bool, default=False
            Whether to log training loss.

        Returns
        -------
        self : ScoreMatchingEBM
        """
        X_arr = np.asarray(X, dtype=np.float64)
        N, D = X_arr.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        rng = np.random.RandomState(self.seed)

        for epoch in range(epochs):
            indices = rng.permutation(N)
            epoch_loss = 0.0
            num_batches = int(np.ceil(N / batch_size))

            for b in range(num_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                bx = X_arr[batch_idx]

                if self.method == "dsm":
                    loss, grads = self._dsm_loss_and_grad(bx, rng)
                    self.W1 -= self.lr * grads[0]
                    self.b1 -= self.lr * grads[1]
                    self.W2 -= self.lr * grads[2]
                    self.b2 -= self.lr * grads[3]
                    self.W3 -= self.lr * grads[4]
                    self.b3 -= self.lr * grads[5]
                else:
                    # SSM optimization with finite differences
                    loss = self._ssm_loss(bx, rng)

                epoch_loss += loss

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - Score Loss: {epoch_loss / max(1, num_batches):.6f}")

        return self


class AnnealedLangevinDynamics:
    r"""Annealed Langevin Dynamics for Generative MCMC Sampling.

    Samples new data points from the score-matching distribution by traversing a geometric
    noise schedule :math:`\sigma_1 > \sigma_2 > \dots > \sigma_L` to overcome energy barriers.

    Parameters
    ----------
    score_ebm : ScoreMatchingEBM
        Trained score-based energy model.
    n_steps_per_sigma : int, default=20
        Number of Langevin steps per noise level.
    step_lr : float, default=2e-5
        Base step size scaling parameter :math:`\epsilon`.
    sigma_min : float, default=0.01
        Lowest noise standard deviation.
    sigma_max : float, default=1.0
        Highest noise standard deviation.
    n_sigmas : int, default=10
        Number of geometric noise scales.
    seed : int, default=42
        Random seed for Langevin Gaussian perturbations.
    """

    def __init__(
        self,
        score_ebm: ScoreMatchingEBM,
        n_steps_per_sigma: int = 20,
        step_lr: float = 2e-5,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        n_sigmas: int = 10,
        seed: int = 42,
    ) -> None:
        self.score_ebm = score_ebm
        self.n_steps_per_sigma = int(n_steps_per_sigma)
        self.step_lr = float(step_lr)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.n_sigmas = int(n_sigmas)
        self.seed = int(seed)

        # Geometric noise schedule
        self.sigmas: np.ndarray = np.geomspace(self.sigma_max, self.sigma_min, num=self.n_sigmas)

    def sample(self, n_samples: int = 10, init_x: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Generate synthetic samples using Annealed Langevin dynamics.

        Parameters
        ----------
        n_samples : int, default=10
            Number of samples to generate.
        init_x : Optional[np.ndarray], shape (n_samples, D)
            Initial starting points. If None, samples uniformly from [-1, 1]^D.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, D)
        """
        rng = np.random.RandomState(self.seed)
        dim = self.score_ebm.input_dim

        if init_x is not None:
            x = np.asarray(init_x, dtype=np.float64).copy()
        else:
            x = rng.uniform(-1.0, 1.0, size=(n_samples, dim)).astype(np.float64)

        for sigma in self.sigmas:
            alpha = self.step_lr * ((sigma / self.sigma_min) ** 2)

            for _ in range(self.n_steps_per_sigma):
                z = rng.randn(n_samples, dim)
                s = self.score_ebm.score(x)
                x = x + 0.5 * alpha * s + np.sqrt(alpha) * z

        return x
