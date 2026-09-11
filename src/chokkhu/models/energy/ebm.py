"""Energy-Based Models, Stochastic Gradient Langevin Dynamics, and Sliced Score Matching.

Formulated from first principles using unnormalized Boltzmann densities, SGLD MCMC sampling,
and Hutchinson trace estimator sliced score matching in pure NumPy.
"""

from typing import Callable, Optional

import numpy as np


class EnergyBasedModel:
    r"""Continuous Energy-Based Model (EBM) with Langevin Dynamics.

        Parameterizes continuous unnormalized density:
        p_	heta(x) =
    rac{e^{-E_	heta(x)}}{Z_	heta}

        Parameters
        ----------
        in_dim : int
            Feature space dimension.
        hidden_dim : int, default=64
            Hidden layer width.
        seed : int, default=42
            Random seed for parameter initialization.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # 2-layer MLP parameterization with Softplus activations
        scale1 = 1.0 / np.sqrt(self.in_dim)
        scale2 = 1.0 / np.sqrt(self.hidden_dim)

        self.W1 = self.rng.randn(self.in_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self.rng.randn(self.hidden_dim, self.hidden_dim) * scale2
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = self.rng.randn(self.hidden_dim, 1) * scale2
        self.b3 = np.zeros(1)

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Evaluate scalar energy E(x) for batch of samples.

        Parameters
        ----------
        x : np.ndarray of shape (N, in_dim) or (in_dim,)

        Returns
        -------
        E : np.ndarray of shape (N,)
            Scalar energy values. Lower energy corresponds to higher probability.
        """
        x_mat = np.atleast_2d(np.asarray(x, dtype=float))

        # Layer 1: Softplus(x W1 + b1) = log(1 + exp(h))
        h1 = np.dot(x_mat, self.W1) + self.b1
        a1 = np.log1p(np.exp(np.clip(h1, -30.0, 30.0)))

        # Layer 2: Softplus(a1 W2 + b2)
        h2 = np.dot(a1, self.W2) + self.b2
        a2 = np.log1p(np.exp(np.clip(h2, -30.0, 30.0)))

        # Output Layer: scalar energy
        E = np.dot(a2, self.W3) + self.b3
        return E.flatten() if x.ndim == 2 else E.flatten()[0]

    def energy_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute energy gradient nabla_x E(x) with respect to inputs."""
        x_mat = np.atleast_2d(np.asarray(x, dtype=float))
        grad = np.zeros_like(x_mat)

        for col in range(x_mat.shape[1]):
            x_plus = x_mat.copy()
            x_minus = x_mat.copy()

            x_plus[:, col] += eps
            x_minus[:, col] -= eps

            e_plus = self.energy(x_plus)
            e_minus = self.energy(x_minus)

            grad[:, col] = (e_plus - e_minus) / (2.0 * eps)

        return grad if x.ndim == 2 else grad[0]

    def sample_sgld(
        self,
        n_samples: int = 100,
        num_steps: int = 60,
        step_size: float = 0.05,
        noise_scale: float = 0.01,
        init_x: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> np.ndarray:
        r"""Sample from Boltzmann density using Stochastic Gradient Langevin Dynamics (SGLD).

                x_{k+1} = x_k -
        rac{\epsilon}{2}
        abla_x E(x_k) + \sqrt{\epsilon} z_k

                Parameters
                ----------
                n_samples : int, default=100
                    Number of MCMC chains to run.
                num_steps : int, default=60
                    Number of Langevin diffusion steps.
                step_size : float, default=0.05
                    Langevin step size epsilon.
                noise_scale : float, default=0.01
                    Thermal noise magnitude.
                init_x : Optional[np.ndarray], default=None
                    Initial chain state. If None, samples from standard normal.
                seed : int, default=42
                    Random seed.

                Returns
                -------
                samples : np.ndarray of shape (n_samples, in_dim)
                    Generated samples.
        """
        rng = np.random.RandomState(seed)
        if init_x is None:
            x_k = rng.randn(n_samples, self.in_dim)
        else:
            x_k = np.array(init_x, dtype=float, copy=True)

        for _ in range(num_steps):
            grad_E = self.energy_gradient(x_k)
            thermal_noise = rng.randn(*x_k.shape) * noise_scale
            # SGLD descent towards low energy (high probability)
            x_k = x_k - 0.5 * step_size * grad_E + np.sqrt(step_size) * thermal_noise

        return x_k


class SlicedScoreMatching:
    r"""Sliced Score Matching (SSM) Objective for Score-Based Models.

        Hutchinson random vector projection estimator:
        \mathcal{L}_{	ext{SSM}}(	heta) = \mathbb{E}_{v \sim p_v, x \sim p_x} \left[ v^T
    abla_x s_	heta(x) v +
    rac{1}{2} (v^T s_	heta(x))^2
    ight]

        Parameters
        ----------
        score_fn : Callable[[np.ndarray], np.ndarray]
            Vector score function mapping input x of shape (N, d) to score vectors nabla_x log p(x) of shape (N, d).
        num_projections : int, default=4
            Number of random Hutchinson projection slices per sample.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        num_projections: int = 4,
    ) -> None:
        self.score_fn = score_fn
        self.num_projections = max(1, int(num_projections))

    def compute_loss(
        self,
        X: np.ndarray,
        eps: float = 1e-4,
        seed: int = 42,
    ) -> float:
        """Evaluate sliced score matching loss over dataset X.

        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            Input training samples.
        eps : float, default=1e-4
            Finite-difference step size for directional Jacobian product.
        seed : int, default=42
            Random seed for projection vectors.

        Returns
        -------
        loss : float
            Scalar sliced score matching loss.
        """
        x_mat = np.atleast_2d(np.asarray(X, dtype=float))
        N, d = x_mat.shape
        rng = np.random.RandomState(seed)

        total_loss: float = 0.0

        scores = self.score_fn(x_mat)  # (N, d)

        for _ in range(self.num_projections):
            # Sample random Rademacher (+1, -1) or Gaussian unit vectors
            v = rng.randn(N, d)
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

            # 1. Quadratic score term: 0.5 * (v^T s(x))^2
            v_dot_s = np.sum(v * scores, axis=-1)  # (N,)
            quad_term = 0.5 * (v_dot_s**2)

            # 2. Trace term: v^T (nabla_x s(x)) v via central directional difference:
            # (s(x + eps*v) - s(x - eps*v))^T v / (2*eps)
            s_plus = self.score_fn(x_mat + eps * v)
            s_minus = self.score_fn(x_mat - eps * v)

            directional_deriv = (s_plus - s_minus) / (2.0 * eps)
            trace_term = np.sum(v * directional_deriv, axis=-1)  # (N,)

            total_loss += float(np.mean(quad_term + trace_term))

        return float(total_loss / self.num_projections)
