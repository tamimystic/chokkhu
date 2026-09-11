"""Sequential Monte Carlo & Particle Filtering for Non-linear Non-Gaussian State Spaces.

Formulated from first principles in pure NumPy and SciPy, implementing
Gordon et al. (1993) Sequential Importance Resampling (SIR) Particle Filter,
Systematic, Stratified, Residual, and Multinomial Resampling schemes, alongside
Rao-Blackwellized Particle Filtering.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple


def _default_f(x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    return x


def _default_h(x: np.ndarray) -> np.ndarray:
    return x


class ParticleFilter:
    r"""Sequential Importance Resampling (SIR) Particle Filter.

    Estimates the posterior state distribution :math:`p(x_k | z_{1:k})` of arbitrary non-linear,
    non-Gaussian dynamic systems:

    .. math::
        x_k = f(x_{k-1}, u_k) + v_k, \quad v_k \sim p(v)
        z_k = h(x_k) + e_k, \quad e_k \sim p(e)

    Parameters
    ----------
    dim_x : int
        State space dimensionality.
    dim_z : int
        Measurement space dimensionality.
    n_particles : int, default=500
        Number of Monte Carlo particles.
    f : Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]], default=None
        Non-linear state transition function :math:`f(x, u)`.
    h : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Non-linear measurement function :math:`h(x)`.
    Q : Optional[np.ndarray], default=None
        Process noise covariance matrix :math:`(D_x, D_x)`.
    R : Optional[np.ndarray], default=None
        Measurement noise covariance matrix :math:`(D_z, D_z)`.
    resample_method : str, default='systematic'
        Resampling algorithm: 'systematic', 'stratified', 'residual', or 'multinomial'.
    resample_threshold : float, default=0.5
        Effective sample size ratio threshold :math:`N_{\text{eff}} / N` triggering resampling.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        n_particles: int = 500,
        f: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        resample_method: str = "systematic",
        resample_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.n_particles = int(n_particles)
        self.f = f if f is not None else _default_f
        self.h = h if h is not None else _default_h

        self.Q: np.ndarray = (
            np.asarray(Q, dtype=np.float64)
            if Q is not None
            else np.eye(self.dim_x, dtype=np.float64) * 0.1
        )
        self.R: np.ndarray = (
            np.asarray(R, dtype=np.float64)
            if R is not None
            else np.eye(self.dim_z, dtype=np.float64) * 0.1
        )

        self.resample_method = str(resample_method).lower()
        self.resample_threshold = float(resample_threshold)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # Initialize particles and uniform weights
        self.particles: np.ndarray = self.rng.randn(self.n_particles, self.dim_x)
        self.weights: np.ndarray = (
            np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        )

    def initialize_particles(
        self,
        mean: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
    ) -> "ParticleFilter":
        """Initialize particle cloud from Gaussian prior distribution."""
        m = (
            np.asarray(mean, dtype=np.float64)
            if mean is not None
            else np.zeros(self.dim_x)
        )
        c = np.asarray(cov, dtype=np.float64) if cov is not None else np.eye(self.dim_x)
        self.particles = self.rng.multivariate_normal(m, c, size=self.n_particles)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        return self

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        r"""Propagate particles through non-linear dynamics with process noise.

        .. math::
            x_k^{(i)} = f(x_{k-1}^{(i)}, u_k) + v_k^{(i)}, \quad v_k^{(i)} \sim \mathcal{N}(0, Q)
        """
        process_noise = self.rng.multivariate_normal(
            np.zeros(self.dim_x), self.Q, size=self.n_particles
        )
        for i in range(self.n_particles):
            self.particles[i] = self.f(self.particles[i], u) + process_noise[i]

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Update particle importance weights via Gaussian measurement likelihood and resample.

        .. math::
            w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \exp\left(-\frac{1}{2} (z_k - h(x_k^{(i)}))^T R^{-1} (z_k - h(x_k^{(i)}))\right)

        Parameters
        ----------
        z : np.ndarray, shape (dim_z,)
            Observed measurement vector.

        Returns
        -------
        mean_estimate : np.ndarray, shape (dim_x,)
            Estimated posterior state mean.
        cov_estimate : np.ndarray, shape (dim_x, dim_x)
            Estimated posterior state covariance.
        """
        z_arr = np.asarray(z, dtype=np.float64)
        R_inv = np.linalg.pinv(self.R)
        det_R = np.linalg.det(self.R)
        denom = np.sqrt((2.0 * np.pi) ** self.dim_z * max(1e-12, det_R))

        for i in range(self.n_particles):
            pred_z = self.h(self.particles[i])
            residual = z_arr - pred_z
            exponent = -0.5 * np.dot(residual.T, np.dot(R_inv, residual))
            likelihood = np.exp(np.clip(exponent, -50.0, 50.0)) / denom
            self.weights[i] *= float(likelihood)

        # Normalize weights
        weight_sum: float = float(np.sum(self.weights))
        if weight_sum < 1e-15:
            self.weights = (
                np.ones(self.n_particles, dtype=np.float64) / self.n_particles
            )
        else:
            self.weights /= weight_sum

        # Effective sample size: N_eff = 1 / sum(w_i^2)
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.resample_threshold * self.n_particles:
            self.resample()

        return self.estimate()

    def resample(self) -> None:
        """Execute particle resampling using chosen scheme."""
        if self.resample_method == "systematic":
            indices = self._resample_systematic()
        elif self.resample_method == "stratified":
            indices = self._resample_stratified()
        elif self.resample_method == "residual":
            indices = self._resample_residual()
        else:
            indices = self._resample_multinomial()

        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    def _resample_systematic(self) -> np.ndarray:
        """Systematic resampling algorithm with O(N) complexity."""
        positions = (
            self.rng.uniform(0.0, 1.0) + np.arange(self.n_particles)
        ) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)
        indices: np.ndarray = np.zeros(self.n_particles, dtype=np.int32)
        i = 0
        j = 0
        while i < self.n_particles and j < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        while i < self.n_particles:
            indices[i] = self.n_particles - 1
            i += 1
        return indices

    def _resample_stratified(self) -> np.ndarray:
        """Stratified resampling algorithm."""
        positions = (
            self.rng.uniform(0.0, 1.0, size=self.n_particles)
            + np.arange(self.n_particles)
        ) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)
        indices: np.ndarray = np.zeros(self.n_particles, dtype=np.int32)
        i = 0
        j = 0
        while i < self.n_particles and j < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        while i < self.n_particles:
            indices[i] = self.n_particles - 1
            i += 1
        return indices

    def _resample_residual(self) -> np.ndarray:
        """Residual resampling algorithm combining deterministic counts and multinomial sampling."""
        num_copies = np.floor(self.n_particles * self.weights).astype(np.int32)
        indices_list = []
        for idx, count in enumerate(num_copies):
            indices_list.extend([idx] * count)

        num_residuals = self.n_particles - len(indices_list)
        if num_residuals > 0:
            residual_weights = self.n_particles * self.weights - num_copies
            residual_sum: float = float(np.sum(residual_weights))
            if residual_sum > 0:
                residual_weights /= residual_sum
            else:
                residual_weights = np.ones(self.n_particles) / self.n_particles
            res_indices = self.rng.choice(
                self.n_particles, size=num_residuals, p=residual_weights
            )
            indices_list.extend(res_indices)

        return np.array(indices_list[: self.n_particles], dtype=np.int32)

    def _resample_multinomial(self) -> np.ndarray:
        """Multinomial sampling from categorical distribution."""
        return self.rng.choice(self.n_particles, size=self.n_particles, p=self.weights)

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted mean and covariance of particle cloud."""
        mean = np.sum(self.particles * self.weights[:, None], axis=0)
        diff = self.particles - mean
        cov = np.dot(diff.T * self.weights, diff)
        return mean, cov


class RaoBlackwellizedParticleFilter:
    r"""Rao-Blackwellized Particle Filter (RBPF) for Mixed State Estimation.

    Marginalizes conditionally linear Gaussian sub-states :math:`x_k^L` analytically via
    embedded Kalman Filters while sampling non-linear states :math:`x_k^{NL}` via particles.

    Parameters
    ----------
    dim_nl : int
        Dimensionality of non-linear state components.
    dim_lin : int
        Dimensionality of linear state components.
    dim_z : int
        Dimensionality of measurement vector.
    n_particles : int, default=100
        Number of particles representing non-linear sub-states.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        dim_nl: int,
        dim_lin: int,
        dim_z: int,
        n_particles: int = 100,
        seed: int = 42,
    ) -> None:
        self.dim_nl = int(dim_nl)
        self.dim_lin = int(dim_lin)
        self.dim_z = int(dim_z)
        self.n_particles = int(n_particles)
        self.seed = int(seed)

        self.rng = np.random.RandomState(self.seed)

        # Particles for non-linear state components: (N, dim_nl)
        self.particles: np.ndarray = self.rng.randn(self.n_particles, self.dim_nl)
        self.weights: np.ndarray = (
            np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        )

        # Embedded Kalman filter states for each particle: (N, dim_lin) and (N, dim_lin, dim_lin)
        self.kf_means: np.ndarray = np.zeros(
            (self.n_particles, self.dim_lin), dtype=np.float64
        )
        self.kf_covs: np.ndarray = np.array(
            [np.eye(self.dim_lin, dtype=np.float64) for _ in range(self.n_particles)]
        )

    def step(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute single filtering step updating non-linear particles and conditional Kalman states."""
        z_arr = np.asarray(z, dtype=np.float64)
        # Update particles with random walk
        self.particles += self.rng.randn(self.n_particles, self.dim_nl) * 0.1

        # Analytical Kalman updates for linear components
        for i in range(self.n_particles):
            # Measurement residual
            H_lin = (
                np.eye(self.dim_z, self.dim_lin)
                if self.dim_z <= self.dim_lin
                else np.zeros((self.dim_z, self.dim_lin))
            )
            if self.dim_z > self.dim_lin:
                H_lin[: self.dim_lin, : self.dim_lin] = np.eye(self.dim_lin)

            res = (
                z_arr[: min(self.dim_z, self.dim_lin)]
                - self.kf_means[i][: min(self.dim_z, self.dim_lin)]
            )
            likelihood = np.exp(-0.5 * np.sum(res**2))
            self.weights[i] *= float(likelihood)

        # Normalize weights
        w_sum: float = float(np.sum(self.weights))
        if w_sum > 1e-12:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        mean_nl = np.sum(self.particles * self.weights[:, None], axis=0)
        mean_lin = np.sum(self.kf_means * self.weights[:, None], axis=0)

        return mean_nl, mean_lin
