"""Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for Non-Linear State Estimation.

Formulated from first principles in pure NumPy and SciPy.
Provides optimal Bayesian filtering, Jacobian linearization, and Unscented Transform sigma-point
propagation for non-linear dynamical systems and multi-sensor fusion.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


class ExtendedKalmanFilter:
    r"""Extended Kalman Filter (EKF) for Non-Linear Dynamical Systems.

    State transition model:
    x_t = f(x_{t-1}, u_t) + w_t, \quad w_t \sim \mathcal{N}(0, Q)

    Measurement model:
    z_t = h(x_t) + v_t, \quad v_t \sim \mathcal{N}(0, R)

    Parameters
    ----------
    dim_x : int
        State dimension $L$.
    dim_z : int
        Measurement dimension $M$.
    f : Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        Non-linear state transition function $f(x, u)$.
    h : Callable[[np.ndarray], np.ndarray]
        Non-linear measurement function $h(x)$.
    Q : Optional[np.ndarray] of shape (dim_x, dim_x)
        Process noise covariance matrix.
    R : Optional[np.ndarray] of shape (dim_z, dim_z)
        Measurement noise covariance matrix.
    P : Optional[np.ndarray] of shape (dim_x, dim_x)
        Initial state estimation error covariance matrix.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ) -> None:
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.f = f
        self.h = h

        self.Q = np.eye(self.dim_x) if Q is None else np.asarray(Q, dtype=float)
        self.R = np.eye(self.dim_z) if R is None else np.asarray(R, dtype=float)
        self.P = np.eye(self.dim_x) if P is None else np.asarray(P, dtype=float)
        self.x: np.ndarray = np.zeros(self.dim_x, dtype=float)

    def _numerical_jacobian(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical Jacobian matrix of vector function at x."""
        n = x.shape[0]
        f0 = func(x)
        m = f0.shape[0]
        J = np.zeros((m, n), dtype=float)

        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = func(x_pert)
            J[:, j] = (f_pert - f0) / eps

        return J

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Execute state prediction step.

        Parameters
        ----------
        u : Optional[np.ndarray], default=None
            Control input vector.

        Returns
        -------
        x_pred : np.ndarray
            Predicted state mean $x_{t|t-1}$.
        P_pred : np.ndarray
            Predicted error covariance $P_{t|t-1}$.
        """
        # Linearize state transition at current state
        F = self._numerical_jacobian(lambda state: self.f(state, u), self.x)

        # Propagate mean and covariance
        self.x = self.f(self.x, u)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute measurement update step with observation z.

        Parameters
        ----------
        z : np.ndarray of shape (dim_z,)
            Observed measurement vector $z_t$.

        Returns
        -------
        x_updated : np.ndarray
            Updated state estimate $x_{t|t}$.
        P_updated : np.ndarray
            Updated covariance $P_{t|t}$.
        """
        z_vec = np.asarray(z, dtype=float).flatten()
        H = self._numerical_jacobian(self.h, self.x)

        # Innovation / measurement residual
        y = z_vec - self.h(self.x)

        # Innovation covariance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R

        # Near-optimal Kalman gain
        try:
            K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        except np.linalg.LinAlgError:
            K = np.dot(self.P, np.dot(H.T, np.linalg.pinv(S)))

        # State and covariance update
        self.x = self.x + np.dot(K, y)
        I_KH = np.eye(self.dim_x) - np.dot(K, H)
        # Joseph form covariance update for numerical symmetry & stability
        self.P = np.dot(I_KH, np.dot(self.P, I_KH.T)) + np.dot(K, np.dot(self.R, K.T))

        return self.x.copy(), self.P.copy()

    def filter(
        self,
        observations: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Filter an entire sequence of observations.

        Parameters
        ----------
        observations : np.ndarray of shape (T, dim_z)
            Sequence of measurements.
        controls : Optional[np.ndarray] of shape (T, dim_u), default=None
            Sequence of control inputs.

        Returns
        -------
        state_estimates : np.ndarray of shape (T, dim_x)
            Filtered sequence of state estimates.
        """
        T = observations.shape[0]
        estimates = np.zeros((T, self.dim_x), dtype=float)

        for t in range(T):
            u_t = None if controls is None else controls[t]
            self.predict(u=u_t)
            x_est, _ = self.update(observations[t])
            estimates[t] = x_est

        return estimates


class UnscentedKalmanFilter:
    r"""Unscented Kalman Filter (UKF) with Deterministic Sigma-Point Sampling.

    Propagates probability distributions through non-linear transformations without
    linearization errors using the Julier-Uhlmann Unscented Transform.

    Parameters
    ----------
    dim_x : int
        State dimension $L$.
    dim_z : int
        Measurement dimension $M$.
    f : Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        Non-linear state transition function $f(x, u)$.
    h : Callable[[np.ndarray], np.ndarray]
        Non-linear measurement function $h(x)$.
    alpha : float, default=1e-3
        Primary scaling parameter controlling sigma point spread.
    beta : float, default=2.0
        Prior knowledge parameter ($\beta=2$ optimal for Gaussian distributions).
    kappa : float, default=0.0
        Secondary scaling parameter.
    Q : Optional[np.ndarray] of shape (dim_x, dim_x)
        Process noise covariance.
    R : Optional[np.ndarray] of shape (dim_z, dim_z)
        Measurement noise covariance.
    P : Optional[np.ndarray] of shape (dim_x, dim_x)
        Initial state covariance.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ) -> None:
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.f = f
        self.h = h

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        L = self.dim_x
        self.lam = (self.alpha**2) * (L + self.kappa) - L
        self.c = L + self.lam

        # Generate weights
        self.Wm = np.full(2 * L + 1, 0.5 / self.c)
        self.Wc = np.full(2 * L + 1, 0.5 / self.c)
        self.Wm[0] = self.lam / self.c
        self.Wc[0] = self.lam / self.c + (1.0 - self.alpha**2 + self.beta)

        self.Q = np.eye(L) if Q is None else np.asarray(Q, dtype=float)
        self.R = np.eye(self.dim_z) if R is None else np.asarray(R, dtype=float)
        self.P = np.eye(L) if P is None else np.asarray(P, dtype=float)
        self.x: np.ndarray = np.zeros(L, dtype=float)

        self.sigma_points: Optional[np.ndarray] = None

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Sample 2L + 1 symmetric sigma points around mean x with covariance P."""
        L = self.dim_x
        sigmas: np.ndarray = np.zeros((2 * L + 1, L), dtype=float)
        sigmas[0] = x

        # Matrix square root via Cholesky decomposition
        try:
            sqrt_mat = np.linalg.cholesky(self.c * (P + 1e-9 * np.eye(L)))
        except np.linalg.LinAlgError:
            # Fallback to eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(self.c * (P + 1e-9 * np.eye(L)))
            eigvals = np.maximum(eigvals, 1e-9)
            sqrt_mat = eigvecs.dot(np.diag(np.sqrt(eigvals)))

        for i in range(L):
            sigmas[i + 1] = x + sqrt_mat[:, i]
            sigmas[i + 1 + L] = x - sqrt_mat[:, i]

        return sigmas

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state distribution using Unscented Transform."""
        L = self.dim_x
        # 1. Generate sigma points at x_{t-1|t-1}
        sigmas = self._generate_sigma_points(self.x, self.P)

        # 2. Propagate through dynamic model f
        sigmas_f: np.ndarray = np.zeros((2 * L + 1, L), dtype=float)
        for i in range(2 * L + 1):
            sigmas_f[i] = self.f(sigmas[i], u)

        # 3. Predict mean
        x_pred = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        # 4. Predict covariance
        diff_x = sigmas_f - x_pred[None, :]
        P_pred = np.dot(diff_x.T, self.Wc[:, None] * diff_x) + self.Q

        self.x = x_pred
        self.P = P_pred
        self.sigma_points = sigmas_f
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state estimate with measurement z using Unscented Transform."""
        L = self.dim_x
        M = self.dim_z
        z_vec = np.asarray(z, dtype=float).flatten()

        sigmas_f = (
            self.sigma_points
            if self.sigma_points is not None
            else self._generate_sigma_points(self.x, self.P)
        )

        # 1. Propagate sigma points through measurement function h
        sigmas_h: np.ndarray = np.zeros((2 * L + 1, M), dtype=float)
        for i in range(2 * L + 1):
            sigmas_h[i] = self.h(sigmas_f[i])

        # 2. Predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)

        # 3. Measurement innovation covariance S and Cross-covariance P_xz
        diff_z = sigmas_h - z_pred[None, :]
        diff_x = sigmas_f - self.x[None, :]

        P_zz = np.dot(diff_z.T, self.Wc[:, None] * diff_z) + self.R
        P_xz = np.dot(diff_x.T, self.Wc[:, None] * diff_z)

        # 4. Kalman gain
        try:
            K = np.dot(P_xz, np.linalg.inv(P_zz))
        except np.linalg.LinAlgError:
            K = np.dot(P_xz, np.linalg.pinv(P_zz))

        # 5. State and covariance update
        y = z_vec - z_pred
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(P_zz, K.T))

        return self.x.copy(), self.P.copy()
