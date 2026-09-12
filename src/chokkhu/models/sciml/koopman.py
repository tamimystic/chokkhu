"""Dynamic Mode Decomposition (DMD) & Koopman Operator Theory in Pure NumPy/SciPy.

References:
- Schmid (2010): "Dynamic Mode Decomposition of Numerical and Experimental Data" (JFM).
- Williams, Kevrekidis, Rowley (2015): "A Data-Driven Approximation of the Koopman Operator" (J Nonlinear Sci).
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class DynamicModeDecomposition:
    """Exact Dynamic Mode Decomposition (DMD) for Non-Linear Dynamical Systems.

    Decomposes spatiotemporal snapshot data into spatial coherent modes and linear temporal frequencies.
    """

    def __init__(self, rank: Optional[int] = None, dt: float = 1.0) -> None:
        self.rank = int(rank) if rank is not None else None
        self.dt = float(dt)
        self.modes: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.amplitudes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DynamicModeDecomposition":
        """Fit Exact DMD from snapshot matrix X of shape [spatial_dim, num_snapshots]."""
        snapshots = np.asarray(X, dtype=np.complex128)
        n, m = snapshots.shape

        X1 = snapshots[:, :-1]
        X2 = snapshots[:, 1:]

        # 1. Economy SVD of X1
        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        V = Vt.conj().T

        r = self.rank if self.rank is not None else min(n, m - 1)
        r = max(1, min(r, len(s)))

        Ur = U[:, :r]
        Vr = V[:, :r]

        # 2. Low-rank projection A_tilde = Ur^* X2 Vr Sr^{-1}
        Sr_inv = np.diag(1.0 / s[:r])
        A_tilde = Ur.conj().T @ X2 @ Vr @ Sr_inv

        # 3. Eigen-decomposition of A_tilde
        eigenvals, W = np.linalg.eig(A_tilde)
        self.eigenvalues = eigenvals

        # 4. Exact DMD Modes Phi = X2 Vr Sr^{-1} W
        self.modes = X2 @ Vr @ Sr_inv @ W

        # 5. Continuous-time frequencies
        self.omega = np.log(self.eigenvalues) / self.dt

        # 6. Mode amplitudes b from initial condition x0
        x0 = snapshots[:, 0]
        self.amplitudes = np.linalg.pinv(self.modes) @ x0

        return self

    def reconstruct(self, num_steps: int) -> np.ndarray:
        """Reconstruct/forecast spatiotemporal trajectory across num_steps."""
        if self.modes is None or self.omega is None or self.amplitudes is None:
            raise RuntimeError("DMD model must be fitted before reconstruct()")

        t = np.arange(num_steps) * self.dt
        # exp(omega_j * t) matrix: [r, num_steps]
        dynamics: np.ndarray = np.zeros(
            (len(self.omega), num_steps), dtype=np.complex128
        )
        for j, om in enumerate(self.omega):
            dynamics[j, :] = np.exp(om * t)

        recon = self.modes @ np.diag(self.amplitudes) @ dynamics
        return np.real(recon)


class ExtendedDMD:
    """Extended Dynamic Mode Decomposition (EDMD) for Non-Linear Koopman Operator Approximation."""

    def __init__(self, polynomial_degree: int = 2) -> None:
        self.polynomial_degree = int(polynomial_degree)
        self.K: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None

    def _lift(self, X: np.ndarray) -> np.ndarray:
        """Lift input states into polynomial dictionary observables Psi(x)."""
        x_arr = np.asarray(X, dtype=np.float64)
        dim, n_samples = x_arr.shape
        features = [np.ones((1, n_samples), dtype=np.float64), x_arr]

        if self.polynomial_degree >= 2:
            # Pairwise second-order cross-terms
            cross_terms = []
            for i in range(dim):
                for j in range(i, dim):
                    cross_terms.append((x_arr[i] * x_arr[j])[np.newaxis, :])
            if cross_terms:
                features.append(np.vstack(cross_terms))

        return np.vstack(features)

    def fit(self, X: np.ndarray) -> "ExtendedDMD":
        """Fit Koopman operator K on state snapshots X of shape [dim, num_snapshots]."""
        x_arr = np.asarray(X, dtype=np.float64)
        dim = x_arr.shape[0]

        X1 = x_arr[:, :-1]
        X2 = x_arr[:, 1:]

        Psi_X1 = self._lift(X1)
        Psi_X2 = self._lift(X2)

        # Koopman operator approximation K = Psi_X2 @ pinv(Psi_X1)
        self.K = Psi_X2 @ np.linalg.pinv(Psi_X1)

        # State reconstruction projection C: x = C @ Psi(x)
        self.C = np.zeros((dim, Psi_X1.shape[0]), dtype=np.float64)
        for d in range(dim):
            self.C[d, d + 1] = 1.0

        return self

    def predict_step(self, x_curr: np.ndarray) -> np.ndarray:
        """Predict next step state x_{t+1} via lifted linear Koopman forward step."""
        if self.K is None or self.C is None:
            raise RuntimeError("ExtendedDMD must be fitted before predict_step()")

        x_col = np.asarray(x_curr, dtype=np.float64).reshape(-1, 1)
        psi_x = self._lift(x_col)
        psi_next = self.K @ psi_x
        x_next = self.C @ psi_next
        return x_next.squeeze()
