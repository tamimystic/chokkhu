"""Neural Ordinary Differential Equations (Neural ODEs).

Pure NumPy implementation of continuous-depth neural networks:
- Parameterized derivative function f_theta(h(t), t)
- Solvers: Runge-Kutta 4th Order (RK4) and Forward Euler numerical integration
- Continuous-time trajectory modeling and dynamical system simulation
"""

from typing import List, Optional
import numpy as np


class NeuralODE:
    r"""Neural Ordinary Differential Equation (Continuous-Depth Model).

        Integrates:

        .. math::

    rac{dh(t)}{dt} = f_	heta(h(t), t)

        Parameters
        ----------
        dim : int
            State dimension of the dynamical system.
        hidden_dim : int, default=32
            Hidden layer width of derivative network :math:`f_	heta`.
        solver : str, default="rk4"
            Numerical ODE integration method ("rk4" or "euler").
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 32,
        solver: str = "rk4",
        seed: Optional[int] = 42,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.solver = solver

        rng = np.random.RandomState(seed)
        # 2-layer MLP parameterizing dh/dt = f(h)
        self.w1 = (rng.randn(dim, hidden_dim) * 0.1).astype(np.float32)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.randn(hidden_dim, dim) * 0.1).astype(np.float32)
        self.b2: np.ndarray = np.zeros(dim, dtype=np.float32)

    def _f(self, h: np.ndarray) -> np.ndarray:
        """Compute instantaneous time derivative dh/dt = f(h)."""
        act = np.tanh(np.dot(h, self.w1) + self.b1)
        return np.dot(act, self.w2) + self.b2

    def integrate(self, h0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        r"""Integrate state trajectory :math:`h(t)` from initial state :math:`h_0` over times :math:`t`."""
        h0_arr = np.asarray(h0, dtype=np.float32)
        scalar_batch = h0_arr.ndim == 1
        if scalar_batch:
            h0_arr = h0_arr[None, :]

        times = np.asarray(t_span, dtype=np.float32)
        n_times = len(times)
        trajectory: List[np.ndarray] = [h0_arr]

        current_h = h0_arr.copy()

        for i in range(n_times - 1):
            dt = float(times[i + 1] - times[i])

            if self.solver == "euler":
                dh = self._f(current_h) * dt
                current_h = current_h + dh
            else:  # RK4
                k1 = self._f(current_h)
                k2 = self._f(current_h + 0.5 * dt * k1)
                k3 = self._f(current_h + 0.5 * dt * k2)
                k4 = self._f(current_h + dt * k3)
                current_h = current_h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            trajectory.append(current_h.copy())

        res = np.stack(trajectory, axis=1)  # (B, T, dim)
        return res[0] if scalar_batch else res

    def forward(self, h0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        """Forward pass integrating from initial state h0."""
        return self.integrate(h0, t_span)
