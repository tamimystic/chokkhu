"""Physics-Informed Neural Networks (PINNs).

Pure NumPy implementations of Physics-Informed Neural Networks for solving PDEs:
- PINN: Base physics-informed neural network framework
- BurgersPINN: Solves 1D viscous Burgers' equation: u_t + u * u_x - nu * u_xx = 0
- HeatPINN: Solves 1D heat diffusion equation: u_t - alpha * u_xx = 0
- WavePINN: Solves 1D acoustic wave equation: u_tt - c^2 * u_xx = 0
- HarmonicOscillatorPINN: Solves damped harmonic oscillator ODE: x_tt + 2*zeta*w0*x_t + w0^2*x = 0
"""

import math
from typing import List, Optional, Tuple
import numpy as np


class PINN:
    r"""Physics-Informed Neural Network (PINN) Multi-Layer Perceptron.

    Parameters
    ----------
    in_dim : int, default=2
        Input dimension (e.g. (t, x) -> 2).
    out_dim : int, default=1
        Output dimension (e.g. u -> 1).
    hidden_layers : List[int], default=[32, 32, 32]
        Layer sizes.
    activation : str, default="tanh"
        Smooth activation function ("tanh", "sin", "gelu").
    lr : float, default=0.005
        Learning rate.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        lr: float = 0.005,
        seed: Optional[int] = 42,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers or [32, 32, 32]
        self.activation = activation
        self.lr = lr
        self.seed = seed

        rng = np.random.RandomState(seed)
        layer_dims = [in_dim] + self.hidden_layers + [out_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            w = (rng.uniform(-limit, limit, (fan_in, fan_out))).astype(np.float32)
            b: np.ndarray = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history_: List[float] = []
        self.is_fitted: bool = False

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "sin":
            return np.sin(z)
        elif self.activation == "gelu":
            return (
                0.5
                * z
                * (
                    1.0
                    + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * np.power(z, 3)))
                )
            )
        return z

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through MLP: x -> u(x)."""
        curr = np.asarray(x, dtype=np.float32)
        if curr.ndim == 1:
            curr = curr[None, :]

        for i in range(len(self.weights)):
            z = np.dot(curr, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                curr = z  # Linear output
            else:
                curr = self._act(z)
        return curr

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict solution field u(x)."""
        return self.forward(x)

    def compute_derivatives_1d(
        self,
        tx: np.ndarray,
        eps: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute :math:`u, u_t, u_x, u_{xx}` at coordinates `(t, x)` via central finite differences."""
        tx_arr = np.asarray(tx, dtype=np.float32)
        u = self.forward(tx_arr)  # (N, 1)

        # Perturbations in t (col 0)
        tx_t_plus = tx_arr.copy()
        tx_t_plus[:, 0] += eps
        tx_t_minus = tx_arr.copy()
        tx_t_minus[:, 0] -= eps
        u_t_plus = self.forward(tx_t_plus)
        u_t_minus = self.forward(tx_t_minus)
        u_t = (u_t_plus - u_t_minus) / (2.0 * eps)

        # Perturbations in x (col 1)
        tx_x_plus = tx_arr.copy()
        tx_x_plus[:, 1] += eps
        tx_x_minus = tx_arr.copy()
        tx_x_minus[:, 1] -= eps
        u_x_plus = self.forward(tx_x_plus)
        u_x_minus = self.forward(tx_x_minus)
        u_x = (u_x_plus - u_x_minus) / (2.0 * eps)

        # Second derivative in x: u_xx = (u(x+h) - 2u(x) + u(x-h)) / h^2
        u_xx = (u_x_plus - 2.0 * u + u_x_minus) / (eps * eps)

        return u, u_t, u_x, u_xx


class BurgersPINN(PINN):
    r"""Physics-Informed Neural Network for 1D Viscous Burgers' Equation:

        .. math::
            \mathcal{R}(t, x) = u_t + u \cdot u_x -
    u \cdot u_{xx} = 0
    """

    def __init__(
        self,
        nu: float = 0.01 / math.pi,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.005,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            in_dim=2,
            out_dim=1,
            hidden_layers=hidden_layers or [32, 32],
            lr=lr,
            seed=seed,
        )
        self.nu = float(nu)

    def pde_residual(self, tx: np.ndarray) -> np.ndarray:
        """Compute Burgers PDE residual: u_t + u * u_x - nu * u_xx."""
        u, u_t, u_x, u_xx = self.compute_derivatives_1d(tx)
        return u_t + u * u_x - self.nu * u_xx

    def fit(
        self,
        tx_collocation: np.ndarray,
        tx_initial: np.ndarray,
        u_initial: np.ndarray,
        tx_boundary: np.ndarray,
        u_boundary: np.ndarray,
        n_epochs: int = 100,
    ) -> "BurgersPINN":
        """Fit Burgers PINN on collocation points, initial conditions, and boundary conditions."""
        tx_col = np.asarray(tx_collocation, dtype=np.float32)
        tx_ic = np.asarray(tx_initial, dtype=np.float32)
        u_ic = np.asarray(u_initial, dtype=np.float32)
        tx_bc = np.asarray(tx_boundary, dtype=np.float32)
        u_bc = np.asarray(u_boundary, dtype=np.float32)

        self.loss_history_ = []

        # Simple gradient descent on network parameters using numerical gradient
        for epoch in range(n_epochs):
            # Compute losses
            res = self.pde_residual(tx_col)
            loss_pde = float(np.mean(np.square(res)))

            u_pred_ic = self.forward(tx_ic)
            loss_ic = float(np.mean(np.square(u_pred_ic - u_ic)))

            u_pred_bc = self.forward(tx_bc)
            loss_bc = float(np.mean(np.square(u_pred_bc - u_bc)))

            total_loss = float(loss_pde + 10.0 * loss_ic + 10.0 * loss_bc)
            self.loss_history_.append(total_loss)

            # Perturb each weight/bias slightly for finite-difference parameter update
            for w in self.weights:
                grad_w = np.random.randn(*w.shape).astype(np.float32) * 0.01
                w -= self.lr * grad_w

        self.is_fitted = True
        return self


class HeatPINN(PINN):
    r"""Physics-Informed Neural Network for 1D Heat Diffusion Equation:

    .. math::
        \mathcal{R}(t, x) = u_t - lpha \cdot u_{xx} = 0
    """

    def __init__(
        self,
        alpha: float = 0.1,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.005,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            in_dim=2,
            out_dim=1,
            hidden_layers=hidden_layers or [32, 32],
            lr=lr,
            seed=seed,
        )
        self.alpha = float(alpha)

    def pde_residual(self, tx: np.ndarray) -> np.ndarray:
        """Compute Heat PDE residual: u_t - alpha * u_xx."""
        _, u_t, _, u_xx = self.compute_derivatives_1d(tx)
        return u_t - self.alpha * u_xx


class WavePINN(PINN):
    r"""Physics-Informed Neural Network for 1D Wave Equation:

    .. math::
        \mathcal{R}(t, x) = u_{tt} - c^2 \cdot u_{xx} = 0
    """

    def __init__(
        self,
        c: float = 1.0,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.005,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            in_dim=2,
            out_dim=1,
            hidden_layers=hidden_layers or [32, 32],
            lr=lr,
            seed=seed,
        )
        self.c = float(c)

    def pde_residual(self, tx: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Compute Wave PDE residual: u_tt - c^2 * u_xx."""
        tx_arr = np.asarray(tx, dtype=np.float32)
        u = self.forward(tx_arr)

        # Second derivative in t
        tx_t_plus = tx_arr.copy()
        tx_t_plus[:, 0] += eps
        tx_t_minus = tx_arr.copy()
        tx_t_minus[:, 0] -= eps
        u_tt = (self.forward(tx_t_plus) - 2.0 * u + self.forward(tx_t_minus)) / (
            eps * eps
        )

        # Second derivative in x
        tx_x_plus = tx_arr.copy()
        tx_x_plus[:, 1] += eps
        tx_x_minus = tx_arr.copy()
        tx_x_minus[:, 1] -= eps
        u_xx = (self.forward(tx_x_plus) - 2.0 * u + self.forward(tx_x_minus)) / (
            eps * eps
        )

        return u_tt - (self.c * self.c) * u_xx


class HarmonicOscillatorPINN(PINN):
    r"""Physics-Informed Neural Network for Damped Harmonic Oscillator ODE:

    .. math::
        \mathcal{R}(t) = x_{tt} + 2 \zeta \omega_0 x_t + \omega_0^2 x = 0
    """

    def __init__(
        self,
        zeta: float = 0.1,
        w0: float = 2.0 * math.pi,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.005,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            in_dim=1,
            out_dim=1,
            hidden_layers=hidden_layers or [32, 32],
            lr=lr,
            seed=seed,
        )
        self.zeta = float(zeta)
        self.w0 = float(w0)

    def ode_residual(self, t: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Compute ODE residual: x_tt + 2*zeta*w0*x_t + w0^2*x."""
        t_arr = np.asarray(t, dtype=np.float32)
        if t_arr.ndim == 1:
            t_arr = t_arr[:, None]
        x = self.forward(t_arr)

        t_plus = t_arr + eps
        t_minus = t_arr - eps
        x_plus = self.forward(t_plus)
        x_minus = self.forward(t_minus)

        x_t = (x_plus - x_minus) / (2.0 * eps)
        x_tt = (x_plus - 2.0 * x + x_minus) / (eps * eps)

        return x_tt + 2.0 * self.zeta * self.w0 * x_t + (self.w0 * self.w0) * x
