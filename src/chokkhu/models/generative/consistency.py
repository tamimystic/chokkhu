"""Sovereign Consistency Models, Optimal Transport Flow Matching & Reflow in Pure NumPy & SciPy.

References:
- Song et al. (2023): "Consistency Models" (ICML 2023).
- Lipman et al. (2023): "Flow Matching for Generative Modeling" (ICLR 2023).
- Liu et al. (2022): "Flow Straight and Fast: Learning to Generate and Transfer Data with Flow".
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment


class ConsistencyMLP:
    """Multi-Layer Perceptron parameterizing the intermediate consistency vector field F_theta(x, t)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        time_embed_dim: int = 16,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.time_embed_dim = time_embed_dim
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Sinusoidal time embedding frequencies
        half_dim = time_embed_dim // 2
        self.freqs: np.ndarray = np.exp(
            -np.log(10000.0) * np.arange(0, half_dim) / max(half_dim, 1)
        ).astype(np.float32)

        # Weight layers
        dims = [input_dim + time_embed_dim] + self.hidden_dims + [input_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(dims) - 1):
            w = self.rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b: np.ndarray = np.zeros((1, dims[i + 1]), dtype=np.float32)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b)

    def _embed_time(self, t: np.ndarray) -> np.ndarray:
        """Compute sinusoidal time embeddings."""
        t_arr = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        args = t_arr * self.freqs.reshape(1, -1) * 2.0 * np.pi
        sin_emb = np.sin(args)
        cos_emb = np.cos(args)
        return np.concatenate([sin_emb, cos_emb], axis=-1).astype(np.float32)

    def forward(self, x: np.ndarray, t: Union[np.ndarray, float]) -> np.ndarray:
        """Forward pass evaluating F_theta(x, t)."""
        x_arr = np.asarray(x, dtype=np.float32)
        if isinstance(t, (int, float)):
            t_arr: np.ndarray = np.full((x_arr.shape[0],), float(t), dtype=np.float32)
        elif isinstance(t, np.ndarray) and t.ndim == 0:
            t_arr = np.full((x_arr.shape[0],), float(t.item()), dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.shape[0] != x_arr.shape[0]:
                t_arr = np.full((x_arr.shape[0],), float(t_arr[0]), dtype=np.float32)

        t_emb = self._embed_time(t_arr)
        h = np.concatenate([x_arr, t_emb], axis=-1)

        for i in range(len(self.weights) - 1):
            h = np.dot(h, self.weights[i]) + self.biases[i]
            # SiLU / Swish activation: h / (1 + exp(-h))
            h = h / (1.0 + np.exp(-np.clip(h, -15.0, 15.0)))

        out = np.dot(h, self.weights[-1]) + self.biases[-1]
        return out.astype(np.float32)

    def get_params(self) -> List[np.ndarray]:
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w.copy(), b.copy()])
        return params

    def set_params(self, params: List[np.ndarray]) -> None:
        idx = 0
        for i in range(len(self.weights)):
            self.weights[i] = params[idx].copy()
            self.biases[i] = params[idx + 1].copy()
            idx += 2


class ConsistencyModel:
    """Sovereign Consistency Model (Song et al., 2023) in pure NumPy.

    Guarantees self-consistency f_theta(x_t, t) = f_theta(x_s, s) = x_0 along the Probability Flow ODE trajectory,
    satisfying the exact boundary condition f_theta(x, eps) = x.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_scales: int = 18,
        ema_decay: float = 0.95,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.num_scales = int(num_scales)
        self.ema_decay = float(ema_decay)
        self.lr = float(lr)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Online network theta
        self.net = ConsistencyMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            random_state=random_state,
        )

        # Target network theta_ema (Exponential Moving Average)
        self.target_net = ConsistencyMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            random_state=random_state,
        )
        self.target_net.set_params(self.net.get_params())

        # Optimizer state (Adam)
        self.m_w = [np.zeros_like(w) for w in self.net.weights]
        self.v_w = [np.zeros_like(w) for w in self.net.weights]
        self.m_b = [np.zeros_like(b) for b in self.net.biases]
        self.v_b = [np.zeros_like(b) for b in self.net.biases]
        self.t_step = 0

        # Discretization time scale schedule
        self.time_schedule = self._build_time_schedule()

    def _build_time_schedule(self) -> np.ndarray:
        """Karras geometric discretization schedule between sigma_min and sigma_max."""
        n = self.num_scales
        indices: np.ndarray = np.arange(n, dtype=np.float32)
        t = (
            self.sigma_min ** (1.0 / self.rho)
            + (indices / max(n - 1, 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho
        return t.astype(np.float32)

    def c_skip(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Skip connection coefficient c_skip(t) = sigma_data^2 / ((t - sigma_min)^2 + sigma_data^2)."""
        t_arr = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        eps = self.sigma_min
        s_data = self.sigma_data
        return (s_data**2) / ((t_arr - eps) ** 2 + s_data**2)

    def c_out(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Output coefficient c_out(t) = sigma_data * (t - sigma_min) / sqrt(sigma_data^2 + t^2)."""
        t_arr = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        eps = self.sigma_min
        s_data = self.sigma_data
        return (s_data * (t_arr - eps)) / np.sqrt(s_data**2 + t_arr**2)

    def forward(
        self,
        x: np.ndarray,
        t: Union[float, np.ndarray],
        use_target: bool = False,
    ) -> np.ndarray:
        """Evaluate consistency function f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)."""
        x_arr = np.asarray(x, dtype=np.float32)
        c_sk = self.c_skip(t)
        c_o = self.c_out(t)

        net = self.target_net if use_target else self.net
        f_val = net.forward(x_arr, t)

        return (c_sk * x_arr + c_o * f_val).astype(np.float32)

    def train_step(self, x: np.ndarray) -> float:
        """Perform one step of Consistency Training (CT) on clean data x."""
        x_arr = np.asarray(x, dtype=np.float32)
        n_samples = x_arr.shape[0]

        # Sample adjacent discrete timesteps t_n, t_{n+1}
        # Choose n uniformly from 0 to num_scales - 2
        n_idx = self.rng.randint(0, max(self.num_scales - 1, 1), size=(n_samples,))
        t_n = self.time_schedule[n_idx]
        t_np1 = self.time_schedule[n_idx + 1]

        # Add Gaussian noise
        z = self.rng.randn(*x_arr.shape).astype(np.float32)
        x_tn = x_arr + t_n.reshape(-1, 1) * z
        x_tnp1 = x_arr + t_np1.reshape(-1, 1) * z

        # Target output from target network (EMA) evaluated at (x_tn, t_n)
        target = self.forward(x_tn, t_n, use_target=True)

        # Online network forward pass at (x_tnp1, t_np1)
        c_sk = self.c_skip(t_np1)
        c_o = self.c_out(t_np1)

        t_emb = self.net._embed_time(t_np1)
        h_in = np.concatenate([x_tnp1, t_emb], axis=-1)

        activations = [h_in]
        linear_outputs = []

        h = h_in
        for i in range(len(self.net.weights) - 1):
            z_layer = np.dot(h, self.net.weights[i]) + self.net.biases[i]
            linear_outputs.append(z_layer)
            sig = 1.0 / (1.0 + np.exp(-np.clip(z_layer, -15.0, 15.0)))
            h = z_layer * sig
            activations.append(h)

        f_val = np.dot(h, self.net.weights[-1]) + self.net.biases[-1]
        pred_x0 = c_sk * x_tnp1 + c_o * f_val

        # Pseudo-Huber loss: d(u, v) = sqrt((u - v)^2 + c^2) - c with c=0.00054 * sqrt(d)
        diff = pred_x0 - target
        c_huber = 0.00054 * np.sqrt(self.input_dim)
        loss = float(np.mean(np.sqrt(diff**2 + c_huber**2) - c_huber))

        # Backward gradient
        grad_pred = diff / (np.sqrt(diff**2 + c_huber**2) + 1e-8)
        # Gradient wrt F_theta: grad_pred * c_o
        grad_out = (grad_pred * c_o) / (n_samples * self.input_dim)

        grad_w = []
        grad_b = []

        gw_last = np.dot(activations[-1].T, grad_out)
        gb_last = np.sum(grad_out, axis=0, keepdims=True)
        grad_w.append(gw_last)
        grad_b.append(gb_last)

        dh = np.dot(grad_out, self.net.weights[-1].T)

        for i in range(len(self.net.weights) - 2, -1, -1):
            z_layer = linear_outputs[i]
            sig = 1.0 / (1.0 + np.exp(-np.clip(z_layer, -15.0, 15.0)))
            dsilu = sig + z_layer * sig * (1.0 - sig)
            dz = dh * dsilu

            gw = np.dot(activations[i].T, dz)
            gb = np.sum(dz, axis=0, keepdims=True)
            grad_w.append(gw)
            grad_b.append(gb)

            if i > 0:
                dh = np.dot(dz, self.net.weights[i].T)

        grad_w.reverse()
        grad_b.reverse()

        # Adam optimization update
        self.t_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for i in range(len(self.net.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1.0 - beta1) * grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1.0 - beta2) * (grad_w[i] ** 2)
            m_hat = self.m_w[i] / (1.0 - beta1**self.t_step)
            v_hat = self.v_w[i] / (1.0 - beta2**self.t_step)
            self.net.weights[i] -= (self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(
                np.float32
            )

            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * (grad_b[i] ** 2)
            mb_hat = self.m_b[i] / (1.0 - beta1**self.t_step)
            vb_hat = self.v_b[i] / (1.0 - beta2**self.t_step)
            self.net.biases[i] -= (self.lr * mb_hat / (np.sqrt(vb_hat) + eps)).astype(
                np.float32
            )

        # Update EMA target network
        for i in range(len(self.net.weights)):
            self.target_net.weights[i] = (
                self.ema_decay * self.target_net.weights[i]
                + (1.0 - self.ema_decay) * self.net.weights[i]
            ).astype(np.float32)
            self.target_net.biases[i] = (
                self.ema_decay * self.target_net.biases[i]
                + (1.0 - self.ema_decay) * self.net.biases[i]
            ).astype(np.float32)

        return loss

    def sample(
        self,
        num_samples: int = 16,
        steps: int = 1,
        sigma_start: Optional[float] = None,
    ) -> np.ndarray:
        """Sample synthetic instances using 1-step or multi-step consistency sampling."""
        s_start = self.sigma_max if sigma_start is None else float(sigma_start)
        # Initial noise: x ~ N(0, sigma_max^2 I)
        x = (self.rng.randn(num_samples, self.input_dim) * s_start).astype(np.float32)

        if steps <= 1:
            # 1-step direct mapping
            return self.forward(x, s_start, use_target=True)

        # Multi-step consistency sampling across sub-sequence of time schedule
        time_seq = np.linspace(s_start, self.sigma_min, steps)

        x_hat = self.forward(x, time_seq[0], use_target=True)

        for i in range(1, steps):
            t_curr = time_seq[i]
            if t_curr <= self.sigma_min:
                break
            # Add noise back to intermediate scale
            z = self.rng.randn(num_samples, self.input_dim).astype(np.float32)
            noise_std = np.sqrt(max(t_curr**2 - self.sigma_min**2, 0.0))
            x_noisy = x_hat + noise_std * z
            x_hat = self.forward(x_noisy, t_curr, use_target=True)

        return x_hat


class OptimalTransportFlowMatching:
    """Optimal Transport Continuous Velocity Flow Matching (OT-CFM) in pure NumPy & SciPy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        time_embed_dim: int = 16,
        lr: float = 0.001,
        sigma_min: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.time_embed_dim = time_embed_dim
        self.lr = lr
        self.sigma_min = sigma_min
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.net = ConsistencyMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            time_embed_dim=time_embed_dim,
            random_state=random_state,
        )

        self.m_w = [np.zeros_like(w) for w in self.net.weights]
        self.v_w = [np.zeros_like(w) for w in self.net.weights]
        self.m_b = [np.zeros_like(b) for b in self.net.biases]
        self.v_b = [np.zeros_like(b) for b in self.net.biases]
        self.t_step = 0

    def compute_ot_pairing(
        self, x0: np.ndarray, x1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimal transport permutation via Earth Mover / Hungarian bipartite assignment."""
        # Pairwise squared Euclidean cost matrix: (N, N)
        # ||x0_i - x1_j||^2 = ||x0_i||^2 + ||x1_j||^2 - 2 <x0_i, x1_j>
        cost_matrix = (
            np.sum(x0**2, axis=1, keepdims=True)
            + np.sum(x1**2, axis=1, keepdims=True).T
            - 2.0 * np.dot(x0, x1.T)
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return x0[row_ind], x1[col_ind]

    def train_step(self, x1: np.ndarray) -> float:
        """Train flow matching with exact mini-batch Optimal Transport pairing."""
        x1_arr = np.asarray(x1, dtype=np.float32)
        n_samples = x1_arr.shape[0]

        # Sample source Gaussian noise
        x0_raw = self.rng.randn(*x1_arr.shape).astype(np.float32)

        # Optimal Transport bipartite matching
        x0_ot, x1_ot = self.compute_ot_pairing(x0_raw, x1_arr)

        # Sample time t ~ U(0, 1)
        t = self.rng.uniform(0.0, 1.0, size=(n_samples,)).astype(np.float32)
        t_col = t.reshape(-1, 1)

        # Straight-line interpolation and target velocity
        xt = (1.0 - t_col) * x0_ot + t_col * x1_ot
        ut = x1_ot - x0_ot

        # Forward pass
        t_emb = self.net._embed_time(t)
        h_in = np.concatenate([xt, t_emb], axis=-1)

        activations = [h_in]
        linear_outputs = []

        h = h_in
        for i in range(len(self.net.weights) - 1):
            z = np.dot(h, self.net.weights[i]) + self.net.biases[i]
            linear_outputs.append(z)
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            h = z * sig
            activations.append(h)

        pred_v = np.dot(h, self.net.weights[-1]) + self.net.biases[-1]
        loss = float(np.mean((pred_v - ut) ** 2))

        # Backward pass
        grad_out = (2.0 / (n_samples * self.input_dim)) * (pred_v - ut)

        grad_w = []
        grad_b = []

        gw_last = np.dot(activations[-1].T, grad_out)
        gb_last = np.sum(grad_out, axis=0, keepdims=True)
        grad_w.append(gw_last)
        grad_b.append(gb_last)

        dh = np.dot(grad_out, self.net.weights[-1].T)

        for i in range(len(self.net.weights) - 2, -1, -1):
            z = linear_outputs[i]
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            dsilu = sig + z * sig * (1.0 - sig)
            dz = dh * dsilu

            gw = np.dot(activations[i].T, dz)
            gb = np.sum(dz, axis=0, keepdims=True)
            grad_w.append(gw)
            grad_b.append(gb)

            if i > 0:
                dh = np.dot(dz, self.net.weights[i].T)

        grad_w.reverse()
        grad_b.reverse()

        # Adam optimization
        self.t_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for i in range(len(self.net.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1.0 - beta1) * grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1.0 - beta2) * (grad_w[i] ** 2)
            m_hat = self.m_w[i] / (1.0 - beta1**self.t_step)
            v_hat = self.v_w[i] / (1.0 - beta2**self.t_step)
            self.net.weights[i] -= (self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(
                np.float32
            )

            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * (grad_b[i] ** 2)
            mb_hat = self.m_b[i] / (1.0 - beta1**self.t_step)
            vb_hat = self.v_b[i] / (1.0 - beta2**self.t_step)
            self.net.biases[i] -= (self.lr * mb_hat / (np.sqrt(vb_hat) + eps)).astype(
                np.float32
            )

        return loss

    def sample(
        self,
        num_samples: int = 16,
        steps: int = 50,
        method: str = "rk4",
    ) -> np.ndarray:
        """Sample synthetic instances by integrating velocity ODE dx/dt = v_theta(x, t) from t=0 to t=1."""
        x = self.rng.randn(num_samples, self.input_dim).astype(np.float32)
        dt = 1.0 / float(steps)

        for step in range(steps):
            t = float(step) * dt

            if method == "euler":
                v = self.net.forward(x, t)
                x = x + dt * v

            elif method == "midpoint":
                v1 = self.net.forward(x, t)
                x_mid = x + 0.5 * dt * v1
                v_mid = self.net.forward(x_mid, t + 0.5 * dt)
                x = x + dt * v_mid

            elif method == "rk4":
                k1 = self.net.forward(x, t)
                k2 = self.net.forward(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self.net.forward(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self.net.forward(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(
                    f"Unknown ODE integration method: {method}. Choose from 'euler', 'midpoint', 'rk4'."
                )

        return x.astype(np.float32)


class ReflowMatching:
    """Reflow Trajectory Straightening (Liu et al., 2022) for 1-Step and 2-Step Ultra-Fast ODE Sampling."""

    def __init__(
        self,
        base_flow: Union[OptimalTransportFlowMatching, Any],
        hidden_dims: Optional[List[int]] = None,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.base_flow = base_flow
        self.input_dim = base_flow.input_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.lr = lr
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # 2-Rectified Flow model
        self.reflow_model = OptimalTransportFlowMatching(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            lr=self.lr,
            random_state=random_state,
        )

    def generate_paired_dataset(
        self,
        num_pairs: int = 200,
        ode_steps: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paired straight endpoints (x_0, x_hat_1) by simulating base ODE from noise."""
        x0 = self.rng.randn(num_pairs, self.input_dim).astype(np.float32)
        # Integrate base flow from x0
        dt = 1.0 / float(ode_steps)
        x_curr = x0.copy()

        for step in range(ode_steps):
            t = float(step) * dt
            # RK4 integration step
            k1 = self.base_flow.net.forward(x_curr, t)
            k2 = self.base_flow.net.forward(x_curr + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.base_flow.net.forward(x_curr + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.base_flow.net.forward(x_curr + dt * k3, t + dt)
            x_curr = x_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        x1_hat = x_curr.astype(np.float32)
        return x0, x1_hat

    def train_reflow_step(self, x0_paired: np.ndarray, x1_paired: np.ndarray) -> float:
        """Perform one step of straight-line 2-Reflow training on paired endpoints."""
        n_samples = x0_paired.shape[0]
        t = self.rng.uniform(0.0, 1.0, size=(n_samples,)).astype(np.float32)
        t_col = t.reshape(-1, 1)

        xt = (1.0 - t_col) * x0_paired + t_col * x1_paired
        ut = x1_paired - x0_paired

        # Forward pass on reflow model
        t_emb = self.reflow_model.net._embed_time(t)
        h_in = np.concatenate([xt, t_emb], axis=-1)

        activations = [h_in]
        linear_outputs = []

        h = h_in
        for i in range(len(self.reflow_model.net.weights) - 1):
            z = (
                np.dot(h, self.reflow_model.net.weights[i])
                + self.reflow_model.net.biases[i]
            )
            linear_outputs.append(z)
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            h = z * sig
            activations.append(h)

        pred_v = (
            np.dot(h, self.reflow_model.net.weights[-1])
            + self.reflow_model.net.biases[-1]
        )
        loss = float(np.mean((pred_v - ut) ** 2))

        # Backward pass
        grad_out = (2.0 / (n_samples * self.input_dim)) * (pred_v - ut)

        grad_w = []
        grad_b = []

        gw_last = np.dot(activations[-1].T, grad_out)
        gb_last = np.sum(grad_out, axis=0, keepdims=True)
        grad_w.append(gw_last)
        grad_b.append(gb_last)

        dh = np.dot(grad_out, self.reflow_model.net.weights[-1].T)

        for i in range(len(self.reflow_model.net.weights) - 2, -1, -1):
            z = linear_outputs[i]
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            dsilu = sig + z * sig * (1.0 - sig)
            dz = dh * dsilu

            gw = np.dot(activations[i].T, dz)
            gb = np.sum(dz, axis=0, keepdims=True)
            grad_w.append(gw)
            grad_b.append(gb)

            if i > 0:
                dh = np.dot(dz, self.reflow_model.net.weights[i].T)

        grad_w.reverse()
        grad_b.reverse()

        # Update
        self.reflow_model.t_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for i in range(len(self.reflow_model.net.weights)):
            self.reflow_model.m_w[i] = (
                beta1 * self.reflow_model.m_w[i] + (1.0 - beta1) * grad_w[i]
            )
            self.reflow_model.v_w[i] = beta2 * self.reflow_model.v_w[i] + (
                1.0 - beta2
            ) * (grad_w[i] ** 2)
            m_hat = self.reflow_model.m_w[i] / (1.0 - beta1**self.reflow_model.t_step)
            v_hat = self.reflow_model.v_w[i] / (1.0 - beta2**self.reflow_model.t_step)
            self.reflow_model.net.weights[i] -= (
                self.lr * m_hat / (np.sqrt(v_hat) + eps)
            ).astype(np.float32)

            self.reflow_model.m_b[i] = (
                beta1 * self.reflow_model.m_b[i] + (1.0 - beta1) * grad_b[i]
            )
            self.reflow_model.v_b[i] = beta2 * self.reflow_model.v_b[i] + (
                1.0 - beta2
            ) * (grad_b[i] ** 2)
            mb_hat = self.reflow_model.m_b[i] / (1.0 - beta1**self.reflow_model.t_step)
            vb_hat = self.reflow_model.v_b[i] / (1.0 - beta2**self.reflow_model.t_step)
            self.reflow_model.net.biases[i] -= (
                self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
            ).astype(np.float32)

        return loss

    def sample_fast(
        self,
        num_samples: int = 16,
        steps: int = 1,
    ) -> np.ndarray:
        """Ultra-fast 1-step or 2-step straight-line ODE generation."""
        return self.reflow_model.sample(
            num_samples=num_samples, steps=steps, method="euler"
        )
