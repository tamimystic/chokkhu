"""Sovereign Rectified Flow Matching & Continuous ODE Velocity Generation in Pure NumPy."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np


class VelocityMLP:
    """Multi-Layer Perceptron parameterizing the continuous velocity vector field v_theta(x, t)."""

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

        # Sinusoidal time embedding projection frequencies
        half_dim = time_embed_dim // 2
        self.freqs = np.exp(-np.log(10000.0) * np.arange(0, half_dim) / max(half_dim, 1))

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
        """Compute sinusoidal time embeddings for t in [0, 1]."""
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        args = t * self.freqs.reshape(1, -1) * 2.0 * np.pi
        sin_emb = np.sin(args)
        cos_emb = np.cos(args)
        return np.concatenate([sin_emb, cos_emb], axis=-1).astype(np.float32)

    def forward(self, x: np.ndarray, t: Union[np.ndarray, float]) -> np.ndarray:
        """Forward pass computing velocity vector field v_theta(x, t)."""
        x = np.asarray(x, dtype=np.float32)
        if isinstance(t, (int, float)):
            t_arr: np.ndarray = np.full((x.shape[0],), float(t), dtype=np.float32)
        elif isinstance(t, np.ndarray) and t.ndim == 0:
            t_arr = np.full((x.shape[0],), float(t.item()), dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.shape[0] != x.shape[0]:
                t_arr = np.full((x.shape[0],), float(t_arr[0]), dtype=np.float32)

        t_emb = self._embed_time(t_arr)
        h = np.concatenate([x, t_emb], axis=-1)

        for i in range(len(self.weights) - 1):
            h = np.dot(h, self.weights[i]) + self.biases[i]
            # SiLU / Swish activation: x * sigmoid(x)
            h = h / (1.0 + np.exp(-np.clip(h, -15.0, 15.0)))

        # Output linear layer
        out = np.dot(h, self.weights[-1]) + self.biases[-1]
        return out

    def get_params(self) -> List[np.ndarray]:
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        return params

    def set_params(self, params: List[np.ndarray]) -> None:
        idx = 0
        for i in range(len(self.weights)):
            self.weights[i] = params[idx]
            self.biases[i] = params[idx + 1]
            idx += 2


class FlowMatching:
    """Sovereign Rectified Flow Matching with Straight-Line Optimal Transport and ODE Solvers."""

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

        self.net = VelocityMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            time_embed_dim=time_embed_dim,
            random_state=random_state,
        )

        # Adam optimizer state
        self.m_w = [np.zeros_like(w) for w in self.net.weights]
        self.v_w = [np.zeros_like(w) for w in self.net.weights]
        self.m_b = [np.zeros_like(b) for b in self.net.biases]
        self.v_b = [np.zeros_like(b) for b in self.net.biases]
        self.t_step = 0

    def compute_target(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute straight-line interpolated point x_t and target velocity u_t."""
        t = t.reshape(-1, 1)
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0
        return xt, ut

    def compute_loss(self, x1: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Sample x0 ~ N(0, I) and t ~ U(0, 1) and compute Flow Matching MSE loss."""
        x1 = np.asarray(x1, dtype=np.float32)
        n_samples = x1.shape[0]
        x0 = self.rng.randn(*x1.shape).astype(np.float32)
        t = self.rng.uniform(0.0, 1.0, size=(n_samples,)).astype(np.float32)

        xt, ut = self.compute_target(x0, x1, t)
        pred_v = self.net.forward(xt, t)
        loss = float(np.mean((pred_v - ut) ** 2))
        return loss, xt, t, ut

    def train_step(self, x1: np.ndarray) -> float:
        """Perform one stochastic gradient descent update using backpropagation."""
        x1 = np.asarray(x1, dtype=np.float32)
        n_samples = x1.shape[0]
        x0 = self.rng.randn(*x1.shape).astype(np.float32)
        t = self.rng.uniform(0.0, 1.0, size=(n_samples,)).astype(np.float32)

        t_col = t.reshape(-1, 1)
        xt = (1.0 - t_col) * x0 + t_col * x1
        ut = x1 - x0

        # Forward pass tracking intermediate activations
        t_emb = self.net._embed_time(t)
        h_in = np.concatenate([xt, t_emb], axis=-1)

        activations = [h_in]
        linear_outputs = []

        h = h_in
        for i in range(len(self.net.weights) - 1):
            z = np.dot(h, self.net.weights[i]) + self.net.biases[i]
            linear_outputs.append(z)
            # SiLU activation
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            h = z * sig
            activations.append(h)

        pred_v = np.dot(h, self.net.weights[-1]) + self.net.biases[-1]
        loss = float(np.mean((pred_v - ut) ** 2))

        # Backward pass
        grad_out = (2.0 / (n_samples * self.input_dim)) * (pred_v - ut)

        grad_w = []
        grad_b = []

        # Last layer
        gw_last = np.dot(activations[-1].T, grad_out)
        gb_last = np.sum(grad_out, axis=0, keepdims=True)
        grad_w.append(gw_last)
        grad_b.append(gb_last)

        dh = np.dot(grad_out, self.net.weights[-1].T)

        for i in range(len(self.net.weights) - 2, -1, -1):
            z = linear_outputs[i]
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            # Derivative of SiLU: sig + z * sig * (1 - sig)
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
            m_hat = self.m_w[i] / (1.0 - beta1 ** self.t_step)
            v_hat = self.v_w[i] / (1.0 - beta2 ** self.t_step)
            self.net.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * (grad_b[i] ** 2)
            mb_hat = self.m_b[i] / (1.0 - beta1 ** self.t_step)
            vb_hat = self.v_b[i] / (1.0 - beta2 ** self.t_step)
            self.net.biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

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
                raise ValueError(f"Unknown ODE integration method: {method}. Choose from 'euler', 'midpoint', 'rk4'.")

        return x


# Alias
RectifiedFlow = FlowMatching
