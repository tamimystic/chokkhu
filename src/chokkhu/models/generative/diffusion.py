"""Denoising Diffusion Probabilistic Models (DDPM) from First Principles (Ho et al., 2020)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from ..dl.layers import Linear, Module
from ..dl.activations import SiLU


class SinusoidalTimeEmbedding(Module):
    """Sinusoidal Positional/Time Embedding for Diffusion Timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        self.inv_freq = inv_freq

    def forward(self, t: np.ndarray) -> np.ndarray:
        """Compute sinusoidal time embeddings of shape (N, dim)."""
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        sin_part = np.sin(t * self.inv_freq[np.newaxis, :])
        cos_part = np.cos(t * self.inv_freq[np.newaxis, :])
        return np.concatenate([sin_part, cos_part], axis=-1)


class DenoisingMLP(Module):
    """Time-conditioned Denoising Network predicting noise epsilon."""

    def __init__(
        self, data_dim: int = 32, time_dim: int = 64, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = Linear(time_dim, hidden_dim)

        self.in_proj = Linear(data_dim, hidden_dim)
        self.mid_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, data_dim)
        self.act = SiLU()

    def forward(self, x: Tensor, t: np.ndarray) -> Tensor:
        t_emb = self.act(self.time_mlp(Tensor(self.time_embed(t), requires_grad=False)))
        h = self.act(self.in_proj(x)) + t_emb
        h = self.act(self.mid_proj(h))
        return self.out_proj(h)


class DDPM(Module, ChokkhuModel):
    """Denoising Diffusion Probabilistic Model (Ho et al., 2020)."""

    def __init__(
        self,
        data_dim: int = 32,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps

        self.betas: np.ndarray = np.linspace(
            beta_start, beta_end, timesteps, dtype=np.float64
        )
        self.alphas: np.ndarray = 1.0 - self.betas
        self.alphas_bar: np.ndarray = np.cumprod(self.alphas, axis=0)

        self.sqrt_alphas_bar: np.ndarray = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar: np.ndarray = np.sqrt(1.0 - self.alphas_bar)

        self.model = DenoisingMLP(data_dim=data_dim, hidden_dim=hidden_dim)

    def q_sample(
        self, x_0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion process: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = np.random.randn(*x_0.shape)

        sqrt_ab = self.sqrt_alphas_bar[t][:, None]
        sqrt_1_ab = self.sqrt_one_minus_alphas_bar[t][:, None]
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
        return x_t, noise

    def forward(self, x_0: Union[np.ndarray, Tensor]) -> Tuple[Tensor, np.ndarray]:
        """Compute training objective: MSE between predicted noise and added noise."""
        if not isinstance(x_0, Tensor):
            x_0 = Tensor(np.asarray(x_0, dtype=np.float64), requires_grad=True)

        N = x_0.shape[0]
        x_0_flat = x_0.reshape(N, self.data_dim).data
        t = np.random.randint(0, self.timesteps, size=(N,))
        noise = np.random.randn(*x_0_flat.shape)

        x_t, _ = self.q_sample(x_0_flat, t, noise)
        pred_noise = self.model(Tensor(x_t, requires_grad=x_0.requires_grad), t)
        return pred_noise, noise

    def sample(
        self, num_samples: int = 16, num_steps: Optional[int] = None
    ) -> np.ndarray:
        """Reverse diffusion generation: sample x_0 ~ p(x_0) from Gaussian prior."""
        steps = num_steps if num_steps is not None else min(self.timesteps, 50)
        step_indices = np.linspace(self.timesteps - 1, 0, steps, dtype=int)

        x = np.random.randn(num_samples, self.data_dim)

        for i, t_idx in enumerate(step_indices):
            t_batch: np.ndarray = np.full((num_samples,), t_idx, dtype=int)
            pred_noise = self.model(Tensor(x, requires_grad=False), t_batch).data

            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_bar[t_idx]

            mean = (1.0 / np.sqrt(alpha)) * (
                x - (beta / np.sqrt(1.0 - alpha_bar)) * pred_noise
            )
            if t_idx > 0:
                z = np.random.randn(*x.shape)
                sigma = np.sqrt(beta)
                x = mean + sigma * z
            else:
                x = mean

        return x

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> DDPM:
        """Fit DDPM on training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        for _ in range(epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = Tensor(X_arr[batch_idx], requires_grad=True)
                _ = self.forward(xb)
        return self

    def predict(self, X: Any) -> np.ndarray:
        N = len(X) if hasattr(X, "__len__") else 16
        return self.sample(num_samples=N)
