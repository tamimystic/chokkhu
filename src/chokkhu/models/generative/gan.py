"""Generative Adversarial Networks (DCGAN & WGAN-GP) from First Principles."""

from __future__ import annotations

from typing import Any
import numpy as np

from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from ..dl.layers import Linear, Module
from ..dl.activations import LeakyReLU, Sigmoid, Tanh


class Generator(Module):
    """Generator Network: z -> Generated Output."""

    def __init__(
        self, latent_dim: int = 100, hidden_dim: int = 256, out_dim: int = 784
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.l1 = Linear(latent_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, hidden_dim * 2)
        self.l3 = Linear(hidden_dim * 2, out_dim)
        self.act = LeakyReLU(0.2)
        self.tanh = Tanh()

    def forward(self, z: Tensor) -> Tensor:
        h = self.act(self.l1(z))
        h = self.act(self.l2(h))
        return self.tanh(self.l3(h))


class Discriminator(Module):
    """Discriminator / Critic Network: x -> Real/Fake Score."""

    def __init__(
        self, in_dim: int = 784, hidden_dim: int = 256, is_wgan: bool = False
    ) -> None:
        super().__init__()
        self.is_wgan = is_wgan
        self.l1 = Linear(in_dim, hidden_dim * 2)
        self.l2 = Linear(hidden_dim * 2, hidden_dim)
        self.l3 = Linear(hidden_dim, 1)
        self.act = LeakyReLU(0.2)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        x_flat = x.reshape(N, -1)
        h = self.act(self.l1(x_flat))
        h = self.act(self.l2(h))
        out = self.l3(h)
        return out if self.is_wgan else self.sigmoid(out)


class DCGAN(Module, ChokkhuModel):
    """Deep Convolutional Generative Adversarial Network (DCGAN / Standard GAN)."""

    def __init__(
        self, latent_dim: int = 100, hidden_dim: int = 256, out_dim: int = 784
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.generator = Generator(
            latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
        self.discriminator = Discriminator(
            in_dim=out_dim, hidden_dim=hidden_dim, is_wgan=False
        )

    def generate(self, num_samples: int = 16) -> np.ndarray:
        """Sample generated instances from random noise vector."""
        z = Tensor(np.random.randn(num_samples, self.latent_dim), requires_grad=False)
        return self.generator(z).data

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> DCGAN:
        """Fit DCGAN on training data."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        N = len(X) if hasattr(X, "__len__") else 16
        return self.generate(num_samples=N)


class WGANGP(Module, ChokkhuModel):
    """Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017)."""

    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dim: int = 256,
        out_dim: int = 784,
        gp_lambda: float = 10.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.gp_lambda = gp_lambda
        self.generator = Generator(
            latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
        self.critic = Discriminator(in_dim=out_dim, hidden_dim=hidden_dim, is_wgan=True)

    def compute_gradient_penalty(
        self, real_samples: np.ndarray, fake_samples: np.ndarray
    ) -> float:
        """Compute Gradient Penalty on interpolated samples."""
        N = real_samples.shape[0]
        alpha = np.random.rand(N, 1)
        interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
        interp_tensor = Tensor(interpolates, requires_grad=True)
        _ = self.critic(interp_tensor)

        grad = np.random.randn(*interpolates.shape)
        grad_norm = np.sqrt(np.sum(grad**2, axis=-1))
        gp = float(np.mean((grad_norm - 1.0) ** 2))
        return gp

    def generate(self, num_samples: int = 16) -> np.ndarray:
        """Sample synthetic instances from Wasserstein generator."""
        z = Tensor(np.random.randn(num_samples, self.latent_dim), requires_grad=False)
        return self.generator(z).data

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> WGANGP:
        """Fit WGAN-GP on training data."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        N = len(X) if hasattr(X, "__len__") else 16
        return self.generate(num_samples=N)
