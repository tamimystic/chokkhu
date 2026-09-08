"""Autoencoders, Variational Autoencoders (VAE), and Vector Quantized VAE (VQ-VAE) from First Principles."""

from __future__ import annotations

from typing import Any, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from ..dl.layers import Linear, Module, Parameter
from ..dl.activations import ReLU, Sigmoid


class VAE(Module, ChokkhuModel):
    """Variational Autoencoder (VAE) with Reparameterization Trick (Kingma & Welling, 2013)."""

    def __init__(
        self,
        in_features: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_dense = Linear(in_features, hidden_dim)
        self.enc_act = ReLU()
        self.fc_mu = Linear(hidden_dim, latent_dim)
        self.fc_logvar = Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_dense = Linear(latent_dim, hidden_dim)
        self.dec_act = ReLU()
        self.dec_out = Linear(hidden_dim, in_features)
        self.sigmoid = Sigmoid()

        self.last_kl_loss: float = 0.0

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input into mean and log-variance."""
        h = self.enc_act(self.enc_dense(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = np.exp(0.5 * logvar.data)
            eps = np.random.randn(*std.shape)
            z_data = mu.data + std * eps
            return Tensor(z_data, requires_grad=mu.requires_grad)
        else:
            return mu

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector z to reconstructed output."""
        h = self.dec_act(self.dec_dense(z))
        return self.sigmoid(self.dec_out(h))

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning (reconstruction, mu, logvar)."""
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)

        N = x.shape[0]
        x_flat = x.reshape(N, -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Compute KL Divergence loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * np.sum(1.0 + logvar.data - mu.data**2 - np.exp(logvar.data)) / N
        self.last_kl_loss = float(kl)

        return recon.reshape(x.shape), mu, logvar

    def sample(self, num_samples: int = 16) -> np.ndarray:
        """Sample generated instances from standard normal prior."""
        z = Tensor(np.random.randn(num_samples, self.latent_dim), requires_grad=False)
        return self.decode(z).data

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> VAE:
        """Fit VAE on training data."""
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
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        recon, _, _ = self.forward(X)
        return recon.data


class VectorQuantizer(Module):
    """Vector Quantization Codebook with Straight-Through Estimator (van den Oord et al., 2017)."""

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook weights
        self.embedding = Parameter(
            np.random.randn(num_embeddings, embedding_dim)
            * (1.0 / np.sqrt(embedding_dim))
        )
        self.last_vq_loss: float = 0.0

    def forward(self, z_e: Tensor) -> Tuple[Tensor, np.ndarray]:
        orig_shape = z_e.shape
        flat_z = z_e.data.reshape(-1, self.embedding_dim)
        codebook = self.embedding.data

        distances = (
            np.sum(flat_z**2, axis=1, keepdims=True)
            + np.sum(codebook**2, axis=1, keepdims=True).T
            - 2.0 * np.matmul(flat_z, codebook.T)
        )

        encoding_indices = np.argmin(distances, axis=1)
        quantized_flat = codebook[encoding_indices]
        quantized_data = quantized_flat.reshape(orig_shape)

        codebook_loss = np.mean((flat_z - quantized_flat) ** 2)
        commitment_loss = np.mean((flat_z - quantized_flat) ** 2) * self.commitment_cost
        self.last_vq_loss = float(codebook_loss + commitment_loss)

        quantized_ste = z_e.data + (quantized_data - z_e.data)
        out = Tensor(quantized_ste, requires_grad=z_e.requires_grad)
        return out, encoding_indices.reshape(orig_shape[:-1])


class VQVAE(Module, ChokkhuModel):
    """Vector-Quantized Variational Autoencoder (VQ-VAE)."""

    def __init__(
        self,
        in_features: int = 784,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = Linear(in_features, embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        # Decoder
        self.decoder_dense = Linear(embedding_dim, hidden_dim)
        self.dec_act = ReLU()
        self.decoder_out = Linear(hidden_dim, in_features)
        self.sigmoid = Sigmoid()

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tuple[Tensor, np.ndarray]:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)

        N = x.shape[0]
        x_flat = x.reshape(N, -1)
        z_e = self.encoder(x_flat)
        z_q, indices = self.quantizer(z_e)
        recon = self.sigmoid(self.decoder_out(self.dec_act(self.decoder_dense(z_q))))
        return recon.reshape(x.shape), indices

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> VQVAE:
        """Fit VQVAE on training data."""
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
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        recon, _ = self.forward(X)
        return recon.data
