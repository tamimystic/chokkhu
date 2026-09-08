"""Unit tests for Generative AI Universe (VAE, VQ-VAE, DDPM, DCGAN, WGAN-GP)."""

import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.generative import (
    DCGAN,
    DDPM,
    VAE,
    VQVAE,
    WGANGP,
)
from chokkhu.models import train


def test_vae_forward_reparameterize_and_sample() -> None:
    x = Tensor(np.random.randn(4, 64), requires_grad=True)
    vae = VAE(in_features=64, hidden_dim=32, latent_dim=8)

    recon, mu, logvar = vae(x)
    assert recon.shape == (4, 64)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert vae.last_kl_loss >= 0.0

    samples = vae.sample(num_samples=5)
    assert samples.shape == (5, 64)


def test_vqvae_quantization_and_ste() -> None:
    x = Tensor(np.random.randn(3, 64), requires_grad=True)
    vqvae = VQVAE(
        in_features=64,
        hidden_dim=32,
        embedding_dim=16,
        num_embeddings=32,
        commitment_cost=0.25,
    )

    recon, indices = vqvae(x)
    assert recon.shape == (3, 64)
    assert indices.shape == (3,)
    assert np.all(indices >= 0) and np.all(indices < 32)
    assert vqvae.quantizer.last_vq_loss >= 0.0


def test_ddpm_diffusion_process_and_sampling() -> None:
    x = Tensor(np.random.randn(4, 16), requires_grad=True)
    ddpm = DDPM(data_dim=16, timesteps=100, hidden_dim=32)

    pred_noise, noise = ddpm(x)
    assert pred_noise.shape == (4, 16)
    assert noise.shape == (4, 16)

    samples = ddpm.sample(num_samples=4, num_steps=10)
    assert samples.shape == (4, 16)


def test_dcgan_generation() -> None:
    dcgan = DCGAN(latent_dim=16, hidden_dim=32, out_dim=64)
    samples = dcgan.generate(num_samples=6)
    assert samples.shape == (6, 64)

    scores = dcgan.discriminator(Tensor(samples))
    assert scores.shape == (6, 1)


def test_wgan_gp_and_critic() -> None:
    wgan = WGANGP(latent_dim=16, hidden_dim=32, out_dim=64, gp_lambda=10.0)
    samples = wgan.generate(num_samples=4)
    assert samples.shape == (4, 64)

    real_data = np.random.randn(4, 64)
    gp = wgan.compute_gradient_penalty(real_data, samples)
    assert gp >= 0.0


def test_train_generative_via_engine() -> None:
    X_train = np.random.randn(4, 32)
    model = train(
        model="vae",
        X_train=X_train,
        in_features=32,
        hidden_dim=16,
        latent_dim=4,
        epochs=1,
        batch_size=2,
    )
    assert isinstance(model, VAE)
