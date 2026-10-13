import numpy as np

from chokkhu.models.generative import DCGAN, DDPM, VAE, VQVAE, WGANGP


def test_ddpm_cosine_schedule_and_ddim_sampling() -> None:
    np.random.seed(42)
    ddpm_linear = DDPM(data_dim=8, timesteps=20, schedule="linear")
    assert ddpm_linear.betas.shape == (20,)
    assert ddpm_linear.schedule == "linear"

    ddpm_cosine = DDPM(data_dim=8, timesteps=20, schedule="cosine")
    assert ddpm_cosine.betas.shape == (20,)
    assert ddpm_cosine.schedule == "cosine"

    samples_ddpm = ddpm_cosine.sample(num_samples=3, num_steps=5, method="ddpm")
    assert samples_ddpm.shape == (3, 8)

    samples_ddim = ddpm_cosine.sample(
        num_samples=3, num_steps=5, method="ddim", eta=0.0
    )
    assert samples_ddim.shape == (3, 8)

    X = np.random.randn(8, 8)
    ddpm_cosine.fit(X, epochs=1, batch_size=4)
    preds = ddpm_cosine.predict(X[:2])
    assert preds.shape == (2, 8)

    params = ddpm_cosine.get_params()
    assert params["schedule"] == "cosine"
    assert params["data_dim"] == 8
    ddpm_cosine.set_params(schedule="linear")
    assert ddpm_cosine.schedule == "linear"


def test_vae_beta_and_free_bits() -> None:
    np.random.seed(42)
    vae = VAE(in_features=16, hidden_dim=8, latent_dim=4, beta=2.5, free_bits=0.02)
    assert vae.beta == 2.5
    assert vae.free_bits == 0.02

    X = np.random.randn(10, 16)
    recon, mu, logvar = vae(X)
    assert recon.shape == (10, 16)
    assert mu.shape == (10, 4)
    assert logvar.shape == (10, 4)
    assert vae.last_kl_loss > 0.0

    samples = vae.sample(num_samples=5)
    assert samples.shape == (5, 16)

    vae.fit(X, epochs=1, batch_size=5)
    preds = vae.predict(X[:3])
    assert preds.shape == (3, 16)

    params = vae.get_params()
    assert params["beta"] == 2.5
    assert params["free_bits"] == 0.02
    vae.set_params(beta=1.0)
    assert vae.beta == 1.0


def test_vqvae_dynamic_params() -> None:
    np.random.seed(42)
    vqvae = VQVAE(
        in_features=16,
        hidden_dim=8,
        embedding_dim=4,
        num_embeddings=16,
        commitment_cost=0.5,
    )
    assert vqvae.num_embeddings == 16
    assert vqvae.commitment_cost == 0.5

    X = np.random.randn(6, 16)
    recon, indices = vqvae(X)
    assert recon.shape == (6, 16)
    assert len(indices) == 6

    vqvae.fit(X, epochs=1, batch_size=3)
    preds = vqvae.predict(X[:2])
    assert preds.shape == (2, 16)

    params = vqvae.get_params()
    assert params["commitment_cost"] == 0.5
    vqvae.set_params(commitment_cost=0.2)
    assert vqvae.commitment_cost == 0.2


def test_dcgan_and_wgangp_dynamic_params() -> None:
    np.random.seed(42)
    dcgan = DCGAN(latent_dim=8, hidden_dim=16, out_dim=12, label_smoothing=0.1)
    assert dcgan.label_smoothing == 0.1
    assert dcgan.latent_dim == 8
    samples_dcgan = dcgan.generate(num_samples=4)
    assert samples_dcgan.shape == (4, 12)
    preds_dcgan = dcgan.predict(np.zeros((3, 12)))
    assert preds_dcgan.shape == (3, 12)

    params_dc = dcgan.get_params()
    assert params_dc["label_smoothing"] == 0.1
    dcgan.set_params(label_smoothing=0.2)
    assert dcgan.label_smoothing == 0.2

    wgan = WGANGP(latent_dim=8, hidden_dim=16, out_dim=12, gp_lambda=7.5)
    assert wgan.gp_lambda == 7.5
    samples_wgan = wgan.generate(num_samples=4)
    assert samples_wgan.shape == (4, 12)

    gp = wgan.compute_gradient_penalty(np.random.randn(4, 12), samples_wgan)
    assert isinstance(gp, float)

    params_wgan = wgan.get_params()
    assert params_wgan["gp_lambda"] == 7.5
    wgan.set_params(gp_lambda=15.0)
    assert wgan.gp_lambda == 15.0
