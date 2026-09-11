"""Robotics Diffusion Policy for Continuous Action Trajectory Generation in pure NumPy."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class DiffusionPolicy:
    r"""Diffusion Policy for Continuous Action Trajectory Synthesis in pure NumPy (Chi et al. 2023).

    Generates multi-modal, temporal action sequences $A \in \mathbb{R}^{T_a \times D_a}$ conditioned
    on observation history $O \in \mathbb{R}^{D_o}$ via conditional Denoising Diffusion (DDPM).

    Parameters
    ----------
    action_dim : int, default=4
        Dimensionality of each robotic action vector.
    pred_horizon : int, default=8
        Number of future action steps predicted in each trajectory.
    obs_dim : int, default=16
        Dimensionality of conditioning observation embedding.
    num_timesteps : int, default=50
        Number of diffusion denoising steps.
    beta_start : float, default=1e-4
        Initial variance schedule value.
    beta_end : float, default=0.02
        Terminal variance schedule value.
    hidden_dim : int, default=64
        Internal width of denoising network.
    seed : int, default=42
    """

    def __init__(
        self,
        action_dim: int = 4,
        pred_horizon: int = 8,
        obs_dim: int = 16,
        num_timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.action_dim = int(action_dim)
        self.pred_horizon = int(pred_horizon)
        self.obs_dim = int(obs_dim)
        self.num_timesteps = int(num_timesteps)
        self.flat_action_dim = self.pred_horizon * self.action_dim
        self.hidden_dim = int(hidden_dim)

        self.rng = np.random.default_rng(seed)

        # Variance schedule
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        # MLP Denoising Network weights: input is (flat_action + obs_dim + time_dim)
        time_dim = 16
        self.time_dim = time_dim
        in_features = self.flat_action_dim + self.obs_dim + time_dim

        scale_1 = np.sqrt(2.0 / in_features)
        scale_h = np.sqrt(2.0 / hidden_dim)

        self.w1 = self.rng.normal(0, scale_1, size=(in_features, hidden_dim)).astype(
            np.float32
        )
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.w2 = self.rng.normal(0, scale_h, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        )
        self.b2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.w_out = self.rng.normal(
            0, scale_h, size=(hidden_dim, self.flat_action_dim)
        ).astype(np.float32)
        self.b_out: np.ndarray = np.zeros(self.flat_action_dim, dtype=np.float32)

    def _time_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """Computes sinusoidal time embeddings (B, time_dim)."""
        t = np.asarray(timesteps, dtype=np.float32)
        half_dim = self.time_dim // 2
        emb_scale = np.log(10000.0) / max(1, half_dim - 1)
        freqs = np.exp(-emb_scale * np.arange(half_dim, dtype=np.float32))

        args = t[:, np.newaxis] * freqs[np.newaxis, :]
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

    def predict_noise(
        self, noisy_actions: np.ndarray, timesteps: np.ndarray, obs_cond: np.ndarray
    ) -> np.ndarray:
        """Predicts injected Gaussian noise vector epsilon_theta."""
        B = len(noisy_actions)
        flat_act = noisy_actions.reshape(B, -1)
        time_emb = self._time_embedding(timesteps)
        obs = np.asarray(obs_cond, dtype=np.float32).reshape(B, -1)

        # Concatenate: (B, flat_act_dim + obs_dim + time_dim)
        h = np.concatenate([flat_act, obs, time_emb], axis=-1)

        # Layer 1: SiLU
        h1 = np.matmul(h, self.w1) + self.b1
        h1_act = h1 * (1.0 / (1.0 + np.exp(-np.clip(h1, -15.0, 15.0))))

        # Layer 2: SiLU
        h2 = np.matmul(h1_act, self.w2) + self.b2
        h2_act = h2 * (1.0 / (1.0 + np.exp(-np.clip(h2, -15.0, 15.0))))

        # Output projection
        noise_pred = np.matmul(h2_act, self.w_out) + self.b_out
        return noise_pred.reshape(B, self.pred_horizon, self.action_dim)

    def q_sample(
        self, actions: np.ndarray, timesteps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: adds calibrated Gaussian noise to action trajectory."""
        actions_arr = np.asarray(actions, dtype=np.float32)
        noise = self.rng.standard_normal(size=actions_arr.shape).astype(np.float32)

        alpha_cum = self.alphas_cumprod[timesteps]
        sqrt_alpha = np.sqrt(alpha_cum)[:, np.newaxis, np.newaxis]
        sqrt_one_minus = np.sqrt(1.0 - alpha_cum)[:, np.newaxis, np.newaxis]

        noisy_actions = sqrt_alpha * actions_arr + sqrt_one_minus * noise
        return noisy_actions, noise

    def sample(self, obs_cond: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Executes reverse diffusion sampling loop to generate planned action trajectories."""
        obs = np.asarray(obs_cond, dtype=np.float32)
        if obs.ndim == 1:
            obs = np.repeat(obs[np.newaxis, :], n_samples, axis=0)

        B = len(obs)
        # Start from pure isotropic Gaussian noise: (B, pred_horizon, action_dim)
        x_t = self.rng.standard_normal(
            size=(B, self.pred_horizon, self.action_dim)
        ).astype(np.float32)

        for step in reversed(range(self.num_timesteps)):
            t_batch: np.ndarray = np.full(B, step, dtype=np.int32)
            noise_pred = self.predict_noise(x_t, t_batch, obs)

            alpha = self.alphas[step]
            alpha_cum = self.alphas_cumprod[step]
            beta = self.betas[step]

            # Mean update
            mean = (x_t - (beta / np.sqrt(1.0 - alpha_cum)) * noise_pred) / np.sqrt(
                alpha
            )

            if step > 0:
                noise_z = self.rng.standard_normal(size=x_t.shape).astype(np.float32)
                sigma = np.sqrt(beta)
                x_t = mean + sigma * noise_z
            else:
                x_t = mean

        return x_t
