"""Recurrent State-Space World Model (Dreamer-style) in pure NumPy."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np


class RecurrentWorldModel:
    """Recurrent State Space Model (RSSM) for Model-Based Reinforcement Learning in pure NumPy (Hafner et al. 2020).

    Learns compact latent environment dynamics enabling synthetic imagination rollouts
    with deterministic recurrence ($h_t$) and stochastic latent representations ($z_t$).

    Parameters
    ----------
    obs_dim : int, default=16
        Dimensionality of observation vectors.
    action_dim : int, default=4
        Dimensionality of continuous action control vectors.
    deter_dim : int, default=32
        Dimensionality of deterministic recurrent hidden state $h_t$.
    stoch_dim : int, default=16
        Dimensionality of stochastic latent Gaussian state $z_t$.
    seed : int, default=42
    """

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        deter_dim: int = 32,
        stoch_dim: int = 16,
        seed: int = 42,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)

        self.rng = np.random.default_rng(seed)

        # 1. Deterministic Recurrence GRU Cell: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        gru_in = self.stoch_dim + self.action_dim
        scale_gru = np.sqrt(2.0 / (gru_in + deter_dim))
        self.w_gru_ih = self.rng.normal(
            0, scale_gru, size=(3 * deter_dim, gru_in)
        ).astype(np.float32)
        self.w_gru_hh = self.rng.normal(
            0, scale_gru, size=(3 * deter_dim, deter_dim)
        ).astype(np.float32)
        self.b_gru: np.ndarray = np.zeros(3 * deter_dim, dtype=np.float32)

        # 2. Stochastic Prior Net: p(z_t | h_t)
        scale_prior = np.sqrt(2.0 / deter_dim)
        self.w_prior_mean = self.rng.normal(
            0, scale_prior, size=(stoch_dim, deter_dim)
        ).astype(np.float32)
        self.w_prior_std = self.rng.normal(
            0, scale_prior, size=(stoch_dim, deter_dim)
        ).astype(np.float32)

        # 3. Stochastic Posterior Net: q(z_t | h_t, o_t)
        post_in = deter_dim + obs_dim
        scale_post = np.sqrt(2.0 / post_in)
        self.w_post_mean = self.rng.normal(
            0, scale_post, size=(stoch_dim, post_in)
        ).astype(np.float32)
        self.w_post_std = self.rng.normal(
            0, scale_post, size=(stoch_dim, post_in)
        ).astype(np.float32)

        # 4. Observation Decoder: p(o_t | h_t, z_t)
        feat_dim = deter_dim + stoch_dim
        scale_feat = np.sqrt(2.0 / feat_dim)
        self.w_obs_dec = self.rng.normal(
            0, scale_feat, size=(obs_dim, feat_dim)
        ).astype(np.float32)
        self.b_obs_dec: np.ndarray = np.zeros(obs_dim, dtype=np.float32)

        # 5. Reward Predictor: p(r_t | h_t, z_t)
        self.w_rew = self.rng.normal(0, scale_feat, size=(1, feat_dim)).astype(
            np.float32
        )
        self.b_rew: np.ndarray = np.zeros(1, dtype=np.float32)

    def _gru_cell(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Executes single GRU recurrent update step."""
        gates = (
            np.matmul(x, self.w_gru_ih.T)
            + np.matmul(h_prev, self.w_gru_hh.T)
            + self.b_gru
        )
        r_gate, z_gate, n_gate = np.split(gates, 3, axis=-1)

        r = 1.0 / (1.0 + np.exp(-np.clip(r_gate, -15.0, 15.0)))
        z = 1.0 / (1.0 + np.exp(-np.clip(z_gate, -15.0, 15.0)))
        n = np.tanh(
            n_gate + r * (np.matmul(h_prev, self.w_gru_hh[2 * self.deter_dim :].T))
        )

        h_next = (1.0 - z) * n + z * h_prev
        return h_next

    def initial_state(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Returns zero-initialized deterministic state $h_0$ and stochastic state $z_0$."""
        h0: np.ndarray = np.zeros((batch_size, self.deter_dim), dtype=np.float32)
        z0: np.ndarray = np.zeros((batch_size, self.stoch_dim), dtype=np.float32)
        return h0, z0

    def step_prior(
        self, h_prev: np.ndarray, z_prev: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rolls dynamics forward with prior distribution p(z_t | h_t)."""
        x_in = np.concatenate([z_prev, action], axis=-1)
        h_next = self._gru_cell(x_in, h_prev)

        mean = np.matmul(h_next, self.w_prior_mean.T)
        std = (
            np.log1p(
                np.exp(np.clip(np.matmul(h_next, self.w_prior_std.T), -10.0, 10.0))
            )
            + 1e-4
        )

        noise = self.rng.standard_normal(size=mean.shape).astype(np.float32)
        z_sample = mean + std * noise
        return h_next, z_sample, mean, std

    def step_posterior(
        self,
        h_prev: np.ndarray,
        z_prev: np.ndarray,
        action: np.ndarray,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rolls dynamics forward incorporating true observation q(z_t | h_t, o_t)."""
        x_in = np.concatenate([z_prev, action], axis=-1)
        h_next = self._gru_cell(x_in, h_prev)

        post_in = np.concatenate([h_next, obs], axis=-1)
        mean = np.matmul(post_in, self.w_post_mean.T)
        std = (
            np.log1p(
                np.exp(np.clip(np.matmul(post_in, self.w_post_std.T), -10.0, 10.0))
            )
            + 1e-4
        )

        noise = self.rng.standard_normal(size=mean.shape).astype(np.float32)
        z_sample = mean + std * noise
        return h_next, z_sample, mean, std

    def decode_observation(self, h: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Decodes reconstructed observation from latent feature state."""
        feat = np.concatenate([h, z], axis=-1)
        return np.matmul(feat, self.w_obs_dec.T) + self.b_obs_dec

    def predict_reward(self, h: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Predicts scalar reward signal from latent feature state."""
        feat = np.concatenate([h, z], axis=-1)
        return np.matmul(feat, self.w_rew.T) + self.b_rew

    def imagine_trajectory(
        self,
        h0: np.ndarray,
        z0: np.ndarray,
        policy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        horizon: int = 15,
    ) -> Dict[str, np.ndarray]:
        """Rolls out synthetic imagination trajectories without environment interactions."""
        h_seq: List[np.ndarray] = []
        z_seq: List[np.ndarray] = []
        a_seq: List[np.ndarray] = []
        r_seq: List[np.ndarray] = []

        curr_h = np.asarray(h0, dtype=np.float32)
        curr_z = np.asarray(z0, dtype=np.float32)

        for _ in range(horizon):
            action = policy_fn(curr_h, curr_z)
            curr_h, curr_z, _, _ = self.step_prior(curr_h, curr_z, action)
            rew = self.predict_reward(curr_h, curr_z)

            h_seq.append(curr_h)
            z_seq.append(curr_z)
            a_seq.append(action)
            r_seq.append(rew)

        return {
            "h": np.stack(h_seq, axis=1),
            "z": np.stack(z_seq, axis=1),
            "actions": np.stack(a_seq, axis=1),
            "rewards": np.stack(r_seq, axis=1),
        }
