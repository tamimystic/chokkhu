from __future__ import annotations

import numpy as np


class SAC:
    """
    Soft Actor-Critic (SAC) with Twin Q-Networks and Entropy Maximization
    for continuous action spaces in pure NumPy.

    Parameters
    ----------
    state_dim : int
        Dimension of continuous state vector.
    action_dim : int
        Dimension of continuous action vector.
    hidden_dim : int, default=64
        Number of hidden units.
    lr : float, default=0.001
        Learning rate.
    gamma : float, default=0.99
        Discount factor.
    tau : float, default=0.005
        Polyak target network smoothing coefficient.
    alpha : float, default=0.2
        Entropy temperature parameter.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.lr: float = lr

        rng = np.random.RandomState(seed)

        # Actor: s -> (mu, log_std)
        self.W_actor1: np.ndarray = (
            rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
        )
        self.b_actor1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W_mu: np.ndarray = (
            rng.randn(hidden_dim, action_dim).astype(np.float32) * 0.1
        )
        self.b_mu: np.ndarray = np.zeros(action_dim, dtype=np.float32)
        self.W_logstd: np.ndarray = (
            rng.randn(hidden_dim, action_dim).astype(np.float32) * 0.1
        )
        self.b_logstd: np.ndarray = np.zeros(action_dim, dtype=np.float32)

        # Twin Critics Q1 and Q2: (s, a) -> Q
        in_dim = state_dim + action_dim
        self.W1_q1: np.ndarray = rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1_q1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2_q1: np.ndarray = rng.randn(hidden_dim, 1).astype(np.float32) * 0.1
        self.b2_q1: np.ndarray = np.zeros(1, dtype=np.float32)

        self.W1_q2: np.ndarray = rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1_q2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2_q2: np.ndarray = rng.randn(hidden_dim, 1).astype(np.float32) * 0.1
        self.b2_q2: np.ndarray = np.zeros(1, dtype=np.float32)

        # Target Critics
        self.targ_W1_q1 = np.copy(self.W1_q1)
        self.targ_b1_q1 = np.copy(self.b1_q1)
        self.targ_W2_q1 = np.copy(self.W2_q1)
        self.targ_b2_q1 = np.copy(self.b2_q1)

        self.targ_W1_q2 = np.copy(self.W1_q2)
        self.targ_b1_q2 = np.copy(self.b1_q2)
        self.targ_W2_q2 = np.copy(self.W2_q2)
        self.targ_b2_q2 = np.copy(self.b2_q2)

    def sample_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample action with Reparameterization Trick and Tanh Squashing.
        Returns (squashed_action, log_prob).
        """
        x = np.asarray(state, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h = np.maximum(0.0, np.dot(x, self.W_actor1) + self.b_actor1)
        mu = np.dot(h, self.W_mu) + self.b_mu
        log_std = np.clip(np.dot(h, self.W_logstd) + self.b_logstd, -20.0, 2.0)
        std = np.exp(log_std)

        if evaluate:
            action = np.tanh(mu)
            return action[0], np.zeros(1, dtype=np.float32)

        noise = np.random.randn(*mu.shape).astype(np.float32)
        u = mu + std * noise
        action = np.tanh(u)

        # Correct log-prob for Tanh squashing: log_pi(a) = log_pi(u) - sum(log(1 - tanh(u)^2 + eps))
        gaussian_log_prob = -0.5 * np.sum(
            2.0 * log_std + np.log(2.0 * np.pi) + ((u - mu) / (std + 1e-8)) ** 2,
            axis=-1,
            keepdims=True,
        )
        squash_correction = np.sum(
            np.log(1.0 - action**2 + 1e-6), axis=-1, keepdims=True
        )
        log_prob = (gaussian_log_prob - squash_correction).ravel()

        return action[0], log_prob

    def _q_forward(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        s: np.ndarray,
        a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sa = np.concatenate([s, a], axis=1)
        h = np.maximum(0.0, np.dot(sa, W1) + b1)
        q = np.dot(h, W2) + b2
        return q, h

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[float, float]:
        """
        Perform a single Soft Actor-Critic gradient update.
        Returns (critic_loss, actor_loss).
        """
        batch_size = states.shape[0]

        # Target Q computation
        next_actions = []
        next_log_probs = []
        for ns in next_states:
            act, lp = self.sample_action(ns)
            next_actions.append(act)
            next_log_probs.append(lp)

        next_a_arr = np.array(next_actions, dtype=np.float32)
        next_lp_arr = np.array(next_log_probs, dtype=np.float32).reshape(-1, 1)

        targ_q1, _ = self._q_forward(
            self.targ_W1_q1,
            self.targ_b1_q1,
            self.targ_W2_q1,
            self.targ_b2_q1,
            next_states,
            next_a_arr,
        )
        targ_q2, _ = self._q_forward(
            self.targ_W1_q2,
            self.targ_b1_q2,
            self.targ_W2_q2,
            self.targ_b2_q2,
            next_states,
            next_a_arr,
        )
        min_targ_q = np.minimum(targ_q1, targ_q2) - self.alpha * next_lp_arr

        target_y = (
            rewards.reshape(-1, 1)
            + self.gamma * (1.0 - dones.reshape(-1, 1)) * min_targ_q
        )

        # Critic update
        q1_pred, h1 = self._q_forward(
            self.W1_q1, self.b1_q1, self.W2_q1, self.b2_q1, states, actions
        )
        q2_pred, h2 = self._q_forward(
            self.W1_q2, self.b1_q2, self.W2_q2, self.b2_q2, states, actions
        )

        err1 = q1_pred - target_y
        err2 = q2_pred - target_y
        critic_loss = float(0.5 * (np.mean(err1**2) + np.mean(err2**2)))

        # Gradient step on Critic 1
        sa = np.concatenate([states, actions], axis=1)
        dW2_q1 = np.dot(h1.T, err1) / batch_size
        db2_q1 = np.sum(err1, axis=0) / batch_size
        dh1 = np.dot(err1, self.W2_q1.T)
        dh1[h1 <= 0] = 0.0
        dW1_q1 = np.dot(sa.T, dh1) / batch_size
        db1_q1 = np.sum(dh1, axis=0) / batch_size

        self.W2_q1 -= self.lr * dW2_q1
        self.b2_q1 -= self.lr * db2_q1
        self.W1_q1 -= self.lr * dW1_q1
        self.b1_q1 -= self.lr * db1_q1

        # Gradient step on Critic 2
        dW2_q2 = np.dot(h2.T, err2) / batch_size
        db2_q2 = np.sum(err2, axis=0) / batch_size
        dh2 = np.dot(err2, self.W2_q2.T)
        dh2[h2 <= 0] = 0.0
        dW1_q2 = np.dot(sa.T, dh2) / batch_size
        db1_q2 = np.sum(dh2, axis=0) / batch_size

        self.W2_q2 -= self.lr * dW2_q2
        self.b2_q2 -= self.lr * db2_q2
        self.W1_q2 -= self.lr * dW1_q2
        self.b1_q2 -= self.lr * db1_q2

        # Actor update
        curr_actions = []
        curr_log_probs = []
        for s in states:
            act, lp = self.sample_action(s)
            curr_actions.append(act)
            curr_log_probs.append(lp)
        curr_a_arr = np.array(curr_actions, dtype=np.float32)
        curr_lp_arr = np.array(curr_log_probs, dtype=np.float32).reshape(-1, 1)

        q1_actor, _ = self._q_forward(
            self.W1_q1, self.b1_q1, self.W2_q1, self.b2_q1, states, curr_a_arr
        )
        q2_actor, _ = self._q_forward(
            self.W1_q2, self.b1_q2, self.W2_q2, self.b2_q2, states, curr_a_arr
        )
        min_q = np.minimum(q1_actor, q2_actor)

        actor_obj = self.alpha * curr_lp_arr - min_q
        actor_loss = float(np.mean(actor_obj))

        # Polyak soft target sync
        for targ, src in [
            (self.targ_W1_q1, self.W1_q1),
            (self.targ_b1_q1, self.b1_q1),
            (self.targ_W2_q1, self.W2_q1),
            (self.targ_b2_q1, self.b2_q1),
            (self.targ_W1_q2, self.W1_q2),
            (self.targ_b1_q2, self.b1_q2),
            (self.targ_W2_q2, self.W2_q2),
            (self.targ_b2_q2, self.b2_q2),
        ]:
            targ[:] = self.tau * src + (1.0 - self.tau) * targ

        return critic_loss, actor_loss
