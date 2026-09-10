from __future__ import annotations

import numpy as np


class PPO:
    """
    Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
    and Clipped Surrogate Objective in pure NumPy.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    action_dim : int
        Number of discrete actions.
    lr_actor : float, default=0.001
        Learning rate for actor.
    lr_critic : float, default=0.002
        Learning rate for critic.
    gamma : float, default=0.99
        Discount factor.
    lam : float, default=0.95
        GAE lambda smoothing parameter.
    clip_ratio : float, default=0.2
        PPO clipping range [1-clip_ratio, 1+clip_ratio].
    epochs : int, default=4
        Number of optimization epochs per rollout.
    entropy_coef : float, default=0.01
        Weight of entropy regularization term.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.002,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 4,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.lr_actor: float = lr_actor
        self.lr_critic: float = lr_critic
        self.gamma: float = gamma
        self.lam: float = lam
        self.clip_ratio: float = clip_ratio
        self.epochs: int = epochs
        self.entropy_coef: float = entropy_coef

        rng = np.random.RandomState(seed)

        # Actor parameters
        self.W_actor: np.ndarray = (
            rng.randn(state_dim, action_dim).astype(np.float32) * 0.1
        )
        self.b_actor: np.ndarray = np.zeros(action_dim, dtype=np.float32)

        # Critic parameters
        self.W_critic: np.ndarray = rng.randn(state_dim, 1).astype(np.float32) * 0.1
        self.b_critic: np.ndarray = np.zeros(1, dtype=np.float32)

        # Trajectory storage
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        logits = np.dot(x, self.W_actor) + self.b_actor
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[0]

    def get_value(self, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        v = np.dot(x, self.W_critic) + self.b_critic
        return float(v[0, 0])

    def select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        """
        Sample action from policy.
        Returns (action, log_prob, value).
        """
        probs = self.get_action_probs(state)
        action = int(np.random.choice(self.action_dim, p=probs))
        log_prob = float(np.log(probs[action] + 1e-12))
        val = self.get_value(state)
        return action, log_prob, val

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store step transition in trajectory buffer."""
        self.states.append(np.asarray(state, dtype=np.float32).ravel())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))

    def compute_gae(self, next_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE) and Target Returns."""
        T = len(self.rewards)
        advantages: np.ndarray = np.zeros(T, dtype=np.float32)
        gae = 0.0

        vals = self.values + [next_value]

        for t in reversed(range(T)):
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * vals[t + 1] * non_terminal - vals[t]
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def update(self, next_value: float = 0.0) -> tuple[float, float]:
        """
        Execute multi-epoch PPO update on collected trajectory.
        Returns (mean_actor_loss, mean_critic_loss).
        """
        if len(self.states) == 0:
            return 0.0, 0.0

        advantages, returns = self.compute_gae(next_value)

        # Standardize advantages
        std_adv = float(np.std(advantages))
        if std_adv > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (std_adv + 1e-8)

        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int64)
        old_log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        T = len(self.states)

        for _ in range(self.epochs):
            # Critic forward & update
            v_preds = (np.dot(states_arr, self.W_critic) + self.b_critic).ravel()
            v_errors = v_preds - returns
            critic_loss = float(0.5 * np.mean(v_errors**2))
            total_critic_loss += critic_loss

            dW_critic = np.dot(states_arr.T, v_errors.reshape(-1, 1)) / T
            db_critic = np.array([np.mean(v_errors)], dtype=np.float32)
            self.W_critic -= self.lr_critic * dW_critic
            self.b_critic -= self.lr_critic * db_critic

            # Actor forward & update
            logits = np.dot(states_arr, self.W_actor) + self.b_actor
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            cur_log_probs = np.log(probs[np.arange(T), actions_arr] + 1e-12)
            ratios = np.exp(cur_log_probs - old_log_probs_arr)

            surr1 = ratios * advantages
            surr2 = (
                np.clip(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            actor_obj = np.minimum(surr1, surr2)
            actor_loss = float(-np.mean(actor_obj))
            total_actor_loss += actor_loss

            # Gradients for actor
            # When surr1 < surr2 or unclipped, d/d_ratio = advantages
            clipped = (ratios > 1.0 + self.clip_ratio) | (
                ratios < 1.0 - self.clip_ratio
            )
            use_clipped = (surr2 < surr1) & clipped
            eff_adv = np.where(use_clipped, 0.0, advantages)

            grad_logits = -probs
            grad_logits[np.arange(T), actions_arr] += 1.0
            grad_scaled = grad_logits * (ratios * eff_adv).reshape(-1, 1)

            # Entropy bonus
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            entropy_grad = -probs * (np.log(probs + 1e-12) + entropy.reshape(-1, 1))

            total_grad = grad_scaled + self.entropy_coef * entropy_grad
            dW_actor = np.dot(states_arr.T, total_grad) / T
            db_actor = np.mean(total_grad, axis=0)

            # Gradient ascent
            self.W_actor += self.lr_actor * dW_actor
            self.b_actor += self.lr_actor * db_actor

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return total_actor_loss / self.epochs, total_critic_loss / self.epochs
