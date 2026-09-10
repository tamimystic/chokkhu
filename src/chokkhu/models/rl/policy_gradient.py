from __future__ import annotations

import numpy as np


class REINFORCE:
    """
    Monte-Carlo Policy Gradient (REINFORCE) algorithm with optional baseline.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    action_dim : int
        Number of discrete actions.
    lr : float, default=0.01
        Learning rate.
    gamma : float, default=0.99
        Reward discount factor.
    normalize_returns : bool, default=True
        Whether to standardize returns across the trajectory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        gamma: float = 0.99,
        normalize_returns: bool = True,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.lr: float = lr
        self.gamma: float = gamma
        self.normalize_returns: bool = normalize_returns

        rng = np.random.RandomState(seed)
        self.W: np.ndarray = rng.randn(state_dim, action_dim).astype(np.float32) * 0.1
        self.b: np.ndarray = np.zeros(action_dim, dtype=np.float32)

        # Episode trajectory buffers
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities using softmax."""
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        logits = np.dot(x, self.W) + self.b
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[0]

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy distribution."""
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.action_dim, p=probs))

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """Record a single step in current episode trajectory."""
        self.states.append(np.asarray(state, dtype=np.float32).ravel())
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def finish_episode(self) -> float:
        """
        Compute Monte-Carlo returns and update policy parameters.
        Returns total undiscounted episode reward.
        """
        if len(self.states) == 0:
            return 0.0

        T = len(self.rewards)
        returns: np.ndarray = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = self.rewards[t] + self.gamma * running_return
            returns[t] = running_return

        total_reward = float(np.sum(self.rewards))

        if self.normalize_returns and T > 1:
            std = np.std(returns)
            if std > 1e-8:
                returns = (returns - np.mean(returns)) / (std + 1e-8)

        # Vectorized policy gradient update
        states_arr = np.array(self.states, dtype=np.float32)
        logits = np.dot(states_arr, self.W) + self.b
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Grad log pi(a|s) = e_a - pi(s)
        grad_logits = -probs
        for t, a in enumerate(self.actions):
            grad_logits[t, a] += 1.0

        # Scale by return: G_t * grad_logits
        grad_scaled = grad_logits * returns.reshape(-1, 1)

        dW = np.dot(states_arr.T, grad_scaled) / T
        db = np.sum(grad_scaled, axis=0) / T

        # Gradient Ascent step
        self.W += self.lr * dW
        self.b += self.lr * db

        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return total_reward


class ActorCritic:
    """
    Advantage Actor-Critic (A2C) with separate Actor and Critic networks.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    action_dim : int
        Number of discrete actions.
    lr_actor : float, default=0.005
        Actor learning rate.
    lr_critic : float, default=0.01
        Critic learning rate.
    gamma : float, default=0.99
        Discount factor.
    entropy_coef : float, default=0.01
        Entropy regularization weight to encourage exploration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.005,
        lr_critic: float = 0.01,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.lr_actor: float = lr_actor
        self.lr_critic: float = lr_critic
        self.gamma: float = gamma
        self.entropy_coef: float = entropy_coef

        rng = np.random.RandomState(seed)

        # Actor parameters: pi_theta(a|s)
        self.W_actor: np.ndarray = (
            rng.randn(state_dim, action_dim).astype(np.float32) * 0.1
        )
        self.b_actor: np.ndarray = np.zeros(action_dim, dtype=np.float32)

        # Critic parameters: V_phi(s)
        self.W_critic: np.ndarray = rng.randn(state_dim, 1).astype(np.float32) * 0.1
        self.b_critic: np.ndarray = np.zeros(1, dtype=np.float32)

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

    def select_action(self, state: np.ndarray) -> int:
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.action_dim, p=probs))

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[float, float]:
        """
        Perform 1-step TD Actor-Critic update.
        Returns (actor_loss, critic_loss).
        """
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)

        v_s = self.get_value(state)
        v_next = self.get_value(next_state) if not done else 0.0

        # TD error / Advantage: delta = r + gamma * V(s') - V(s)
        td_target = reward + self.gamma * v_next
        advantage = td_target - v_s

        # Critic update (MSE loss: 0.5 * (V(s) - td_target)^2)
        # dL/dV = (V(s) - td_target) = -advantage
        dW_critic = -advantage * s.T
        db_critic = np.array([-advantage], dtype=np.float32)

        self.W_critic -= self.lr_critic * dW_critic
        self.b_critic -= self.lr_critic * db_critic
        critic_loss = float(0.5 * (advantage**2))

        # Actor update
        probs = self.get_action_probs(state)
        grad_logits = -probs
        grad_logits[action] += 1.0

        # Entropy bonus: H = -sum(p * log p)
        # dH/dlogits = -p * (1 + log p) - p * sum(-p * (1 + log p))
        log_probs = np.log(probs + 1e-12)
        entropy_val: float = -1.0 * float(np.sum(probs * log_probs))
        entropy_grad = -probs * (log_probs + entropy_val)

        actor_grad = grad_logits * advantage + self.entropy_coef * entropy_grad

        dW_actor = np.dot(s.T, actor_grad.reshape(1, -1))
        db_actor = actor_grad

        # Gradient ascent on objective -> parameter += lr * grad
        self.W_actor += self.lr_actor * dW_actor
        self.b_actor += self.lr_actor * db_actor
        actor_loss = float(
            -np.log(probs[action] + 1e-12) * advantage - self.entropy_coef * entropy_val
        )

        return actor_loss, critic_loss
