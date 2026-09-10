from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """
    Uniform Experience Replay Buffer for Q-learning and Deep RL.
    Stores transitions (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int = 10000, state_dim: int = 4) -> None:
        self.capacity: int = capacity
        self.state_dim: int = state_dim
        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.ptr: int = 0
        self.size: int = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32).ravel()
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32).ravel()
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a batch of transitions."""
        if self.size < batch_size:
            raise ValueError(
                f"Replay buffer contains only {self.size} samples, requested {batch_size}"
            )
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer with proportional prioritization.
    Computes importance sampling weights: w_i = (N * P(i))^(-beta) / max(w).
    """

    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 4,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity: int = capacity
        self.state_dim: int = state_dim
        self.alpha: float = alpha
        self.beta: float = beta
        self.beta_increment: float = beta_increment
        self.epsilon: float = epsilon

        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0
        self.max_priority: float = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with maximum current priority."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32).ravel()
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32).ravel()
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Sample a priority-weighted batch with importance sampling weights.
        Returns (states, actions, rewards, next_states, dones, indices, weights).
        """
        if self.size < batch_size:
            raise ValueError(
                f"PER buffer contains only {self.size} samples, requested {batch_size}"
            )

        priorities = self.priorities[: self.size]
        probs = priorities**self.alpha
        probs /= np.sum(probs)

        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # Importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= np.max(weights) + 1e-8

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on absolute TD errors."""
        abs_errors = np.abs(td_errors).ravel() + self.epsilon
        self.priorities[indices] = abs_errors
        self.max_priority = max(self.max_priority, float(np.max(abs_errors)))

    def __len__(self) -> int:
        return self.size


class _MLP:
    """Pure NumPy 2-layer MLP Q-network with Adam optimizer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Weights initialization (He normal)
        self.W1: np.ndarray = rng.randn(state_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / state_dim)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2: np.ndarray = rng.randn(hidden_dim, action_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b2: np.ndarray = np.zeros(action_dim, dtype=np.float32)

        # Adam optimizer state
        self.mW1: np.ndarray = np.zeros_like(self.W1)
        self.vW1: np.ndarray = np.zeros_like(self.W1)
        self.mb1: np.ndarray = np.zeros_like(self.b1)
        self.vb1: np.ndarray = np.zeros_like(self.b1)

        self.mW2: np.ndarray = np.zeros_like(self.W2)
        self.vW2: np.ndarray = np.zeros_like(self.W2)
        self.mb2: np.ndarray = np.zeros_like(self.b2)
        self.vb2: np.ndarray = np.zeros_like(self.b2)
        self.t: int = 0

    def forward(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (q_values, hidden_activation)."""
        x = np.asarray(state, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)  # ReLU
        q = np.dot(h, self.W2) + self.b2
        return q, h

    def backward_and_update(
        self,
        states: np.ndarray,
        hidden: np.ndarray,
        grad_out: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Compute gradients and perform Adam optimization step."""
        if weights is not None:
            grad_out = grad_out * weights.reshape(-1, 1)

        batch_size = states.shape[0]

        # Backprop through layer 2
        dW2 = np.dot(hidden.T, grad_out) / batch_size
        db2 = np.sum(grad_out, axis=0) / batch_size

        # Backprop through ReLU and layer 1
        dhidden = np.dot(grad_out, self.W2.T)
        dhidden[hidden <= 0] = 0.0

        dW1 = np.dot(states.T, dhidden) / batch_size
        db1 = np.sum(dhidden, axis=0) / batch_size

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for param, grad, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
        ]:
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_weights_from(self, source: _MLP) -> None:
        """Hard copy weights from source network."""
        self.W1 = np.copy(source.W1)
        self.b1 = np.copy(source.b1)
        self.W2 = np.copy(source.W2)
        self.b2 = np.copy(source.b2)


class DQN:
    """
    Deep Q-Network (DQN) with target network and epsilon-greedy exploration.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    action_dim : int
        Number of discrete actions.
    hidden_dim : int, default=64
        Number of units in hidden layer.
    lr : float, default=0.001
        Learning rate for Adam optimizer.
    gamma : float, default=0.99
        Discount factor for future rewards.
    epsilon : float, default=1.0
        Initial exploration probability.
    epsilon_min : float, default=0.05
        Minimum exploration probability.
    epsilon_decay : float, default=0.995
        Multiplicative decay factor per episode/step.
    target_update_freq : int, default=100
        Frequency of updating target network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count: int = 0

        self.online_net = _MLP(state_dim, action_dim, hidden_dim, lr=lr, seed=seed)
        self.target_net = _MLP(state_dim, action_dim, hidden_dim, lr=lr, seed=seed + 1)
        self.target_net.copy_weights_from(self.online_net)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        q_values, _ = self.online_net.forward(state)
        return int(np.argmax(q_values[0]))

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Perform a single DQN gradient descent training step.
        Returns (loss, td_errors).
        """
        batch_size = states.shape[0]

        # Forward pass on online network for current states
        q_pred, hidden = self.online_net.forward(states)

        # Forward pass on target network for next states
        q_next, _ = self.target_net.forward(next_states)
        max_next_q = np.max(q_next, axis=1)

        # Bellman target: y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        # Q-values of chosen actions
        q_action = q_pred[np.arange(batch_size), actions]
        td_errors = q_action - targets

        # Loss (MSE)
        if weights is not None:
            loss = float(np.mean(weights * (td_errors**2)))
        else:
            loss = float(np.mean(td_errors**2))

        # Output gradients for backprop: dL/dq
        grad_out = np.zeros_like(q_pred)
        grad_out[np.arange(batch_size), actions] = td_errors

        # Update online network
        self.online_net.backward_and_update(states, hidden, grad_out, weights=weights)

        # Periodic target network sync
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss, td_errors


class DoubleDQN(DQN):
    """
    Double Deep Q-Network (Double DQN).
    Decouples action selection (online network) from evaluation (target network)
    to mitigate Q-value overestimation bias.
    """

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        batch_size = states.shape[0]

        # Forward pass on online network for current states
        q_pred, hidden = self.online_net.forward(states)

        # Select greedy action using ONLINE network
        q_next_online, _ = self.online_net.forward(next_states)
        best_actions = np.argmax(q_next_online, axis=1)

        # Evaluate target Q-value using TARGET network at online best action
        q_next_target, _ = self.target_net.forward(next_states)
        target_next_q = q_next_target[np.arange(batch_size), best_actions]

        # Double DQN Bellman target
        targets = rewards + self.gamma * (1.0 - dones) * target_next_q

        q_action = q_pred[np.arange(batch_size), actions]
        td_errors = q_action - targets

        if weights is not None:
            loss = float(np.mean(weights * (td_errors**2)))
        else:
            loss = float(np.mean(td_errors**2))

        grad_out = np.zeros_like(q_pred)
        grad_out[np.arange(batch_size), actions] = td_errors

        self.online_net.backward_and_update(states, hidden, grad_out, weights=weights)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss, td_errors


class DuelingDQN:
    """
    Dueling Deep Q-Network (Dueling DQN).
    Decomposes state-action value into state value V(s) and advantage A(s, a):
    Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a')).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        rng = np.random.RandomState(seed)

        # Feature extractor
        self.W_feat: np.ndarray = rng.randn(state_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / state_dim)
        self.b_feat: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        # Value stream V(s)
        self.W_v: np.ndarray = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(
            2.0 / hidden_dim
        )
        self.b_v: np.ndarray = np.zeros(1, dtype=np.float32)

        # Advantage stream A(s, a)
        self.W_a: np.ndarray = rng.randn(hidden_dim, action_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b_a: np.ndarray = np.zeros(action_dim, dtype=np.float32)

    def forward(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass.
        Returns (Q, feat, V, A).
        """
        x = np.asarray(state, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        feat = np.maximum(0.0, np.dot(x, self.W_feat) + self.b_feat)  # [B, H]
        v = np.dot(feat, self.W_v) + self.b_v  # [B, 1]
        a = np.dot(feat, self.W_a) + self.b_a  # [B, Action]

        # Aggregation: Q = V + (A - mean(A))
        q = v + (a - np.mean(a, axis=1, keepdims=True))
        return q, feat, v, a

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        q, _, _, _ = self.forward(state)
        return int(np.argmax(q[0]))

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        batch_size = states.shape[0]
        q_pred, feat, v, a = self.forward(states)
        q_next, _, _, _ = self.forward(next_states)
        max_next_q = np.max(q_next, axis=1)

        targets = rewards + self.gamma * (1.0 - dones) * max_next_q
        q_action = q_pred[np.arange(batch_size), actions]
        td_errors = q_action - targets
        loss = float(np.mean(td_errors**2))

        # Gradient with respect to Q
        grad_q = np.zeros_like(q_pred)
        grad_q[np.arange(batch_size), actions] = td_errors / batch_size

        # Gradients for V and A streams
        grad_v = np.sum(grad_q, axis=1, keepdims=True)  # [B, 1]
        grad_a = grad_q - np.mean(grad_q, axis=1, keepdims=True)  # [B, Action]

        dW_v = np.dot(feat.T, grad_v)
        db_v = np.sum(grad_v, axis=0)

        dW_a = np.dot(feat.T, grad_a)
        db_a = np.sum(grad_a, axis=0)

        dfeat = np.dot(grad_v, self.W_v.T) + np.dot(grad_a, self.W_a.T)
        dfeat[feat <= 0] = 0.0

        dW_feat = np.dot(states.T, dfeat)
        db_feat = np.sum(dfeat, axis=0)

        # SGD step
        self.W_v -= self.lr * dW_v
        self.b_v -= self.lr * db_v
        self.W_a -= self.lr * dW_a
        self.b_a -= self.lr * db_a
        self.W_feat -= self.lr * dW_feat
        self.b_feat -= self.lr * db_feat

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss
