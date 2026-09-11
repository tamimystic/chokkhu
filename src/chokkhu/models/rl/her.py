"""Hindsight Experience Replay (HER) and Goal-Conditioned Reinforcement Learning.

Formulated from first principles in pure NumPy following Andrychowicz et al. (NeurIPS 2017).
Enables sample-efficient learning in sparse-reward environments by converting past failures
into successful counterfactual experiences via virtual goal replay.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


class HindsightExperienceReplay:
    r"""Hindsight Experience Replay (HER) Buffer.

    Stores goal-oriented trajectories and synthesizes virtual experiences by substituting
    failed goals with actually achieved end states $g' = s_{t'}$.

    Parameters
    ----------
    capacity : int, default=50000
        Maximum replay buffer transitions.
    strategy : str, default="future"
        Replay strategy: "future" (samples future achieved goals), "final" (uses episode terminal state),
        or "random" (samples random achieved state from buffer).
    k : int, default=4
        Ratio of synthesized hindsight transitions per original transition.
    reward_func : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Sparse binary reward function $r(s, g)$. If None, uses negative Euclidean distance threshold.
    distance_threshold : float, default=0.05
        Success tolerance distance $\epsilon$ for default reward function.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        capacity: int = 50000,
        strategy: str = "future",
        k: int = 4,
        reward_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        distance_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.capacity = max(100, int(capacity))
        self.strategy = strategy
        self.k = max(1, int(k))
        self.distance_threshold = float(distance_threshold)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.reward_func = (
            reward_func if reward_func is not None else self._default_reward
        )

        # Flat transition storage: (state, action, reward, next_state, done, goal)
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.goals: List[np.ndarray] = []

    def _default_reward(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Default sparse binary reward: 0.0 if close enough to goal, else -1.0."""
        dist = float(np.linalg.norm(state - goal))
        return 0.0 if dist <= self.distance_threshold else -1.0

    def add_episode(
        self,
        episode_states: List[np.ndarray],
        episode_actions: List[int],
        desired_goal: np.ndarray,
    ) -> None:
        """Process and store a full episode trajectory with hindsight goal substitution.

        Parameters
        ----------
        episode_states : List[np.ndarray] of length T+1
            Sequence of visited states including initial state and transitions: [s_0, s_1, ..., s_T].
        episode_actions : List[int] of length T
            Sequence of taken actions [a_0, ..., a_{T-1}].
        desired_goal : np.ndarray
            Original target goal for the episode.
        """
        T = len(episode_actions)
        goal_vec = np.asarray(desired_goal, dtype=float)

        for t in range(T):
            s_t = episode_states[t]
            a_t = episode_actions[t]
            s_next = episode_states[t + 1]

            # 1. Store original transition with original desired goal
            r_t = self.reward_func(s_next, goal_vec)
            done_t = bool(r_t == 0.0)

            self._append_transition(s_t, a_t, r_t, s_next, done_t, goal_vec)

            # 2. Synthesize K hindsight transitions
            for _ in range(self.k):
                if self.strategy == "future":
                    # Sample future index t' in [t+1, T]
                    t_prime = int(self.rng.randint(t + 1, T + 1))
                    hindsight_goal = episode_states[t_prime]
                elif self.strategy == "final":
                    hindsight_goal = episode_states[-1]
                else:  # random
                    rand_idx = int(self.rng.randint(0, T + 1))
                    hindsight_goal = episode_states[rand_idx]

                r_her = self.reward_func(s_next, hindsight_goal)
                done_her = bool(r_her == 0.0)
                self._append_transition(
                    s_t, a_t, r_her, s_next, done_her, hindsight_goal
                )

    def _append_transition(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        done: bool,
        g: np.ndarray,
    ) -> None:
        if len(self.states) >= self.capacity:
            idx = int(self.rng.randint(0, len(self.states)))
            self.states[idx] = s
            self.actions[idx] = a
            self.rewards[idx] = r
            self.next_states[idx] = s_next
            self.dones[idx] = done
            self.goals[idx] = g
        else:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s_next)
            self.dones.append(done)
            self.goals.append(g)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a uniform mini-batch of transitions."""
        n = len(self.states)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = self.rng.choice(n, size=min(batch_size, n), replace=False)

        batch_s = np.array([self.states[i] for i in indices])
        batch_a = np.array([self.actions[i] for i in indices])
        batch_r = np.array([self.rewards[i] for i in indices])
        batch_s_next = np.array([self.next_states[i] for i in indices])
        batch_done = np.array([self.dones[i] for i in indices], dtype=float)
        batch_g = np.array([self.goals[i] for i in indices])

        return batch_s, batch_a, batch_r, batch_s_next, batch_done, batch_g

    def __len__(self) -> int:
        return len(self.states)


class GoalConditionedDQN:
    r"""Goal-Conditioned Deep Q-Network Q(s, a, g).

    Learns multi-goal value representations $[s; g] \to Q(s, \cdot, g)$ for sparse reward tasks.

    Parameters
    ----------
    state_dim : int
        Observation state dimension.
    goal_dim : int
        Goal state dimension.
    num_actions : int
        Discrete action space size.
    hidden_dim : int, default=64
        Neural network hidden layer width.
    lr : float, default=0.01
        Learning rate.
    gamma : float, default=0.98
        Bellman discount factor.
    epsilon : float, default=0.1
        Epsilon-greedy exploration probability.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        lr: float = 0.01,
        gamma: float = 0.98,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.num_actions = int(num_actions)
        self.input_dim = self.state_dim + self.goal_dim
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # 2-layer MLP Q-network
        self.W1 = self.rng.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self.rng.randn(self.hidden_dim, self.num_actions) * 0.1
        self.b2 = np.zeros(self.num_actions)

    def _forward(self, sg: np.ndarray) -> np.ndarray:
        """Compute Q(s, a, g) for all actions."""
        h = np.dot(sg, self.W1) + self.b1
        h_relu = np.maximum(0.0, h)
        return np.dot(h_relu, self.W2) + self.b2

    def select_action(self, state: np.ndarray, goal: np.ndarray) -> int:
        """Select epsilon-greedy action."""
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, self.num_actions))

        sg = np.concatenate([state, goal])[None, :]
        q_values = self._forward(sg)[0]
        return int(np.argmax(q_values))

    def train_step(
        self, buffer: HindsightExperienceReplay, batch_size: int = 32
    ) -> float:
        """Execute one Bellman Q-learning gradient step on HER buffer."""
        if len(buffer) < batch_size:
            return 0.0

        batch_s, batch_a, batch_r, batch_s_next, batch_done, batch_g = (
            buffer.sample_batch(batch_size)
        )

        sg = np.concatenate([batch_s, batch_g], axis=-1)
        sg_next = np.concatenate([batch_s_next, batch_g], axis=-1)

        # Target: r + gamma * max_a' Q(s', a', g) * (1 - done)
        q_next = self._forward(sg_next)
        max_q_next = np.max(q_next, axis=-1)
        targets = batch_r + self.gamma * max_q_next * (1.0 - batch_done)

        # Current Q-values
        h1 = np.dot(sg, self.W1) + self.b1
        relu_mask = h1 > 0
        h1_relu = np.maximum(0.0, h1)
        q_pred = np.dot(h1_relu, self.W2) + self.b2

        # Compute TD error
        loss = 0.0
        grad_out = np.zeros_like(q_pred)
        for i in range(len(batch_a)):
            a = batch_a[i]
            diff = q_pred[i, a] - targets[i]
            loss += diff**2
            grad_out[i, a] = diff

        loss = float(loss / len(batch_a))
        grad_out = grad_out / len(batch_a)

        # Backpropagation
        dW2 = np.dot(h1_relu.T, grad_out)
        db2 = np.sum(grad_out, axis=0)

        dh1 = np.dot(grad_out, self.W2.T) * relu_mask
        dW1 = np.dot(sg.T, dh1)
        db1 = np.sum(dh1, axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return loss
