r"""QMIX Multi-Agent Value Factorization Architecture in pure NumPy."""

from __future__ import annotations


import numpy as np


class QMIX:
    r"""QMIX Multi-Agent Reinforcement Learning Mixing Network in pure NumPy (Rashid et al. 2018).

    Implements Centralized Training with Decentralized Execution (CTDE) using non-negative
    hypernetworks to enforce monotonic value factorization: $\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$.

    Parameters
    ----------
    n_agents : int, default=4
        Number of cooperative agents in multi-agent environment.
    state_dim : int, default=32
        Global environment state feature dimensionality.
    mixing_embed_dim : int, default=32
        Internal hidden dimensionality of monotonic mixing network.
    seed : int, default=42
    """

    def __init__(
        self,
        n_agents: int = 4,
        state_dim: int = 32,
        mixing_embed_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)

        self.rng = np.random.default_rng(seed)
        scale_s = np.sqrt(2.0 / state_dim)

        # 1. Hypernetwork 1: state -> W1 (n_agents * mixing_embed_dim)
        self.hyper_w1 = self.rng.normal(
            0, scale_s, size=(state_dim, self.n_agents * self.mixing_embed_dim)
        ).astype(np.float32)
        self.hyper_b1: np.ndarray = np.zeros(
            self.n_agents * self.mixing_embed_dim, dtype=np.float32
        )

        # Bias 1: state -> b1 (mixing_embed_dim)
        self.hyper_bias1 = self.rng.normal(
            0, scale_s, size=(state_dim, self.mixing_embed_dim)
        ).astype(np.float32)
        self.b_bias1: np.ndarray = np.zeros(self.mixing_embed_dim, dtype=np.float32)

        # 2. Hypernetwork 2: state -> W2 (mixing_embed_dim * 1)
        self.hyper_w2 = self.rng.normal(
            0, scale_s, size=(state_dim, self.mixing_embed_dim)
        ).astype(np.float32)
        self.hyper_b2: np.ndarray = np.zeros(self.mixing_embed_dim, dtype=np.float32)

        # State Value baseline V(s): state -> (1)
        self.w_val = self.rng.normal(0, scale_s, size=(state_dim, 1)).astype(np.float32)
        self.b_val: np.ndarray = np.zeros(1, dtype=np.float32)

    def forward(self, agent_qs: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Computes centralized joint value Q_tot from individual utilities and state.

        Parameters
        ----------
        agent_qs : np.ndarray
            Individual utilities matrix of shape (B, n_agents) or (n_agents,).
        states : np.ndarray
            Global state matrix of shape (B, state_dim) or (state_dim,).

        Returns
        -------
        np.ndarray
            Joint Q_tot values of shape (B, 1).
        """
        qs = np.asarray(agent_qs, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs[np.newaxis, :]
        st = np.asarray(states, dtype=np.float32)
        if st.ndim == 1:
            st = st[np.newaxis, :]

        B = len(qs)

        # 1. Hypernetwork 1: Generate non-negative weights W1: (B, n_agents, mixing_embed_dim)
        raw_w1 = np.matmul(st, self.hyper_w1) + self.hyper_b1
        w1 = np.abs(raw_w1).reshape(B, self.n_agents, self.mixing_embed_dim)

        b1 = (np.matmul(st, self.hyper_bias1) + self.b_bias1).reshape(
            B, 1, self.mixing_embed_dim
        )

        # 2. First mixing layer: ELU(qs @ W1 + b1)
        # qs: (B, 1, n_agents), w1: (B, n_agents, embed_dim) -> (B, 1, embed_dim)
        qs_reshaped = qs.reshape(B, 1, self.n_agents)
        hidden = np.matmul(qs_reshaped, w1) + b1
        hidden_act = np.where(
            hidden > 0, hidden, np.exp(np.clip(hidden, -15.0, 0.0)) - 1.0
        )

        # 3. Hypernetwork 2: Generate non-negative weights W2: (B, mixing_embed_dim, 1)
        raw_w2 = np.matmul(st, self.hyper_w2) + self.hyper_b2
        w2 = np.abs(raw_w2).reshape(B, self.mixing_embed_dim, 1)

        # State value bias V(s)
        v_s = (np.matmul(st, self.w_val) + self.b_val).reshape(B, 1, 1)

        # 4. Joint Q_tot computation: hidden @ W2 + V(s)
        q_tot = np.matmul(hidden_act, w2) + v_s
        return q_tot.reshape(B, 1)

    def td_loss(
        self,
        agent_qs: np.ndarray,
        states: np.ndarray,
        rewards: np.ndarray,
        next_agent_qs: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
    ) -> float:
        """Computes Mean Squared TD error loss for QMIX network."""
        current_q_tot = self.forward(agent_qs, states).flatten()
        target_q_tot = self.forward(next_agent_qs, next_states).flatten()

        rew = np.asarray(rewards, dtype=np.float32).flatten()
        don = np.asarray(dones, dtype=np.float32).flatten()

        targets = rew + gamma * (1.0 - don) * target_q_tot
        return float(np.mean((current_q_tot - targets) ** 2))
