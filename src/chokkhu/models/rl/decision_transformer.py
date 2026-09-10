from __future__ import annotations

import numpy as np


class DecisionTransformer:
    """
    Decision Transformer: Offline Reinforcement Learning as Autoregressive Sequence Modeling.
    Transforms trajectories (Return-To-Go, State, Action) into tokens and predicts next actions
    conditioned on target returns.

    Parameters
    ----------
    state_dim : int
        Dimension of continuous state vector.
    act_dim : int
        Dimension of continuous action vector.
    hidden_dim : int, default=64
        Transformer embedding dimension.
    max_length : int, default=20
        Context window length (K timesteps).
    max_ep_len : int, default=1000
        Maximum episode length for timestep positional embeddings.
    n_heads : int, default=4
        Number of self-attention heads.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        max_length: int = 20,
        max_ep_len: int = 1000,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = state_dim
        self.act_dim: int = act_dim
        self.hidden_dim: int = hidden_dim
        self.max_length: int = max_length
        self.max_ep_len: int = max_ep_len
        self.n_heads: int = n_heads

        rng = np.random.RandomState(seed)

        # Modality Encoders
        self.W_rtg: np.ndarray = rng.randn(1, hidden_dim).astype(np.float32) * 0.1
        self.W_state: np.ndarray = (
            rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
        )
        self.W_act: np.ndarray = rng.randn(act_dim, hidden_dim).astype(np.float32) * 0.1

        # Timestep Positional Embedding table
        self.time_emb: np.ndarray = (
            rng.randn(max_ep_len, hidden_dim).astype(np.float32) * 0.02
        )

        # Causal Attention weights (Q, K, V)
        self.W_q: np.ndarray = (
            rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        )
        self.W_k: np.ndarray = (
            rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        )
        self.W_v: np.ndarray = (
            rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        )
        self.W_o: np.ndarray = (
            rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        )

        # Action Prediction Head
        self.W_action_pred: np.ndarray = (
            rng.randn(hidden_dim, act_dim).astype(np.float32) * 0.1
        )
        self.b_action_pred: np.ndarray = np.zeros(act_dim, dtype=np.float32)

    def forward(
        self,
        returns_to_go: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass through Decision Transformer sequence.

        Parameters
        ----------
        returns_to_go : np.ndarray
            Shape [B, T, 1]
        states : np.ndarray
            Shape [B, T, state_dim]
        actions : np.ndarray
            Shape [B, T, act_dim]
        timesteps : np.ndarray
            Shape [B, T]

        Returns
        -------
        action_preds : np.ndarray
            Shape [B, T, act_dim]
        """
        B, T, _ = states.shape

        # Modality projections
        rtg_emb = np.dot(returns_to_go, self.W_rtg)  # [B, T, H]
        state_emb = np.dot(states, self.W_state)  # [B, T, H]
        act_emb = np.dot(actions, self.W_act)  # [B, T, H]

        # Add timestep embeddings
        t_emb = self.time_emb[np.clip(timesteps, 0, self.max_ep_len - 1)]  # [B, T, H]
        rtg_emb += t_emb
        state_emb += t_emb
        act_emb += t_emb

        # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        seq_tokens = np.zeros((B, 3 * T, self.hidden_dim), dtype=np.float32)
        seq_tokens[:, 0::3, :] = rtg_emb
        seq_tokens[:, 1::3, :] = state_emb
        seq_tokens[:, 2::3, :] = act_emb

        # Causal Self-Attention
        Q = np.dot(seq_tokens, self.W_q)
        K = np.dot(seq_tokens, self.W_k)
        V = np.dot(seq_tokens, self.W_v)

        d_k = self.hidden_dim
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # Causal triangular mask
        seq_len = 3 * T
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores += mask

        # Softmax
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        attn_out = np.matmul(attn_weights, V)
        proj_out = np.dot(attn_out, self.W_o)

        # Predict actions from state token positions (indices: 1, 4, 7, ... -> 1::3)
        state_reps = proj_out[:, 1::3, :]  # [B, T, H]
        action_preds = np.tanh(
            np.dot(state_reps, self.W_action_pred) + self.b_action_pred
        )

        return action_preds

    def get_action(
        self,
        returns_to_go: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Predict action for the most recent timestep."""
        rtg = np.asarray(returns_to_go, dtype=np.float32)
        s = np.asarray(states, dtype=np.float32)
        a = np.asarray(actions, dtype=np.float32)
        t = np.asarray(timesteps, dtype=np.int64)

        if s.ndim == 2:
            s = s.reshape(1, *s.shape)
            rtg = rtg.reshape(1, *rtg.shape)
            a = a.reshape(1, *a.shape)
            t = t.reshape(1, *t.shape)

        # Truncate context to max_length
        if s.shape[1] > self.max_length:
            s = s[:, -self.max_length :]
            rtg = rtg[:, -self.max_length :]
            a = a[:, -self.max_length :]
            t = t[:, -self.max_length :]

        action_preds = self.forward(rtg, s, a, t)
        return action_preds[0, -1]  # Latest action prediction
