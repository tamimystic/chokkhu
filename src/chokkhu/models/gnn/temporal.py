from __future__ import annotations

import numpy as np


class TemporalGraphNetwork:
    """
    Temporal Graph Network (TGN) for Continuous-Time Dynamic Graphs.
    Maintains dynamic node memories updated sequentially by interaction events.

    Parameters
    ----------
    node_dim : int
        Static node feature dimension.
    edge_dim : int
        Edge feature dimension.
    memory_dim : int, default=32
        Dimensionality of dynamic node memory state.
    time_dim : int, default=16
        Dimension of Bochner time encoding.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        memory_dim: int = 32,
        time_dim: int = 16,
        seed: int = 42,
    ) -> None:
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        rng = np.random.RandomState(seed)

        # Time encoding frequencies (Fourier / Bochner)
        self.time_freqs: np.ndarray = rng.randn(time_dim).astype(np.float32) * 0.1

        # Message function MLP: (s_src, s_dst, dt, e_feat) -> m_ij
        msg_in_dim = 2 * memory_dim + time_dim + edge_dim
        self.W_msg: np.ndarray = rng.randn(msg_in_dim, memory_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / msg_in_dim)
        self.b_msg: np.ndarray = np.zeros(memory_dim, dtype=np.float32)

        # GRU Memory Updater: (m_i, s_i) -> s_i_new
        # Update gate z, Reset gate r, Candidate h~
        gru_in_dim = memory_dim + memory_dim
        self.W_z: np.ndarray = (
            rng.randn(gru_in_dim, memory_dim).astype(np.float32) * 0.1
        )
        self.W_r: np.ndarray = (
            rng.randn(gru_in_dim, memory_dim).astype(np.float32) * 0.1
        )
        self.W_c: np.ndarray = (
            rng.randn(gru_in_dim, memory_dim).astype(np.float32) * 0.1
        )

    def encode_time(self, dt: np.ndarray) -> np.ndarray:
        """Harmonic Bochner sinusoidal time encoding."""
        dt_arr = np.asarray(dt, dtype=np.float32).reshape(-1, 1)
        phases = np.dot(dt_arr, self.time_freqs.reshape(1, -1))
        return np.cos(phases)

    def compute_messages(
        self,
        src_mem: np.ndarray,
        dst_mem: np.ndarray,
        dt: np.ndarray,
        edge_feat: np.ndarray,
    ) -> np.ndarray:
        """Compute interaction messages for temporal events."""
        time_enc = self.encode_time(dt)
        raw_input = np.concatenate([src_mem, dst_mem, time_enc, edge_feat], axis=1)
        return np.maximum(0.0, np.dot(raw_input, self.W_msg) + self.b_msg)

    def update_memory(
        self,
        prev_mem: np.ndarray,
        messages: np.ndarray,
    ) -> np.ndarray:
        """GRU memory state transition."""
        gru_in = np.concatenate([messages, prev_mem], axis=1)

        # Sigmoid gate activations
        z = 1.0 / (1.0 + np.exp(-np.dot(gru_in, self.W_z)))
        r = 1.0 / (1.0 + np.exp(-np.dot(gru_in, self.W_r)))

        cand_in = np.concatenate([messages, r * prev_mem], axis=1)
        c = np.tanh(np.dot(cand_in, self.W_c))

        new_mem = (1.0 - z) * prev_mem + z * c
        return new_mem
