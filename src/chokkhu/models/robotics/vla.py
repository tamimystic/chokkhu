"""Vision-Language-Action (VLA) & Action Chunking with Transformers (ACT) in Pure NumPy.

References:
- Brohan et al. (2023): "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (CoRL 2023).
- Kim et al. (2024): "OpenVLA: An Open-Source Vision-Language-Action Model".
- Zhao et al. (2023): "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)" (RSS 2023).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np


class ActionTokenizer:
    """Discrete 7-DoF Robot Action Tokenizer (RT-2 / OpenVLA style).

    Discretizes continuous 7-DoF robot commands:
    [dx, dy, dz, droll, dpitch, dyaw, gripper] into K=256 uniform discrete tokens per dimension.
    """

    def __init__(
        self,
        num_bins: int = 256,
        min_action: Union[float, np.ndarray] = -1.0,
        max_action: Union[float, np.ndarray] = 1.0,
        action_dim: int = 7,
    ) -> None:
        self.num_bins = int(num_bins)
        self.action_dim = int(action_dim)

        if isinstance(min_action, (int, float)):
            self.min_action: np.ndarray = np.full(
                self.action_dim, float(min_action), dtype=np.float64
            )
        else:
            self.min_action = np.asarray(min_action, dtype=np.float64)

        if isinstance(max_action, (int, float)):
            self.max_action: np.ndarray = np.full(
                self.action_dim, float(max_action), dtype=np.float64
            )
        else:
            self.max_action = np.asarray(max_action, dtype=np.float64)

        # Bin step sizes and centers per dimension
        self.bin_step = (self.max_action - self.min_action) / float(self.num_bins)
        self.bin_centers: np.ndarray = np.zeros(
            (self.action_dim, self.num_bins), dtype=np.float64
        )
        for d in range(self.action_dim):
            self.bin_centers[d] = (
                self.min_action[d] + (np.arange(self.num_bins) + 0.5) * self.bin_step[d]
            )

    def encode(self, continuous_action: np.ndarray) -> np.ndarray:
        """Discretize continuous actions into integer token IDs in [0, num_bins - 1].

        Args:
            continuous_action: Array of shape (..., action_dim)

        Returns:
            Token array of shape (..., action_dim) with integer values in [0, num_bins - 1]
        """
        act = np.asarray(continuous_action, dtype=np.float64)
        clipped = np.clip(act, self.min_action, self.max_action)

        # Normalized to [0, num_bins - 1]
        normalized = (clipped - self.min_action) / (
            self.max_action - self.min_action + 1e-12
        )
        tokens = np.floor(normalized * self.num_bins).astype(np.int64)
        return np.clip(tokens, 0, self.num_bins - 1)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """De-quantize discrete token IDs back into continuous action estimates using bin centers.

        Args:
            tokens: Array of shape (..., action_dim) with integer token IDs

        Returns:
            Continuous action estimates of shape (..., action_dim)
        """
        toks = np.asarray(tokens, dtype=np.int64)
        toks_clipped = np.clip(toks, 0, self.num_bins - 1)

        # Reconstruct coordinates from bin centers
        orig_shape = toks.shape
        flat_toks = toks_clipped.reshape(-1, self.action_dim)
        reconstructed = np.zeros_like(flat_toks, dtype=np.float64)

        for d in range(self.action_dim):
            reconstructed[:, d] = self.bin_centers[d, flat_toks[:, d]]

        return reconstructed.reshape(orig_shape)


class TemporalEnsembler:
    """Exponential Receding Horizon Temporal Ensembler for Action Chunking (Zhao et al., 2023).

    Maintains a rolling buffer of predicted future action trajectories and computes an
    exponentially decaying weighted average to guarantee silky-smooth trajectory execution.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 7,
        exp_weight: float = 0.01,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.exp_weight = float(exp_weight)

        # Buffer: list of predicted action chunks, stored as tuples (start_timestep, chunk_array)
        self.predicted_chunks: List[Tuple[int, np.ndarray]] = []
        self.current_step = 0

    def reset(self) -> None:
        """Reset internal history buffer for a new robotic episode."""
        self.predicted_chunks.clear()
        self.current_step = 0

    def update(self, action_chunk: np.ndarray) -> np.ndarray:
        """Add a newly predicted action chunk of shape (chunk_size, action_dim) and return the smoothed action for current timestep.

        Args:
            action_chunk: Predicted future actions of shape (chunk_size, action_dim)

        Returns:
            Smoothed action for the current timestep of shape (action_dim,)
        """
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected chunk shape ({self.chunk_size}, {self.action_dim}), got {chunk.shape}"
            )

        self.predicted_chunks.append((self.current_step, chunk))

        # Filter out chunks that are older than chunk_size steps
        self.predicted_chunks = [
            (t0, c)
            for (t0, c) in self.predicted_chunks
            if (self.current_step - t0) < self.chunk_size
        ]

        # Aggregate overlapping predictions for current_step
        weighted_action: np.ndarray = np.zeros(self.action_dim, dtype=np.float64)
        total_weight = 0.0

        for t0, c in self.predicted_chunks:
            offset = self.current_step - t0
            # Exponential decay weight: w_i = exp(-exp_weight * offset)
            weight = np.exp(-self.exp_weight * float(offset))
            weighted_action += weight * c[offset]
            total_weight += weight

        smoothed_action = weighted_action / max(total_weight, 1e-12)
        self.current_step += 1
        return smoothed_action


class ActionChunkingTransformer:
    """Action Chunking with Transformers (ACT) Policy in Pure NumPy.

    Predicts a sequence of future action steps A_{t:t+k} conditioned on visual tokens and proprioception,
    with an integrated TemporalEnsembler for smooth multi-step robot control.
    """

    def __init__(
        self,
        proprio_dim: int = 7,
        visual_dim: int = 64,
        action_dim: int = 7,
        chunk_size: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        random_state: int = 42,
    ) -> None:
        self.proprio_dim = int(proprio_dim)
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Proprioception and Visual encoders
        self.W_proprio: np.ndarray = (
            self.rng.randn(proprio_dim, hidden_dim).astype(np.float64) * 0.1
        )
        self.b_proprio: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)

        self.W_visual: np.ndarray = (
            self.rng.randn(visual_dim, hidden_dim).astype(np.float64) * 0.1
        )
        self.b_visual: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)

        # Causal / Learned Position Embeddings for Chunk steps
        self.pos_embed: np.ndarray = (
            self.rng.randn(chunk_size, hidden_dim).astype(np.float64) * 0.05
        )

        # Decoder Feed-Forward Layers
        self.W_dec1: np.ndarray = (
            self.rng.randn(hidden_dim, hidden_dim).astype(np.float64) * 0.1
        )
        self.b_dec1: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)
        self.W_dec2: np.ndarray = (
            self.rng.randn(hidden_dim, action_dim).astype(np.float64) * 0.1
        )
        self.b_dec2: np.ndarray = np.zeros(action_dim, dtype=np.float64)

        # Temporal ensembling buffer
        self.ensembler = TemporalEnsembler(chunk_size=chunk_size, action_dim=action_dim)

    def forward_chunk(
        self,
        visual_features: np.ndarray,
        proprio_state: np.ndarray,
        latent_style: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict a full action chunk A_{t:t+k} of shape (batch_size, chunk_size, action_dim) or (chunk_size, action_dim).

        Args:
            visual_features: Vision features (..., visual_dim)
            proprio_state: Robot joint state (..., proprio_dim)
            latent_style: Optional latent style vector (..., latent_dim)

        Returns:
            Action chunk of shape (..., chunk_size, action_dim)
        """
        vis = np.asarray(visual_features, dtype=np.float64)
        prop = np.asarray(proprio_state, dtype=np.float64)

        # Encode inputs
        h_vis = vis @ self.W_visual + self.b_visual
        h_vis = h_vis / (1.0 + np.exp(-np.clip(h_vis, -15.0, 15.0)))

        h_prop = prop @ self.W_proprio + self.b_proprio
        h_prop = h_prop / (1.0 + np.exp(-np.clip(h_prop, -15.0, 15.0)))

        # Combined context embedding
        h_context = h_vis + h_prop  # (..., hidden_dim)

        # Broadcast across chunk steps with position embeddings
        if h_context.ndim == 1:
            h_chunk = (
                h_context[np.newaxis, :] + self.pos_embed
            )  # (chunk_size, hidden_dim)
        else:
            h_chunk = (
                h_context[..., np.newaxis, :] + self.pos_embed
            )  # (B, chunk_size, hidden_dim)

        # Decoder pass
        z1 = h_chunk @ self.W_dec1 + self.b_dec1
        a1 = z1 / (1.0 + np.exp(-np.clip(z1, -15.0, 15.0)))
        chunk_actions = a1 @ self.W_dec2 + self.b_dec2  # (..., chunk_size, action_dim)

        return chunk_actions

    def step(
        self,
        visual_features: np.ndarray,
        proprio_state: np.ndarray,
    ) -> np.ndarray:
        """Execute one closed-loop environment step with smooth temporal ensembling.

        Args:
            visual_features: Current visual observation (visual_dim,)
            proprio_state: Current joint state (proprio_dim,)

        Returns:
            Smoothed action for immediate motor execution of shape (action_dim,)
        """
        chunk = self.forward_chunk(visual_features, proprio_state)
        if chunk.ndim == 3:
            chunk = chunk[0]
        return self.ensembler.update(chunk)

    def reset(self) -> None:
        """Reset temporal ensembling for new episode."""
        self.ensembler.reset()


class OpenVLAPolicy:
    """End-to-End Vision-Language-Action Policy (VLA) uniting Multi-Modal Vision, Text & Robot Control."""

    def __init__(
        self,
        vocab_size: int = 512,
        action_bins: int = 256,
        embed_dim: int = 64,
        action_dim: int = 7,
        chunk_size: int = 8,
        random_state: int = 42,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.action_bins = int(action_bins)
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.rng = np.random.RandomState(random_state)

        self.tokenizer = ActionTokenizer(num_bins=action_bins, action_dim=action_dim)
        self.act_model = ActionChunkingTransformer(
            proprio_dim=action_dim,
            visual_dim=embed_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=embed_dim,
            random_state=random_state,
        )

        # Text token embedding table
        self.text_embed: np.ndarray = (
            self.rng.randn(vocab_size, embed_dim).astype(np.float64) * 0.1
        )

    def predict_action(
        self,
        visual_tokens: np.ndarray,
        instruction_ids: np.ndarray,
        proprio_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict continuous action and discrete action tokens conditioned on vision, text and proprioception.

        Returns:
            continuous_action: (action_dim,) continuous command
            discrete_tokens: (action_dim,) discrete token IDs in [0, action_bins - 1]
        """
        # Average text instruction embeddings
        txt_ids = np.asarray(instruction_ids, dtype=np.int64)
        txt_emb = np.mean(self.text_embed[txt_ids], axis=0)

        vis_toks = np.asarray(visual_tokens, dtype=np.float64)
        vis_emb = np.mean(vis_toks, axis=0) if vis_toks.ndim > 1 else vis_toks

        # Fuse vision and text representations
        fused_vis = vis_emb + txt_emb

        # Predict smooth action step via ACT
        cont_act = self.act_model.step(fused_vis, proprio_state)
        disc_tokens = self.tokenizer.encode(cont_act)

        return cont_act, disc_tokens
