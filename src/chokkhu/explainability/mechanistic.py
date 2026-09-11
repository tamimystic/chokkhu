from __future__ import annotations

from typing import List, Optional, Sequence, Union
import numpy as np


class AttentionRollout:
    """Attention Rollout for Mechanistic Transformer Interpretability in pure NumPy.

    Tracks the flow of information through all multi-head attention layers
    by recursive matrix multiplication incorporating identity residual skip connections:
        A_hat_l = 0.5 * A_l + 0.5 * I
        Rollout = A_hat_L @ A_hat_{L-1} @ ... @ A_hat_1
    """

    def __init__(
        self, head_reduction: str = "mean", discard_ratio: float = 0.0
    ) -> None:
        self.head_reduction = head_reduction
        self.discard_ratio = float(discard_ratio)

    def compute(self, all_layer_attentions: Sequence[np.ndarray]) -> np.ndarray:
        """Computes cumulative attention rollout matrix across transformer layers.

        Args:
            all_layer_attentions: List of attention matrices per layer.
                Each array has shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
                or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            np.ndarray: Cumulative rollout attention matrix of shape (seq_len, seq_len)
                or (batch_size, seq_len, seq_len).
        """
        if not all_layer_attentions:
            raise ValueError("all_layer_attentions cannot be empty.")

        # Standardize layers to (batch, seq, seq)
        processed_layers: List[np.ndarray] = []
        for att in all_layer_attentions:
            arr = np.asarray(att, dtype=np.float64)
            if arr.ndim == 2:
                # (seq, seq) -> (1, seq, seq)
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                # (heads, seq, seq) -> reduce over heads -> (1, seq, seq)
                if self.head_reduction == "mean":
                    arr = np.mean(arr, axis=0, keepdims=True)
                elif self.head_reduction == "max":
                    arr = np.max(arr, axis=0, keepdims=True)
                elif self.head_reduction == "min":
                    arr = np.min(arr, axis=0, keepdims=True)
                else:
                    arr = np.mean(arr, axis=0, keepdims=True)
            elif arr.ndim == 4:
                # (batch, heads, seq, seq) -> reduce heads
                if self.head_reduction == "mean":
                    arr = np.mean(arr, axis=1)
                elif self.head_reduction == "max":
                    arr = np.max(arr, axis=1)
                elif self.head_reduction == "min":
                    arr = np.min(arr, axis=1)
                else:
                    arr = np.mean(arr, axis=1)
            else:
                raise ValueError(f"Invalid attention matrix dimension: {arr.ndim}")

            # Discard lowest attention weights if discard_ratio > 0
            if self.discard_ratio > 0.0:
                for b in range(arr.shape[0]):
                    flat = arr[b].flatten()
                    k = int(len(flat) * self.discard_ratio)
                    if k > 0:
                        threshold = np.partition(flat, k)[k]
                        arr[b][arr[b] < threshold] = 0.0
                        # Re-normalize rows
                        row_sums = np.sum(arr[b], axis=-1, keepdims=True) + 1e-12
                        arr[b] = arr[b] / row_sums

            processed_layers.append(arr)

        batch_size, seq_len, _ = processed_layers[0].shape
        eye: np.ndarray = np.eye(seq_len, dtype=np.float64)

        # Initialize rollout with first layer + residual
        rollout = 0.5 * processed_layers[0] + 0.5 * eye

        for layer_att in processed_layers[1:]:
            a_hat = 0.5 * layer_att + 0.5 * eye
            rollout = np.matmul(a_hat, rollout)

        # Normalize rows
        row_sums = np.sum(rollout, axis=-1, keepdims=True) + 1e-12
        rollout = rollout / row_sums

        if rollout.shape[0] == 1:
            return rollout[0]
        return rollout


class DirectLogitAttribution:
    """Direct Logit Attribution (DLA) for mechanistic residual stream decomposition.

    Projects intermediate hidden layer representations directly to the vocabulary
    space using the unembedding matrix W_U:
        logit_attribution_l = h_l @ W_U
    """

    def __init__(self, unembedding_matrix: np.ndarray) -> None:
        self.W_U = np.asarray(
            unembedding_matrix, dtype=np.float64
        )  # (hidden_dim, vocab_size)

    def attribute(
        self,
        hidden_states: Union[np.ndarray, Sequence[np.ndarray]],
        target_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """Attributes output logits to specific hidden layers or components.

        Args:
            hidden_states: Array of shape (num_layers, hidden_dim) or (hidden_dim,)
            target_token_id: Optional specific vocabulary token index.

        Returns:
            np.ndarray: Logit attribution scores per layer / component.
        """
        h = np.asarray(hidden_states, dtype=np.float64)
        if h.ndim == 1:
            # (hidden_dim,) -> logits: (vocab_size,)
            logits = np.dot(h, self.W_U)
            if target_token_id is not None:
                return float(logits[target_token_id])  # type: ignore[return-value]
            return logits

        # (num_layers, hidden_dim) @ (hidden_dim, vocab_size) -> (num_layers, vocab_size)
        logits = np.matmul(h, self.W_U)
        if target_token_id is not None:
            return logits[:, target_token_id]
        return logits
