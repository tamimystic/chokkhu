"""Model Pruning and Sparsification Engine.

Pure NumPy implementations of:
- MagnitudePruner: Unstructured and structured magnitude-based weight pruning
- GlobalMagnitudePruner: Global layer-wise sparsity thresholding
"""

from typing import Dict, Tuple
import numpy as np


class MagnitudePruner:
    r"""Magnitude-Based Weight Pruning.

    Zeroes out weights with lowest absolute magnitude up to sparsity level :math:`\kappa \in [0.0, 1.0)`.

    Parameters
    ----------
    amount : float, default=0.5
        Fraction of weights/channels to prune (sparsity ratio).
    structured : bool, default=False
        Whether to perform structured channel/filter pruning (pruning entire output neurons/channels).
    dim : int, default=0
        Dimension along which structured pruning evaluates L1/L2 norm.
    """

    def __init__(
        self,
        amount: float = 0.5,
        structured: bool = False,
        dim: int = 0,
    ) -> None:
        if not (0.0 <= amount < 1.0):
            raise ValueError("amount must be in [0.0, 1.0).")
        self.amount = float(amount)
        self.structured = structured
        self.dim = dim

    def prune(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pruning mask and return (pruned_weights, binary_mask)."""
        w = np.asarray(weights, dtype=np.float32).copy()
        if self.amount == 0.0:
            mask = np.ones_like(w, dtype=np.float32)
            return w, mask

        if not self.structured:
            # Unstructured pruning: zero out smallest absolute values
            abs_w = np.abs(w)
            threshold = float(np.percentile(abs_w, self.amount * 100.0))
            mask = (abs_w >= threshold).astype(np.float32)
            pruned_w = w * mask
            return pruned_w, mask
        else:
            # Structured channel/neuron pruning based on L1 norm along specified axis
            axes = tuple(i for i in range(w.ndim) if i != self.dim)
            norms = np.sum(np.abs(w), axis=axes)
            threshold = float(np.percentile(norms, self.amount * 100.0))
            keep_indices = norms >= threshold

            # Construct structured broadcasting mask
            mask_shape = [1] * w.ndim
            mask_shape[self.dim] = w.shape[self.dim]
            channel_mask = keep_indices.reshape(mask_shape).astype(np.float32)

            mask = np.broadcast_to(channel_mask, w.shape).copy()
            pruned_w = w * mask
            return pruned_w, mask

    def compute_sparsity(self, weights: np.ndarray) -> float:
        """Compute fraction of zero weights."""
        w = np.asarray(weights)
        if w.size == 0:
            return 0.0
        return float(np.sum(w == 0.0)) / float(w.size)


class GlobalMagnitudePruner:
    """Global Multi-Layer Magnitude Pruner.

    Computes a single global magnitude threshold across all model weight tensors.
    """

    def __init__(self, amount: float = 0.5) -> None:
        if not (0.0 <= amount < 1.0):
            raise ValueError("amount must be in [0.0, 1.0).")
        self.amount = float(amount)

    def prune_dict(
        self, weights_dict: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prune dictionary of weights and return (pruned_dict, masks_dict)."""
        if not weights_dict:
            return {}, {}

        # Collect all weights into a single 1D array
        all_weights: list[np.ndarray] = [
            np.abs(w.ravel()) for w in weights_dict.values()
        ]
        concatenated = np.concatenate(all_weights)

        global_thresh = float(np.percentile(concatenated, self.amount * 100.0))

        pruned_dict: Dict[str, np.ndarray] = {}
        masks_dict: Dict[str, np.ndarray] = {}

        for k, v in weights_dict.items():
            mask = (np.abs(v) >= global_thresh).astype(np.float32)
            pruned_dict[k] = v * mask
            masks_dict[k] = mask

        return pruned_dict, masks_dict
