"""DARE (Drop And REscale) Model Merging.

Reference:
    Yu et al., "Language Models are Super Mario: Absorbing Abilities from
    Homologous Models as a Free Lunch", ICML 2024.
"""

from typing import Dict, List, Optional
import numpy as np
from .ties import TIESMerging


class DARE:
    """DARE (Drop And REscale) algorithm for stochastic parameter sparsification and merging.

    Drop: Zero out delta parameters with drop probability p (Bernoulli mask).
    Rescale: Multiply retained deltas by 1 / (1 - p) to ensure unbiased expected updates.
    """

    def __init__(
        self,
        drop_rate: float = 0.7,
        scaling_factor: float = 1.0,
        mode: str = "ties",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize DARE.

        Args:
            drop_rate: Probability of dropping task parameter deltas p in [0.0, 1.0).
            scaling_factor: Final scaling multiplier lambda.
            mode: Merging mode after DARE sparsification ('linear' or 'ties').
            seed: Random seed for deterministic Bernoulli masking.
        """
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0.0, 1.0), got {drop_rate}")
        if mode not in ("linear", "ties"):
            raise ValueError(f"mode must be 'linear' or 'ties', got {mode}")

        self.drop_rate = float(drop_rate)
        self.scaling_factor = float(scaling_factor)
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sparsify_task_vector(
        self,
        delta: np.ndarray,
        drop_rate: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Drop And REscale to a single task delta vector."""
        p = self.drop_rate if drop_rate is None else drop_rate
        if p <= 0.0 or delta.size == 0:
            return np.copy(delta)

        # Bernoulli mask with keep probability (1 - p)
        keep_prob = 1.0 - p
        mask = self.rng.random(size=delta.shape) < keep_prob
        rescale_factor = 1.0 / keep_prob
        return np.where(mask, delta * rescale_factor, 0.0)

    def merge_tensors(
        self,
        base_tensor: np.ndarray,
        task_tensors: List[np.ndarray],
        drop_rate: Optional[float] = None,
        scaling_factor: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Merge task tensors into base tensor using DARE."""
        if not task_tensors:
            return np.copy(base_tensor)

        p = self.drop_rate if drop_rate is None else drop_rate
        lam = self.scaling_factor if scaling_factor is None else scaling_factor
        m = self.mode if mode is None else mode

        # 1. Compute and sparsify task vectors
        dare_task_tensors = []
        for task_t in task_tensors:
            delta = task_t.astype(np.float64) - base_tensor.astype(np.float64)
            dare_delta = self.sparsify_task_vector(delta, drop_rate=p)
            dare_task_tensors.append(base_tensor.astype(np.float64) + dare_delta)

        if m == "linear":
            # Direct average of DARE task vectors
            stacked = np.stack(
                [tt - base_tensor.astype(np.float64) for tt in dare_task_tensors],
                axis=0,
            )
            mean_delta = np.mean(stacked, axis=0)
            merged = base_tensor.astype(np.float64) + lam * mean_delta
            return merged.astype(base_tensor.dtype)
        else:
            # DARE + TIES consensus merge
            ties = TIESMerging(density=1.0, scaling_factor=lam)
            return ties.merge_tensors(base_tensor, dare_task_tensors)

    def merge_weight_dicts(
        self,
        base_weights: Dict[str, np.ndarray],
        task_weights_list: List[Dict[str, np.ndarray]],
        drop_rate: Optional[float] = None,
        scaling_factor: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Merge dictionaries of weights using DARE."""
        merged_dict: Dict[str, np.ndarray] = {}
        for key, base_t in base_weights.items():
            task_t_list = [tw[key] for tw in task_weights_list if key in tw]
            if not task_t_list:
                merged_dict[key] = np.copy(base_t)
            else:
                merged_dict[key] = self.merge_tensors(
                    base_t,
                    task_t_list,
                    drop_rate=drop_rate,
                    scaling_factor=scaling_factor,
                    mode=mode,
                )
        return merged_dict
