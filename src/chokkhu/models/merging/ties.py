"""TIES-Merging (Trimming, Electing Sign, and Disjoint Merge).

Reference:
    Yadav et al., "Resolving Interference When Merging Models", NeurIPS 2023.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class TIESMerging:
    """TIES-Merging algorithm for resolving parameter interference across multiple models.

    Steps:
        1. Task Vector Computation: delta_m = theta_m - theta_base
        2. Trimming: Keep top k% highest magnitude coordinates per tensor.
        3. Electing Sign: Compute dominant direction gamma = sign(sum(delta_m)).
        4. Disjoint Merge: Average only those task vectors that agree with elected sign.
        5. Scaling & Addition: theta_merged = theta_base + lambda * delta_merged
    """

    def __init__(self, density: float = 0.2, scaling_factor: float = 1.0) -> None:
        """Initialize TIES-Merging.

        Args:
            density: Fraction of parameters to retain during trimming (0.0 < density <= 1.0).
            scaling_factor: Scaling multiplier lambda applied to the merged task vector.
        """
        if not (0.0 < density <= 1.0):
            raise ValueError(f"density must be in (0.0, 1.0], got {density}")
        self.density = float(density)
        self.scaling_factor = float(scaling_factor)

    def trim_tensor(
        self, tensor: np.ndarray, density: Optional[float] = None
    ) -> np.ndarray:
        """Trim tensor keeping only the top k fraction by absolute magnitude."""
        d = self.density if density is None else density
        if d >= 1.0 or tensor.size == 0:
            return np.copy(tensor)

        abs_vals = np.abs(tensor).flatten()
        k = int(np.ceil(d * tensor.size))
        if k >= tensor.size:
            return np.copy(tensor)
        if k <= 0:
            return np.zeros_like(tensor)

        threshold = np.partition(abs_vals, -k)[-k]
        mask = np.abs(tensor) >= threshold
        return np.where(mask, tensor, 0.0)

    def merge_tensors(
        self,
        base_tensor: np.ndarray,
        task_tensors: List[np.ndarray],
        density: Optional[float] = None,
        scaling_factor: Optional[float] = None,
    ) -> np.ndarray:
        """Merge a list of task parameter tensors with a base parameter tensor using TIES."""
        if not task_tensors:
            return np.copy(base_tensor)

        d = self.density if density is None else density
        lam = self.scaling_factor if scaling_factor is None else scaling_factor

        trimmed_deltas = []
        for task_t in task_tensors:
            delta = task_t.astype(np.float64) - base_tensor.astype(np.float64)
            trimmed_delta = self.trim_tensor(delta, density=d)
            trimmed_deltas.append(trimmed_delta)

        stacked_deltas = np.stack(trimmed_deltas, axis=0)  # [M, ...]

        # Elect Sign
        sum_deltas = np.sum(stacked_deltas, axis=0)
        majority_sign = np.sign(sum_deltas)

        # Disjoint Merge
        signs = np.sign(stacked_deltas)
        agree_mask = (signs == majority_sign[np.newaxis, ...]) & (
            majority_sign[np.newaxis, ...] != 0
        )

        filtered_deltas = np.where(agree_mask, stacked_deltas, 0.0)
        sum_agreeing = np.sum(filtered_deltas, axis=0)
        count_agreeing = np.sum(agree_mask, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            disjoint_merged = np.where(
                count_agreeing > 0, sum_agreeing / count_agreeing, 0.0
            )

        merged_tensor = base_tensor.astype(np.float64) + lam * disjoint_merged
        return merged_tensor.astype(base_tensor.dtype)

    def merge_weight_dicts(
        self,
        base_weights: Dict[str, np.ndarray],
        task_weights_list: List[Dict[str, np.ndarray]],
        density: Optional[float] = None,
        scaling_factor: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Merge dictionaries of parameter tensors."""
        merged_dict: Dict[str, np.ndarray] = {}
        for key, base_t in base_weights.items():
            task_t_list = [tw[key] for tw in task_weights_list if key in tw]
            if not task_t_list:
                merged_dict[key] = np.copy(base_t)
            else:
                merged_dict[key] = self.merge_tensors(
                    base_t, task_t_list, density=density, scaling_factor=scaling_factor
                )
        return merged_dict

    def merge_models(
        self,
        base_model: Any,
        task_models: List[Any],
        density: Optional[float] = None,
        scaling_factor: Optional[float] = None,
    ) -> Any:
        """Merge model objects supporting state_dict() or parameters()."""
        if hasattr(base_model, "state_dict") and callable(base_model.state_dict):
            base_dict = base_model.state_dict()
            task_dicts = [m.state_dict() for m in task_models]
            merged_dict = self.merge_weight_dicts(
                base_dict, task_dicts, density=density, scaling_factor=scaling_factor
            )
            base_model.load_state_dict(merged_dict)
            return base_model
        elif hasattr(base_model, "parameters") and callable(base_model.parameters):
            base_params = list(base_model.parameters())
            for idx, bp in enumerate(base_params):
                task_p = [list(m.parameters())[idx] for m in task_models]
                if hasattr(bp, "data"):
                    merged_arr = self.merge_tensors(
                        bp.data,
                        [p.data for p in task_p],
                        density=density,
                        scaling_factor=scaling_factor,
                    )
                    bp.data = merged_arr
                elif isinstance(bp, np.ndarray):
                    bp[:] = self.merge_tensors(
                        bp, task_p, density=density, scaling_factor=scaling_factor
                    )
            return base_model
        else:
            raise TypeError("Models must have state_dict() or parameters() method")
