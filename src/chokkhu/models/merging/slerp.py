"""SLERP (Spherical Linear Interpolation) for Neural Model Weights.

Reference:
    Shoemake, "Animating rotation with quaternion curves", ACM SIGGRAPH 1985.
"""

from typing import Dict, Union, Any, Optional
import numpy as np


class SLERP:
    """Spherical Linear Interpolation (SLERP) for geometric neural network weight merging."""

    def __init__(self, t: float = 0.5, eps: float = 1e-8, collinear_threshold: float = 0.9995) -> None:
        """Initialize SLERP.

        Args:
            t: Interpolation factor between weights_a and weights_b (0.0 to 1.0).
            eps: Numerical stability constant.
            collinear_threshold: Cosine similarity threshold above which standard LERP is used.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t must be in [0.0, 1.0], got {t}")
        self.t = float(t)
        self.eps = float(eps)
        self.collinear_threshold = float(collinear_threshold)

    def interpolate_tensors(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """Interpolate two matching parameter tensors along the geodesic sphere."""
        interp_t = self.t if t is None else float(t)

        if interp_t <= 0.0:
            return np.copy(tensor_a)
        if interp_t >= 1.0:
            return np.copy(tensor_b)

        shape = tensor_a.shape
        v0 = tensor_a.astype(np.float64).flatten()
        v1 = tensor_b.astype(np.float64).flatten()

        norm0 = np.linalg.norm(v0)
        norm1 = np.linalg.norm(v1)

        # Fallback to linear interpolation if norm is zero
        if norm0 < self.eps or norm1 < self.eps:
            res = (1.0 - interp_t) * v0 + interp_t * v1
            return res.reshape(shape).astype(tensor_a.dtype)

        # Normalize unit direction vectors
        u0 = v0 / norm0
        u1 = v1 / norm1

        # Cosine of angle between vectors
        dot = np.clip(np.dot(u0, u1), -1.0, 1.0)

        # If vectors are almost collinear, use regular LERP to avoid division by zero
        if np.abs(dot) > self.collinear_threshold:
            res = (1.0 - interp_t) * v0 + interp_t * v1
            return res.reshape(shape).astype(tensor_a.dtype)

        # SLERP formula
        omega = np.arccos(dot)
        sin_omega = np.sin(omega)

        scale0 = np.sin((1.0 - interp_t) * omega) / sin_omega
        scale1 = np.sin(interp_t * omega) / sin_omega

        # Interpolated direction and magnitude
        interpolated_dir = scale0 * u0 + scale1 * u1
        interpolated_norm = (1.0 - interp_t) * norm0 + interp_t * norm1
        res = interpolated_norm * interpolated_dir

        return res.reshape(shape).astype(tensor_a.dtype)

    def merge_weight_dicts(
        self,
        weights_a: Dict[str, np.ndarray],
        weights_b: Dict[str, np.ndarray],
        t: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Merge two weight dictionaries using SLERP."""
        merged: Dict[str, np.ndarray] = {}
        for key in weights_a:
            if key in weights_b:
                merged[key] = self.interpolate_tensors(weights_a[key], weights_b[key], t=t)
            else:
                merged[key] = np.copy(weights_a[key])
        return merged
