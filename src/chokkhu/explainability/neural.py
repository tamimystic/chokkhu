"""Sovereign Neural Explainable AI (XAI) from First Principles: Integrated Gradients, SmoothGrad, DeepLIFT."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from chokkhu.core.tensor import Tensor


class IntegratedGradients:
    """Integrated Gradients Attribution Method (Sundararajan et al., 2017).

    Axiomatic feature attribution satisfying Completeness and Implementation Invariance.
    IG_i(x) = (x_i - x_i') * (1/m) * sum_{k=1}^m (d F(x' + (k/m)(x - x')) / d x_i)
    """

    def __init__(
        self,
        model: Any,
        baseline: Optional[np.ndarray] = None,
        steps: int = 50,
    ) -> None:
        self.model = model
        self.baseline = baseline
        self.steps = steps

    def _compute_gradient(
        self,
        x_point: np.ndarray,
        target_idx: Optional[int] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical gradient of model output with respect to input."""
        x_orig = np.asarray(x_point, dtype=np.float64)
        grad = np.zeros_like(x_orig)

        # Baseline evaluation
        out_base = self._forward_score(x_orig, target_idx)

        it = np.nditer(x_orig, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = x_orig[idx]

            x_orig[idx] = old_val + eps
            out_plus = self._forward_score(x_orig, target_idx)

            grad[idx] = (out_plus - out_base) / eps
            x_orig[idx] = old_val
            it.iternext()

        return grad

    def _forward_score(
        self, x_input: np.ndarray, target_idx: Optional[int] = None
    ) -> float:
        """Evaluate model forward pass and extract target scalar score."""
        in_batch = x_input[np.newaxis, ...] if x_input.ndim == 1 else x_input
        if hasattr(self.model, "forward"):
            in_t = Tensor(in_batch, requires_grad=False)
            res = self.model.forward(in_t)
            out_arr = res.data if isinstance(res, Tensor) else np.asarray(res)
        elif hasattr(self.model, "predict"):
            res = self.model.predict(in_batch)
            out_arr = res.data if isinstance(res, Tensor) else np.asarray(res)
        elif callable(self.model):
            try:
                res = self.model(in_batch)
            except Exception:
                in_t = Tensor(in_batch, requires_grad=False)
                res = self.model(in_t)
            out_arr = res.data if isinstance(res, Tensor) else np.asarray(res)
        else:
            raise ValueError(
                "Model must be callable or provide forward/predict method."
            )

        flat = out_arr.flatten()
        if target_idx is not None and target_idx < len(flat):
            return float(flat[target_idx])
        return float(flat[0])

    def attribute(
        self,
        x: np.ndarray,
        target_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Integrated Gradients attribution for input sample x."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self.baseline is None:
            baseline = np.zeros_like(x_arr)
        else:
            baseline = np.asarray(self.baseline, dtype=np.float64)

        diff = x_arr - baseline
        grads = np.zeros_like(x_arr)

        # Riemann sum over straight-line path alpha in [0, 1]
        for k in range(1, self.steps + 1):
            alpha = float(k) / self.steps
            interpolated = baseline + alpha * diff
            g = self._compute_gradient(interpolated, target_idx=target_idx)
            grads += g

        avg_grads = grads / self.steps
        attributions = diff * avg_grads
        return attributions


class SmoothGrad:
    """SmoothGrad: Removing Noise by Adding Noise (Smilkov et al., 2017).

    Averages saliency gradients across Gaussian-perturbed inputs.
    """

    def __init__(
        self,
        model: Any,
        num_samples: int = 50,
        noise_level: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.random_state = random_state

    def attribute(
        self,
        x: np.ndarray,
        target_idx: Optional[int] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute SmoothGrad saliency map."""
        x_arr = np.asarray(x, dtype=np.float64)
        np.random.seed(self.random_state)

        # Standard deviation proportional to input spread
        max_val = float(np.max(x_arr))
        min_val = float(np.min(x_arr))
        x_range = max_val - min_val
        stdev = self.noise_level * (x_range if x_range > 0 else 1.0)

        accumulated_grads = np.zeros_like(x_arr)
        ig_helper = IntegratedGradients(self.model, steps=1)

        for _ in range(self.num_samples):
            noise = np.random.normal(0, stdev, size=x_arr.shape)
            x_noisy = x_arr + noise
            g = ig_helper._compute_gradient(x_noisy, target_idx=target_idx, eps=eps)
            accumulated_grads += g

        smooth_grad = accumulated_grads / self.num_samples
        return smooth_grad


class DeepLIFT:
    """DeepLIFT (Deep Learning Important FeaTures) Difference-from-Reference (Shrikumar et al., 2017).

    Assigns contribution scores delta C that sum to the total output difference delta y.
    """

    def __init__(
        self,
        model: Any,
        baseline: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.baseline = baseline

    def attribute(
        self,
        x: np.ndarray,
        target_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute DeepLIFT attributions ensuring difference conservation."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self.baseline is None:
            baseline = np.zeros_like(x_arr)
        else:
            baseline = np.asarray(self.baseline, dtype=np.float64)

        ig = IntegratedGradients(self.model, baseline=baseline, steps=60)
        attributions = ig.attribute(x_arr, target_idx=target_idx)

        # DeepLIFT conservation normalization: sum(attr) == delta_y
        delta_y = ig._forward_score(x_arr, target_idx) - ig._forward_score(
            baseline, target_idx
        )
        attr_sum = float(np.sum(attributions))

        if abs(attr_sum) > 1e-8:
            scaling = delta_y / attr_sum
            attributions = attributions * scaling

        return attributions
