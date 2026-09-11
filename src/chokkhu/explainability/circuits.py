"""Mechanistic Interpretability, Causal Activation Patching, and Integrated Gradients.

Formulated from first principles using interventional causal tracing across transformer
layers/heads and axiomatic path integral attributions (Integrated Gradients) in pure NumPy.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class ActivationPatchingEngine:
    """Interventional Causal Tracing and Activation Patching Engine.

    Localizes factual associations and mechanistic circuits by intervening on
    hidden representations between clean and corrupted model executions.

    Parameters
    ----------
    forward_with_cache : Callable[[np.ndarray, Optional[Dict[str, np.ndarray]]], Tuple[np.ndarray, Dict[str, np.ndarray]]]
        Function that accepts (input_tensor, optional_patch_dict) and returns
        (output_logits, activations_cache_dict).
    """

    def __init__(
        self,
        forward_with_cache: Callable[
            [np.ndarray, Optional[Dict[str, np.ndarray]]],
            Tuple[np.ndarray, Dict[str, np.ndarray]],
        ],
    ) -> None:
        self.forward_with_cache = forward_with_cache

    def compute_causal_effect(
        self,
        clean_input: np.ndarray,
        corrupted_input: np.ndarray,
        layer_name: str,
        token_pos: Optional[int] = None,
        metric_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> float:
        """Measure the causal recovery effect of restoring clean activation at layer_name.

        Parameters
        ----------
        clean_input : np.ndarray
            Input prompt that generates the correct target behavior.
        corrupted_input : np.ndarray
            Perturbed/corrupted prompt where behavior is degraded.
        layer_name : str
            Identifier of activation hook to intervene upon.
        token_pos : Optional[int], default=None
            Specific token index to patch. If None, patches across all tokens.
        metric_fn : Optional[Callable[[np.ndarray], float]], default=None
            Scalar evaluation metric on output logits (e.g. logit difference).
            Defaults to target logit difference.

        Returns
        -------
        recovery_ratio : float
            Normalized causal mediation effect in [0, 1] (or unbounded percentage).
        """
        if metric_fn is None:

            def _default_metric(logits: np.ndarray) -> float:
                return float(np.sum(logits))

            metric_fn = _default_metric

        # 1. Clean and corrupted baseline runs
        clean_logits, clean_cache = self.forward_with_cache(clean_input, None)
        corrupt_logits, _ = self.forward_with_cache(corrupted_input, None)

        clean_score = metric_fn(clean_logits)
        corrupt_score = metric_fn(corrupt_logits)

        if abs(clean_score - corrupt_score) < 1e-12:
            return 0.0

        # 2. Interventional run: corrupted input + clean patch at layer_name
        clean_act = clean_cache[layer_name].copy()
        if token_pos is not None:
            # Create partial patch dictionary
            patch_dict = {
                f"{layer_name}_pos_{token_pos}": clean_act[:, token_pos : token_pos + 1]
            }
        else:
            patch_dict = {layer_name: clean_act}

        patched_logits, _ = self.forward_with_cache(corrupted_input, patch_dict)
        patched_score = metric_fn(patched_logits)

        # 3. Normalized recovery metric: (patched - corrupted) / (clean - corrupted)
        recovery = (patched_score - corrupt_score) / (
            clean_score - corrupt_score + 1e-12
        )
        return float(recovery)

    def trace_circuit(
        self,
        clean_input: np.ndarray,
        corrupted_input: np.ndarray,
        layer_names: List[str],
        metric_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Dict[str, float]:
        """Perform full causal sweep across all specified network layers.

        Returns
        -------
        attribution_map : Dict[str, float]
            Mapping from layer names to causal mediation effect scores.
        """
        attribution: Dict[str, float] = {}
        for layer in layer_names:
            effect = self.compute_causal_effect(
                clean_input, corrupted_input, layer, metric_fn=metric_fn
            )
            attribution[layer] = effect
        return attribution


class IntegratedGradients:
    r"""Axiomatic Path-Integral Attribution (Integrated Gradients).

        Computes feature importance satisfying Completeness and Implementation Invariance:
        IG_i(x) = (x_i - x'_i) * \int_0^1
    rac{\partial F(x' + lpha(x - x'))}{\partial x_i} dlpha

        Parameters
        ----------
        model_fn : Callable[[np.ndarray], np.ndarray]
            Forward pass function mapping input features to scalar or 1D class output.
        grad_fn : Optional[Callable[[np.ndarray], np.ndarray]], default=None
            Analytical gradient function. If None, central finite differences are used.
        steps : int, default=50
            Number of quadrature integration steps along the straight-line path.
        method : str, default="gauss_legendre"
            Numerical quadrature scheme: "gauss_legendre" or "riemann".
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        steps: int = 50,
        method: str = "gauss_legendre",
    ) -> None:
        self.model_fn = model_fn
        self.grad_fn = grad_fn
        self.steps = max(5, int(steps))
        self.method = method

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient via central differences."""
        grad = np.zeros_like(x, dtype=float)
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig_val = x[idx]

            x[idx] = orig_val + eps
            f_plus: float = float(np.sum(self.model_fn(x)))

            x[idx] = orig_val - eps
            f_minus: float = float(np.sum(self.model_fn(x)))

            x[idx] = orig_val
            grad[idx] = (f_plus - f_minus) / (2.0 * eps)
            it.iternext()
        return grad

    def attribute(
        self,
        inputs: np.ndarray,
        baselines: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute integrated gradient attributions for input features.

        Parameters
        ----------
        inputs : np.ndarray
            Input feature tensor.
        baselines : Optional[np.ndarray], default=None
            Reference baseline tensor x' (e.g. zeros). Defaults to zeros_like(inputs).

        Returns
        -------
        attributions : np.ndarray
            Feature attributions matching inputs.shape.
        """
        x = np.asarray(inputs, dtype=float)
        x_prime = (
            np.zeros_like(x)
            if baselines is None
            else np.asarray(baselines, dtype=float)
        )

        delta = x - x_prime

        if self.method == "gauss_legendre":
            # Gauss-Legendre quadrature nodes and weights on [0, 1]
            roots, weights = np.polynomial.legendre.leggauss(self.steps)
            alphas = 0.5 * (roots + 1.0)
            quad_weights = 0.5 * weights
        else:  # Riemann trapezoidal
            alphas = np.linspace(0.0, 1.0, self.steps)
            quad_weights = np.full(self.steps, 1.0 / self.steps)

        # Accumulate gradients along path
        accum_grads = np.zeros_like(x)
        for alpha, weight in zip(alphas, quad_weights):
            path_point = x_prime + alpha * delta
            if self.grad_fn is not None:
                g = self.grad_fn(path_point)
            else:
                g = self._numerical_gradient(path_point)
            accum_grads += weight * g

        attributions = delta * accum_grads
        return attributions
