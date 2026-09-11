"""Refusal Direction Probing, Activation Steering, and Alignment Safety.

Formulated from first principles for linear probe estimation, orthogonal subspace
ablation, and directional activation steering in pure NumPy.
"""

import numpy as np
from typing import Optional


class RefusalDirectionProbe:
    """Linear Refusal Direction Probe & Representation Steering Engine.

    Parameters
    ----------
    dim : int, default=64
        Hidden activation representation dimension.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        self.direction: Optional[np.ndarray] = None
        self.mu_harmful: Optional[np.ndarray] = None
        self.mu_harmless: Optional[np.ndarray] = None

    def fit(
        self, harmful_activations: np.ndarray, harmless_activations: np.ndarray
    ) -> "RefusalDirectionProbe":
        """Compute the unit refusal vector from harmful vs harmless activation distributions.

        Parameters
        ----------
        harmful_activations : np.ndarray of shape (N_harmful, dim)
            Residual stream activations on safety-violating prompts.
        harmless_activations : np.ndarray of shape (N_harmless, dim)
            Residual stream activations on compliant prompts.

        Returns
        -------
        self : RefusalDirectionProbe
        """
        harmful_act = np.asarray(harmful_activations, dtype=float)
        harmless_act = np.asarray(harmless_activations, dtype=float)

        if harmful_act.ndim == 1:
            harmful_act = harmful_act.reshape(1, -1)
        if harmless_act.ndim == 1:
            harmless_act = harmless_act.reshape(1, -1)

        self.mu_harmful = np.mean(harmful_act, axis=0)
        self.mu_harmless = np.mean(harmless_act, axis=0)

        delta = self.mu_harmful - self.mu_harmless
        norm = np.linalg.norm(delta)
        if norm < 1e-12:
            self.direction = np.zeros(self.dim)
        else:
            self.direction = delta / norm

        return self

    def steer(self, activations: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Inject refusal direction vector to strengthen safety guardrails.

        Parameters
        ----------
        activations : np.ndarray of shape (..., dim)
            Input hidden representations.
        alpha : float, default=1.0
            Steering intensity coefficient.

        Returns
        -------
        steered_act : np.ndarray of shape (..., dim)
        """
        if self.direction is None:
            raise ValueError("Probe has not been fitted. Call fit() first.")

        return activations + alpha * self.direction

    def ablate(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto the orthogonal subspace to remove refusal behavior.

        Parameters
        ----------
        activations : np.ndarray of shape (..., dim)
            Input hidden representations.

        Returns
        -------
        ablated_act : np.ndarray of shape (..., dim)
        """
        if self.direction is None:
            raise ValueError("Probe has not been fitted. Call fit() first.")

        # h_orth = h - (h . v) * v
        dot = np.sum(activations * self.direction, axis=-1, keepdims=True)
        return activations - dot * self.direction

    def score_refusal_intent(self, activations: np.ndarray) -> np.ndarray:
        """Compute scalar projection / cosine alignment along refusal axis.

        Returns
        -------
        scores : np.ndarray of shape (...)
        """
        if self.direction is None:
            raise ValueError("Probe has not been fitted. Call fit() first.")

        return np.sum(activations * self.direction, axis=-1)
