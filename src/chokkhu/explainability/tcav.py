from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import numpy as np


class TCAV:
    r"""Testing with Concept Activation Vectors (TCAV - Kim et al. 2018).

    Quantifies the degree to which a high-level human-understandable concept C
    influences a model's prediction for a specific class k:
        S_{C, k}(x) = \nabla h_l(x) \cdot v_C
        TCAV_{C, k} = \frac{|x \in X_k : S_{C, k}(x) > 0|}{|X_k|}
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.cav: Optional[np.ndarray] = None

    def compute_cav(
        self,
        concept_activations: np.ndarray,
        random_activations: np.ndarray,
    ) -> np.ndarray:
        """Trains a linear classifier to find the orthogonal Concept Activation Vector (CAV)."""
        pos = np.asarray(concept_activations, dtype=np.float64)
        neg = np.asarray(random_activations, dtype=np.float64)

        # Solve linear separation using regularized mean difference
        mean_pos = np.mean(pos, axis=0)
        mean_neg = np.mean(neg, axis=0)
        diff = mean_pos - mean_neg
        norm = np.linalg.norm(diff)

        if norm < 1e-12:
            self.cav = np.zeros_like(diff)
        else:
            self.cav = diff / norm

        return self.cav

    def directional_derivative(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        activations: np.ndarray,
        target_class: int = 0,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Calculates directional derivative of model prediction along the CAV direction."""
        if self.cav is None:
            raise ValueError("CAV must be computed first using compute_cav().")

        acts = np.asarray(activations, dtype=np.float64)
        if acts.ndim == 1:
            acts = acts[np.newaxis, :]

        derivatives: np.ndarray = np.zeros(len(acts), dtype=np.float64)
        for i, act in enumerate(acts):
            act_perturbed = act + eps * self.cav
            out_base = model_predict_fn(act.reshape(1, -1))
            out_pert = model_predict_fn(act_perturbed.reshape(1, -1))

            score_base = (
                out_base[0, target_class]
                if out_base.ndim > 1
                else (
                    out_base[target_class]
                    if len(out_base) > target_class
                    else out_base[0]
                )
            )
            score_pert = (
                out_pert[0, target_class]
                if out_pert.ndim > 1
                else (
                    out_pert[target_class]
                    if len(out_pert) > target_class
                    else out_pert[0]
                )
            )

            derivatives[i] = (score_pert - score_base) / eps

        return derivatives

    def compute_tcav_score(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        class_activations: np.ndarray,
        target_class: int = 0,
    ) -> Dict[str, Any]:
        """Computes the overall TCAV score (fraction of positive directional derivatives)."""
        derivatives = self.directional_derivative(
            model_predict_fn, class_activations, target_class=target_class
        )
        positive_count = int(np.sum(derivatives > 0))
        total_count = len(derivatives)
        score = float(positive_count / total_count) if total_count > 0 else 0.0

        return {
            "tcav_score": score,
            "positive_count": positive_count,
            "total_count": total_count,
            "mean_derivative": float(np.mean(derivatives)) if total_count > 0 else 0.0,
            "directional_derivatives": derivatives,
        }
