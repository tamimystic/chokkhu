"""Differentiable Fuzzy First-Order Logic Engine in pure NumPy."""

from __future__ import annotations


import numpy as np


class DifferentiableLogicEngine:
    """Continuous Fuzzy First-Order Logic Engine in pure NumPy.

    Relaxes Boolean logic operators to continuous t-norms and t-conorms, enabling
    differentiable logical reasoning and symbolic constraint satisfaction via gradient descent.

    Parameters
    ----------
    t_norm : str, default="product"
        Fuzzy logic t-norm system: "product", "godel", or "lukasiewicz".
    """

    def __init__(self, t_norm: str = "product") -> None:
        if t_norm not in ["product", "godel", "lukasiewicz"]:
            raise ValueError(
                f"Unknown t_norm: {t_norm}. Choose 'product', 'godel', or 'lukasiewicz'."
            )
        self.t_norm = t_norm

    def conjunction(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes logical AND (conjunction: a AND b)."""
        a_arr = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        b_arr = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)

        if self.t_norm == "product":
            return a_arr * b_arr
        elif self.t_norm == "godel":
            return np.minimum(a_arr, b_arr)
        else:  # lukasiewicz
            return np.maximum(0.0, a_arr + b_arr - 1.0)

    def disjunction(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes logical OR (disjunction: a OR b)."""
        a_arr = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        b_arr = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)

        if self.t_norm == "product":
            return a_arr + b_arr - a_arr * b_arr
        elif self.t_norm == "godel":
            return np.maximum(a_arr, b_arr)
        else:  # lukasiewicz
            return np.minimum(1.0, a_arr + b_arr)

    def negation(self, a: np.ndarray) -> np.ndarray:
        """Computes logical NOT (negation: NOT a)."""
        a_arr = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        return 1.0 - a_arr

    def implication(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes logical IMPLIES (implication: a => b)."""
        a_arr = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        b_arr = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)

        if self.t_norm == "product":
            # Reichenbach or Goguen implication
            return np.where(a_arr <= b_arr, 1.0, b_arr / (a_arr + 1e-12))
        elif self.t_norm == "godel":
            return np.where(a_arr <= b_arr, 1.0, b_arr)
        else:  # lukasiewicz
            return np.minimum(1.0, 1.0 - a_arr + b_arr)

    def universal_quantifier(self, values: np.ndarray, axis: int = -1) -> np.ndarray:
        """Evaluates universal quantifier (FORALL x phi(x))."""
        v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        return np.min(v, axis=axis)

    def existential_quantifier(self, values: np.ndarray, axis: int = -1) -> np.ndarray:
        """Evaluates existential quantifier (EXISTS x phi(x))."""
        v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        return np.max(v, axis=axis)

    def satisfaction_loss(self, formula_truth_value: np.ndarray) -> float:
        """Computes constraint satisfaction loss: L = 1.0 - truth_value."""
        v = np.clip(np.asarray(formula_truth_value, dtype=np.float32), 0.0, 1.0)
        return float(np.mean(1.0 - v))
