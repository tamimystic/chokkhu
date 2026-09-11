from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np


class WachterCounterfactualExplainer:
    """Actionable Counterfactual Explanation Optimizer (Wachter et al. 2017).

    Finds the minimal, sparse input perturbation x' closest to query sample x
    such that the model's prediction changes to the desired target outcome y_target:
        min_{x'} (f(x') - y_target)^2 + lambda_1 * ||x' - x||_1 + lambda_2 * ||x' - x||_2^2
    """

    def __init__(
        self,
        model: Any = None,
        predict_fn: Optional[Callable[[np.ndarray], Any]] = None,
        lambda_1: float = 0.01,
        lambda_2: float = 0.05,
        lr: float = 0.05,
        max_iter: int = 300,
        tolerance: float = 0.05,
    ) -> None:
        self.model = model
        self.predict_fn = predict_fn or (
            lambda x: (
                model.predict_proba(x)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(x)
            )
        )
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def _forward_score(self, x: np.ndarray) -> float:
        """Evaluates model prediction for single sample."""
        in_batch = x.reshape(1, -1)
        res = self.predict_fn(in_batch)
        if isinstance(res, np.ndarray):
            return float(res.flatten()[0])
        return float(res)

    def _numerical_gradient(
        self,
        x_curr: np.ndarray,
        x_orig: np.ndarray,
        y_target: float,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Computes numerical gradient of Wachter counterfactual loss with respect to x_curr."""
        pred_curr = self._forward_score(x_curr)
        loss_pred_curr = (pred_curr - y_target) ** 2

        diff = x_curr - x_orig
        l1_reg = self.lambda_1 * np.sum(np.abs(diff))
        l2_reg = self.lambda_2 * np.sum(diff**2)
        base_loss = loss_pred_curr + l1_reg + l2_reg

        grad = np.zeros_like(x_curr)
        for i in range(len(x_curr)):
            x_pert = x_curr.copy()
            x_pert[i] += eps
            pred_pert = self._forward_score(x_pert)
            diff_pert = x_pert - x_orig

            loss_pert = (
                (pred_pert - y_target) ** 2
                + self.lambda_1 * np.sum(np.abs(diff_pert))
                + self.lambda_2 * np.sum(diff_pert**2)
            )
            grad[i] = (loss_pert - base_loss) / eps

        return grad

    def explain(
        self,
        x: np.ndarray,
        target_prediction: float = 1.0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Finds counterfactual explanation sample x_cf for input sample x using Wachter's adaptive schedule."""
        x_orig = np.asarray(x, dtype=np.float64).flatten()
        x_cf = x_orig.copy()
        y_target = float(target_prediction)

        curr_l1 = self.lambda_1
        curr_l2 = self.lambda_2

        # Outer loop: dynamically adapt lambda if prediction hasn't reached target tolerance
        for outer_iter in range(5):
            # Adam optimizer state
            m = np.zeros_like(x_cf)
            v = np.zeros_like(x_cf)
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

            inner_steps = max(50, self.max_iter // 3)

            for step in range(1, inner_steps + 1):
                pred_curr = self._forward_score(x_cf)
                if abs(pred_curr - y_target) < self.tolerance:
                    break

                # Numerical gradient with current lambda
                loss_pred_curr = (pred_curr - y_target) ** 2
                diff = x_cf - x_orig
                base_loss = (
                    loss_pred_curr
                    + curr_l1 * np.sum(np.abs(diff))
                    + curr_l2 * np.sum(diff**2)
                )

                grad = np.zeros_like(x_cf)
                eps = 1e-4
                for i in range(len(x_cf)):
                    x_pert = x_cf.copy()
                    x_pert[i] += eps
                    pred_pert = self._forward_score(x_pert)
                    diff_pert = x_pert - x_orig
                    loss_pert = (
                        (pred_pert - y_target) ** 2
                        + curr_l1 * np.sum(np.abs(diff_pert))
                        + curr_l2 * np.sum(diff_pert**2)
                    )
                    grad[i] = (loss_pert - base_loss) / eps

                # Adam update
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad**2)
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)

                x_cf -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

                if bounds is not None:
                    lower, upper = bounds
                    x_cf = np.clip(x_cf, lower, upper)

            final_pred = self._forward_score(x_cf)
            if abs(final_pred - y_target) <= self.tolerance:
                break
            # Reduce distance penalty weights to prioritize reaching prediction target
            curr_l1 *= 0.2
            curr_l2 *= 0.2

        final_pred = self._forward_score(x_cf)
        perturbation = x_cf - x_orig

        return {
            "counterfactual": x_cf,
            "perturbation": perturbation,
            "original_prediction": self._forward_score(x_orig),
            "target_prediction": y_target,
            "counterfactual_prediction": final_pred,
            "l1_distance": float(np.sum(np.abs(perturbation))),
            "l2_distance": float(np.linalg.norm(perturbation)),
            "converged": bool(abs(final_pred - y_target) <= self.tolerance * 2.0),
        }
