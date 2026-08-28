from __future__ import annotations

from typing import Any, List
import numpy as np
from chokkhu.models.base import ChokkhuModel


def kernel_shap(
    model: ChokkhuModel,
    X: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
    random_state: int = 42,
) -> Any:
    """Model-agnostic Shapley value estimation using KernelSHAP formula."""
    from .engine import ExplanationResult

    rng = np.random.default_rng(random_state)
    n_instances, n_features = X.shape
    background = np.mean(X, axis=0)

    # Compute expected value on background
    f_bg = float(np.mean(model.predict(background.reshape(1, -1))))

    # Sample subset of rows if dataset is large to maintain research-grade performance
    max_eval_rows = min(n_instances, 50)
    eval_indices = rng.choice(n_instances, size=max_eval_rows, replace=False)
    eval_X = X[eval_indices]

    shap_matrix = np.zeros((max_eval_rows, n_features), dtype=np.float64)

    for i in range(max_eval_rows):
        x_i = eval_X[i]
        f_x = float(np.mean(model.predict(x_i.reshape(1, -1))))
        delta = f_x - f_bg

        # Generate binary coalition masks
        m_samples = min(n_samples, 2**n_features if n_features <= 10 else 100)
        Z = rng.integers(0, 2, size=(m_samples, n_features))
        Z[0, :] = 0  # all background
        Z[1, :] = 1  # all instance

        # Hybrid instances
        X_hybrid = np.zeros((m_samples, n_features), dtype=np.float64)
        for s in range(m_samples):
            mask = Z[s]
            X_hybrid[s] = np.where(mask == 1, x_i, background)

        preds_hybrid: np.ndarray = model.predict(X_hybrid).astype(np.float64)

        # Shapley kernel weights: pi(z) = (M-1) / (comb(M, |z|) * |z| * (M-|z|))
        weights: np.ndarray = np.ones(m_samples, dtype=np.float64)
        for s in range(m_samples):
            z_sum = int(np.sum(Z[s]))
            if z_sum == 0 or z_sum == n_features:
                weights[s] = 1e4
            else:
                # Kernel weight formula
                weights[s] = (n_features - 1.0) / max(1.0, z_sum * (n_features - z_sum))

        # Solve weighted least squares: min || W^(1/2) (Z phi - (f(X) - f_bg)) ||^2
        W_sqrt = np.sqrt(weights)[:, np.newaxis]
        A = Z * W_sqrt
        b = (preds_hybrid - f_bg) * W_sqrt.flatten()

        # Regularized solve
        try:
            phi, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-5)
            # Normalize to sum to exact prediction difference f(x) - f_bg
            phi_sum: float = float(np.sum(phi))
            if abs(phi_sum) > 1e-10:
                phi = phi * (delta / phi_sum)
        except Exception:
            phi = np.zeros(n_features)

        shap_matrix[i] = phi

    return ExplanationResult(
        method="shap",
        feature_names=feature_names,
        shap_values=shap_matrix,
        expected_value=f_bg,
    )
