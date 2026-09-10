"""Sparse Symbolic Regression & Equation Discovery (SINDy Framework).

Pure NumPy implementation of Sparse Identification of Non-linear Dynamics:
- Library of polynomial, trigonometric, and transcendental basis functions
- Sequentially Thresholded Least Squares (STLSQ) optimizer
- Extracts clean, parsimonious symbolic mathematical formulas from observational data
"""

from typing import List, Optional, Tuple
import numpy as np


class SymbolicRegressor:
    r"""Sparse Symbolic Regressor using Sequentially Thresholded Least Squares (SINDy).

    Constructs a candidate library :math:`\Theta(X)` and solves:

    .. math::
        \min_{\Xi} \|Y - \Theta(X) \Xi\|_2^2 + \lambda \|\Xi\|_0

    Parameters
    ----------
    threshold : float, default=0.1
        Sparsity threshold for coefficient truncation.
    degree : int, default=2
        Maximum polynomial degree.
    include_trig : bool, default=True
        Whether to include sin(x) and cos(x) in candidate library.
    max_iter : int, default=20
        Maximum STLSQ iterations.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        degree: int = 2,
        include_trig: bool = True,
        max_iter: int = 20,
    ) -> None:
        self.threshold = threshold
        self.degree = degree
        self.include_trig = include_trig
        self.max_iter = max_iter

        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _build_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]

        n_samples, n_features = X_arr.shape
        cols: List[np.ndarray] = [np.ones((n_samples, 1), dtype=np.float64)]
        names: List[str] = ["1"]

        # Linear terms
        for j in range(n_features):
            cols.append(X_arr[:, [j]])
            names.append(f"x{j}")

        # Polynomial terms up to degree
        if self.degree >= 2:
            for j1 in range(n_features):
                for j2 in range(j1, n_features):
                    cols.append((X_arr[:, [j1]] * X_arr[:, [j2]]))
                    names.append(f"x{j1}*x{j2}" if j1 != j2 else f"x{j1}^2")

        if self.degree >= 3:
            for j1 in range(n_features):
                for j2 in range(j1, n_features):
                    for j3 in range(j2, n_features):
                        cols.append((X_arr[:, [j1]] * X_arr[:, [j2]] * X_arr[:, [j3]]))
                        names.append(f"x{j1}*x{j2}*x{j3}")

        # Trigonometric terms
        if self.include_trig:
            for j in range(n_features):
                cols.append(np.sin(X_arr[:, [j]]))
                names.append(f"sin(x{j})")
                cols.append(np.cos(X_arr[:, [j]]))
                names.append(f"cos(x{j})")

        theta = np.concatenate(cols, axis=1)
        return theta, names

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        """Fit sparse symbolic regression model."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        theta, names = self._build_library(X_arr)

        # Initial least-squares solution
        xi, _, _, _ = np.linalg.lstsq(theta, y_arr, rcond=None)

        # Sequentially Thresholded Least Squares (STLSQ)
        for _ in range(self.max_iter):
            small_idx = np.abs(xi) < self.threshold
            xi[small_idx] = 0.0

            active_idx = np.where(~small_idx.ravel())[0]
            if len(active_idx) == 0:
                break

            # Re-solve least squares only on active features
            theta_active = theta[:, active_idx]
            xi_active, _, _, _ = np.linalg.lstsq(theta_active, y_arr, rcond=None)
            xi[active_idx] = xi_active

        self.coef_ = xi.ravel()
        self.feature_names_ = names
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict y using discovered symbolic equation."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        theta, _ = self._build_library(X_arr)
        return np.dot(theta, self.coef_)

    def equation(self) -> str:
        """Return human-readable algebraic formula."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling equation.")

        terms = []
        for c, name in zip(self.coef_, self.feature_names_):
            if abs(c) > 1e-4:
                if name == "1":
                    terms.append(f"{c:.4f}")
                else:
                    terms.append(f"{c:.4f}*{name}")

        return " + ".join(terms) if terms else "0.0"
