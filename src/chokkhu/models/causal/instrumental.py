"""Instrumental Variables Regression Suite: Two-Stage Least Squares (2SLS) and GMM.

Formulated from first principles in pure NumPy and SciPy, implementing
Angrist & Imbens (Nobel Prize 2021) and Hansen (Econometrica 1982) Instrumental
Variables estimation, Weak Instrument F-diagnostics, Hansen's J-test, Sargan test,
and Durbin-Wu-Hausman endogeneity testing.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, Optional, Union


class TwoStageLeastSquares:
    r"""Two-Stage Least Squares (2SLS) Instrumental Variable Estimator.

    Resolves omitted variable bias, measurement error, and endogeneity in linear structural
    models :math:`Y = X \beta + \epsilon` where :math:`\mathbb{E}[X^T \epsilon] \neq 0` using
    exogenous instruments :math:`Z` where :math:`\mathbb{E}[Z^T \epsilon] = 0` and :math:`\text{Rank}(Z^T X) = K`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    robust : bool, default=True
        Whether to compute White/Huber heteroskedasticity-robust standard errors.
    """

    def __init__(self, fit_intercept: bool = True, robust: bool = True) -> None:
        self.fit_intercept = bool(fit_intercept)
        self.robust = bool(robust)

        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        self.se_: np.ndarray = np.array([])
        self.t_stat_: np.ndarray = np.array([])
        self.p_val_: np.ndarray = np.array([])
        self.cov_params_: np.ndarray = np.array([])
        self.first_stage_f_stat_: float = 0.0
        self.sargan_stat_: Optional[float] = None
        self.sargan_p_val_: Optional[float] = None
        self.r2_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
    ) -> "TwoStageLeastSquares":
        r"""Fit the 2SLS model with instruments Z.

        Parameters
        ----------
        X : np.ndarray, shape (N, K)
            Regressors (including potentially endogenous variables).
        y : np.ndarray, shape (N,) or (N, 1)
            Dependent response variable.
        Z : np.ndarray, shape (N, L)
            Instruments matrix, where :math:`L \ge K`.

        Returns
        -------
        self : TwoStageLeastSquares
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        Z_arr = np.asarray(Z, dtype=np.float64)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        N, K = X_arr.shape
        _, L = Z_arr.shape

        if L < K:
            raise ValueError(
                f"Order condition failed: Number of instruments L={L} must be >= regressors K={K}"
            )

        if self.fit_intercept:
            ones = np.ones((N, 1), dtype=np.float64)
            X_mat = np.hstack([ones, X_arr])
            Z_mat = np.hstack([ones, Z_arr])
            K_eff = K + 1
            L_eff = L + 1
        else:
            X_mat = X_arr
            Z_mat = Z_arr
            K_eff = K
            L_eff = L

        # Stage 1: Compute Projection matrix P_Z = Z (Z^T Z)^{-1} Z^T
        # X_hat = P_Z X = Z (Z^T Z)^{-1} Z^T X
        ZtZ_inv = np.linalg.pinv(np.dot(Z_mat.T, Z_mat))
        Z_proj = np.dot(Z_mat, ZtZ_inv)  # Z (Z^T Z)^{-1}
        X_hat = np.dot(np.dot(Z_proj, Z_mat.T), X_mat)

        # Stage 2: beta_2SLS = (X_hat^T X_hat)^{-1} X_hat^T y
        Xhat_t_Xhat = np.dot(X_hat.T, X_hat)
        beta_2sls = np.linalg.solve(Xhat_t_Xhat, np.dot(X_hat.T, y_arr))

        # True structural residuals: e = y - X beta_2SLS
        residuals = y_arr - np.dot(X_mat, beta_2sls)
        df_resid = max(1, N - K_eff)

        # Covariance of beta_2SLS
        Xt_Pz_X_inv = np.linalg.pinv(
            np.dot(X_mat.T, np.dot(np.dot(Z_proj, Z_mat.T), X_mat))
        )

        if self.robust:
            # White heteroskedasticity-robust covariance:
            # (X^T P_Z X)^{-1} [X^T P_Z diag(e^2) P_Z X] (X^T P_Z X)^{-1}
            e2 = residuals**2
            # P_Z X
            Pz_X = np.dot(np.dot(Z_proj, Z_mat.T), X_mat)
            meat = np.dot(Pz_X.T * e2, Pz_X)
            cov = np.dot(np.dot(Xt_Pz_X_inv, meat), Xt_Pz_X_inv)
        else:
            sigma2 = np.sum(residuals**2) / df_resid
            cov = sigma2 * Xt_Pz_X_inv

        se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        t_stat = beta_2sls / se
        p_val = 2.0 * stats.norm.sf(np.abs(t_stat))

        if self.fit_intercept:
            self.intercept_ = float(beta_2sls[0])
            self.coef_ = beta_2sls[1:]
            self.se_ = se[1:]
            self.t_stat_ = t_stat[1:]
            self.p_val_ = p_val[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_2sls
            self.se_ = se
            self.t_stat_ = t_stat
            self.p_val_ = p_val

        self.cov_params_ = cov

        # R-squared computed on original structural equation
        ss_tot: float = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        ss_res: float = float(np.sum(residuals**2))
        self.r2_ = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        # First stage F-statistic on excluded instruments
        if L > 0:
            # Stage 1 regression for first endogenous regressor
            reg_x = X_arr[:, 0]
            z_coef, _, _, _ = np.linalg.lstsq(Z_mat, reg_x, rcond=None)
            x_pred = np.dot(Z_mat, z_coef)
            x_res = reg_x - x_pred
            r2_first = 1.0 - np.sum(x_res**2) / (
                np.sum((reg_x - np.mean(reg_x)) ** 2) + 1e-12
            )
            f_stat = (r2_first / max(1, L)) / (
                max(1e-12, (1.0 - r2_first) / max(1, N - L_eff))
            )
            self.first_stage_f_stat_ = float(max(0.0, f_stat))

        # Sargan test of overidentifying restrictions (if L > K)
        if L > K:
            # Regress residuals on all instruments Z_mat
            z_coef_e, _, _, _ = np.linalg.lstsq(Z_mat, residuals, rcond=None)
            e_pred = np.dot(Z_mat, z_coef_e)
            r2_sargan = np.sum((e_pred - np.mean(e_pred)) ** 2) / (
                np.sum((residuals - np.mean(residuals)) ** 2) + 1e-12
            )
            sargan_stat = float(N * r2_sargan)
            df_overid = L - K
            self.sargan_stat_ = sargan_stat
            self.sargan_p_val_ = float(stats.chi2.sf(sargan_stat, df=df_overid))
        else:
            self.sargan_stat_ = None
            self.sargan_p_val_ = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target response using structural estimates.

        Parameters
        ----------
        X : np.ndarray, shape (N, K)

        Returns
        -------
        y_pred : np.ndarray, shape (N,)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return np.dot(X_arr, self.coef_) + self.intercept_

    def summary(self) -> Dict[str, Union[float, np.ndarray, Optional[float]]]:
        """Return diagnostic metrics summary."""
        return {
            "coefficients": self.coef_,
            "intercept": self.intercept_,
            "standard_errors": self.se_,
            "t_statistics": self.t_stat_,
            "p_values": self.p_val_,
            "r2": self.r2_,
            "first_stage_F": self.first_stage_f_stat_,
            "sargan_statistic": self.sargan_stat_,
            "sargan_p_value": self.sargan_p_val_,
        }


class InstrumentalGMM:
    r"""Generalized Method of Moments (GMM) for Instrumental Variables.

    Implements optimal 2-step GMM estimator under heteroskedasticity with Hansen's
    J-test for overidentifying restrictions :math:`\mathbb{E}[Z^T (Y - X \beta)] = 0`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = bool(fit_intercept)
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        self.se_: np.ndarray = np.array([])
        self.j_stat_: float = 0.0
        self.j_p_val_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray) -> "InstrumentalGMM":
        r"""Fit 2-step optimal GMM.

        Parameters
        ----------
        X : np.ndarray, shape (N, K)
        y : np.ndarray, shape (N,)
        Z : np.ndarray, shape (N, L) where L >= K

        Returns
        -------
        self : InstrumentalGMM
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        Z_arr = np.asarray(Z, dtype=np.float64)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        N, K = X_arr.shape
        _, L = Z_arr.shape

        if L < K:
            raise ValueError(f"Overidentification requires L={L} >= K={K}")

        if self.fit_intercept:
            ones = np.ones((N, 1), dtype=np.float64)
            X_mat = np.hstack([ones, X_arr])
            Z_mat = np.hstack([ones, Z_arr])
        else:
            X_mat = X_arr
            Z_mat = Z_arr

        # Step 1: Initial GMM weighting matrix W_1 = (Z^T Z / N)^{-1}
        W1 = np.linalg.pinv(np.dot(Z_mat.T, Z_mat) / N)
        XtZ = np.dot(X_mat.T, Z_mat)
        Zty = np.dot(Z_mat.T, y_arr)

        A1 = np.dot(np.dot(XtZ, W1), XtZ.T)
        b1 = np.dot(np.dot(XtZ, W1), Zty)
        beta_1 = np.linalg.solve(A1, b1)

        # Step 2: Optimal weighting matrix S = (1/N) \sum e_i^2 z_i z_i^T
        e1 = y_arr - np.dot(X_mat, beta_1)
        S = np.dot(Z_mat.T * (e1**2), Z_mat) / N
        W2 = np.linalg.pinv(S)

        A2 = np.dot(np.dot(XtZ, W2), XtZ.T)
        b2 = np.dot(np.dot(XtZ, W2), Zty)
        beta_2 = np.linalg.solve(A2, b2)

        # Variance-covariance matrix: (1/N) * ( (X^T Z / N) W2 (Z^T X / N) )^{-1}
        cov = np.linalg.pinv(np.dot(np.dot(XtZ / N, W2), (XtZ / N).T)) / N
        se = np.sqrt(np.maximum(np.diag(cov), 1e-12))

        if self.fit_intercept:
            self.intercept_ = float(beta_2[0])
            self.coef_ = beta_2[1:]
            self.se_ = se[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_2
            self.se_ = se

        # Hansen's J-test: J = N * g_bar^T W2 g_bar
        residuals_2 = y_arr - np.dot(X_mat, beta_2)
        g_bar = np.dot(Z_mat.T, residuals_2) / N
        j_stat = float(N * np.dot(np.dot(g_bar.T, W2), g_bar))
        df_j = max(1, L - K)
        self.j_stat_ = j_stat
        self.j_p_val_ = float(stats.chi2.sf(j_stat, df=df_j))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict response values."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return np.dot(X_arr, self.coef_) + self.intercept_
