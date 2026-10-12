from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def cdist(
    XA: np.ndarray,
    XB: np.ndarray,
    metric: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    A = np.asarray(XA, dtype=np.float64)
    B = np.asarray(XB, dtype=np.float64)
    if A.ndim == 1:
        A = A[None, :]
    if B.ndim == 1:
        B = B[None, :]
    if metric == "euclidean":
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.maximum(0.0, np.sum(diff**2, axis=-1)))
    elif metric == "sqeuclidean":
        diff = A[:, None, :] - B[None, :, :]
        return np.maximum(0.0, np.sum(diff**2, axis=-1))
    elif metric in ("cityblock", "manhattan"):
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=-1)
    elif metric == "cosine":
        norm_A = np.linalg.norm(A, axis=1, keepdims=True)
        norm_B = np.linalg.norm(B, axis=1, keepdims=True)
        sim = (A @ B.T) / (norm_A @ norm_B.T + 1e-15)
        return np.clip(1.0 - sim, 0.0, 2.0)
    elif metric == "minkowski":
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]) ** p, axis=-1) ** (1.0 / p)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.maximum(0.0, np.sum(diff**2, axis=-1)))


def erf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    v_erf = np.vectorize(math.erf, otypes=[np.float64])
    res = v_erf(arr)
    return float(res) if np.isscalar(x) else res


def erfc(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    v_erfc = np.vectorize(math.erfc, otypes=[np.float64])
    res = v_erfc(arr)
    return float(res) if np.isscalar(x) else res


def gamma(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    v_gamma = np.vectorize(math.gamma, otypes=[np.float64])
    res = v_gamma(arr)
    return float(res) if np.isscalar(x) else res


def lgamma(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    v_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])
    res = v_lgamma(arr)
    return float(res) if np.isscalar(x) else res


def digamma(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    scalar = np.isscalar(x)
    flat = arr.ravel().copy()
    res = np.zeros_like(flat)

    for i, val in enumerate(flat):
        if val <= 0.0 and val == math.floor(val):
            res[i] = float("nan")
            continue
        v = val
        shift = 0.0
        while v < 8.0:
            shift -= 1.0 / v
            v += 1.0
        inv_v = 1.0 / v
        inv_v2 = inv_v * inv_v
        res[i] = (
            math.log(v)
            - 0.5 * inv_v
            - inv_v2 * (1.0 / 12.0 - inv_v2 * (1.0 / 120.0 - inv_v2 * (1.0 / 252.0)))
            + shift
        )
    out = res.reshape(arr.shape)
    return float(out) if scalar else out


class _NormDist:
    @staticmethod
    def pdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (np.asarray(x, dtype=np.float64) - loc) / scale
        res = (1.0 / (math.sqrt(2.0 * math.pi) * scale)) * np.exp(-0.5 * z**2)
        return float(res) if np.isscalar(x) else res

    @staticmethod
    def cdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (np.asarray(x, dtype=np.float64) - loc) / (scale * math.sqrt(2.0))
        res = 0.5 * (1.0 + erf(z))
        return float(res) if np.isscalar(x) else res

    @staticmethod
    def sf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (np.asarray(x, dtype=np.float64) - loc) / (scale * math.sqrt(2.0))
        res = 0.5 * erfc(z)
        return float(res) if np.isscalar(x) else res

    @staticmethod
    def ppf(q: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        p = np.clip(np.asarray(q, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        scalar = np.isscalar(q)
        flat_p = p.ravel()
        res = np.zeros_like(flat_p)

        a1, a2, a3, a4, a5, a6 = (
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        )
        b1, b2, b3, b4, b5 = (
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        )
        c1, c2, c3, c4, c5, c6 = (
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        )
        d1, d2, d3, d4 = (
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        )
        p_low = 0.02425
        p_high = 1.0 - p_low

        for i, val in enumerate(flat_p):
            if val < p_low:
                q_val = math.sqrt(-2.0 * math.log(val))
                res[i] = (
                    ((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5)
                    * q_val
                    + c6
                ) / ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)
            elif val > p_high:
                q_val = math.sqrt(-2.0 * math.log(1.0 - val))
                res[i] = -(
                    ((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5)
                    * q_val
                    + c6
                ) / ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)
            else:
                q_val = val - 0.5
                r = q_val * q_val
                res[i] = (
                    (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_val
                ) / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

        out = loc + scale * res.reshape(p.shape)
        return float(out) if scalar else out


norm = _NormDist()


def _incgamma_lower(a: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for n in range(1, 100):
            term *= x / (a + n)
            total += term
            if term < total * 1e-12:
                break
        return float(math.exp(a * math.log(x) - x - math.lgamma(a)) * total)
    else:
        # Lentz continued fraction for upper incomplete gamma
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            del_h = d * c
            h *= del_h
            if abs(del_h - 1.0) < 1e-12:
                break
        upper = math.exp(a * math.log(x) - x - math.lgamma(a)) * h
        return float(max(0.0, 1.0 - upper))


class _Chi2Dist:
    @staticmethod
    def cdf(x: Any, df: float) -> Any:
        arr = np.asarray(x, dtype=np.float64)
        v_func = np.vectorize(
            lambda v: _incgamma_lower(0.5 * df, 0.5 * max(0.0, float(v))),
            otypes=[np.float64],
        )
        res = v_func(arr)
        return float(res) if np.isscalar(x) else res

    @staticmethod
    def sf(x: Any, df: float) -> Any:
        return np.maximum(0.0, 1.0 - _Chi2Dist.cdf(x, df))

    @staticmethod
    def ppf(q: Any, df: float) -> Any:
        # Wilson-Hilferty transformation approximation
        z = norm.ppf(q)
        term = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
        res = df * np.maximum(0.0, term) ** 3
        return float(res) if np.isscalar(q) else res


chi2 = _Chi2Dist()


def skew(a: np.ndarray, axis: int = 0, bias: bool = True) -> np.ndarray | float:
    arr = np.asarray(a, dtype=np.float64)
    m = np.mean(arr, axis=axis, keepdims=True)
    m2 = np.mean((arr - m) ** 2, axis=axis)
    m3 = np.mean((arr - m) ** 3, axis=axis)
    res = m3 / (m2**1.5 + 1e-15)
    if not bias:
        n = arr.shape[axis]
        if n > 2:
            res *= math.sqrt(n * (n - 1)) / (n - 2)
    return float(res) if res.ndim == 0 else res


def kurtosis(
    a: np.ndarray, axis: int = 0, fisher: bool = True, bias: bool = True
) -> np.ndarray | float:
    arr = np.asarray(a, dtype=np.float64)
    m = np.mean(arr, axis=axis, keepdims=True)
    m2 = np.mean((arr - m) ** 2, axis=axis)
    m4 = np.mean((arr - m) ** 4, axis=axis)
    res = m4 / (m2**2 + 1e-15)
    if fisher:
        res -= 3.0
    return float(res) if res.ndim == 0 else res


def zscore(a: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    m = np.mean(arr, axis=axis, keepdims=True)
    s = np.std(arr, axis=axis, keepdims=True)
    return (arr - m) / (s + 1e-15)


def f_oneway(*args: Sequence[float] | np.ndarray) -> Tuple[float, float]:
    groups = [np.asarray(g, dtype=np.float64).ravel() for g in args if len(g) > 0]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0
    n_total = sum(len(g) for g in groups)
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_b = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_w = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    df_b = k - 1
    df_w = n_total - k
    if df_w <= 0 or ss_w <= 1e-15:
        return 0.0, 1.0
    ms_b = ss_b / df_b
    ms_w = ss_w / df_w
    f_stat = float(ms_b / ms_w)
    # p-value approximation via F to Chi2 or normal
    p_val = float(chi2.sf(f_stat * df_b, df=df_b))
    return f_stat, p_val


def ttest_ind(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    equal_var: bool = True,
) -> Tuple[float, float]:
    x1 = np.asarray(a, dtype=np.float64).ravel()
    x2 = np.asarray(b, dtype=np.float64).ravel()
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2) + 1e-15)
    else:
        se = math.sqrt(v1 / n1 + v2 / n2 + 1e-15)
    t_stat = float((m1 - m2) / se)
    p_val = float(2.0 * norm.sf(abs(t_stat)))
    return t_stat, p_val


def kruskal(*args: Sequence[float] | np.ndarray) -> Tuple[float, float]:
    groups = [np.asarray(g, dtype=np.float64).ravel() for g in args if len(g) > 0]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0
    all_vals = np.concatenate(groups)
    n = len(all_vals)
    ranks = np.argsort(np.argsort(all_vals)) + 1.0
    r_sums: list[float] = []
    idx = 0
    for g in groups:
        r_sums.append(float(np.sum(ranks[idx : idx + len(g)])))
        idx += len(g)
    h_stat = float(
        (12.0 / float(n * (n + 1)))
        * sum(float(r**2) / len(g) for r, g in zip(r_sums, groups))
        - 3.0 * float(n + 1)
    )
    p_val = float(chi2.sf(max(0.0, h_stat), df=k - 1))
    return max(0.0, h_stat), p_val


def chi2_contingency(observed: np.ndarray) -> Tuple[float, float, int, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)
    r_sums = np.sum(obs, axis=1, keepdims=True)
    c_sums = np.sum(obs, axis=0, keepdims=True)
    total: float = float(np.sum(obs))
    expected = (r_sums @ c_sums) / max(total, 1e-15)
    chi2_val = float(np.sum((obs - expected) ** 2 / (expected + 1e-15)))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    p_val = float(chi2.sf(chi2_val, df=max(1, dof)))
    return chi2_val, p_val, dof, expected


def shapiro(x: Sequence[float] | np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)
    if n < 3:
        return 1.0, 1.0
    s_val = float(skew(arr))
    k_val = float(kurtosis(arr))
    k2 = s_val**2 + k_val**2
    p_val = float(chi2.sf(k2, df=2))
    w_stat = float(1.0 / (1.0 + 0.1 * k2))
    return w_stat, p_val


def kstest(
    rvs: Sequence[float] | np.ndarray, cdf: str | Callable[[float], float] = "norm"
) -> Tuple[float, float]:
    arr = np.sort(np.asarray(rvs, dtype=np.float64).ravel())
    n = len(arr)
    if n == 0:
        return 0.0, 1.0
    cdf_fn: Callable[[Any], Any]
    if isinstance(cdf, str) and cdf == "norm":
        cdf_fn = norm.cdf
    elif callable(cdf):
        cdf_fn = cdf
    else:
        cdf_fn = norm.cdf
    cdf_vals = np.asarray([float(cdf_fn(v)) for v in arr])
    d_plus: float = float(np.max((np.arange(1, n + 1) / n) - cdf_vals))
    d_minus: float = float(np.max(cdf_vals - (np.arange(0, n) / n)))
    d: float = max(d_plus, d_minus)
    p_val = float(2.0 * math.exp(-2.0 * n * d**2))
    return d, min(1.0, max(0.0, p_val))


def probplot(
    x: Sequence[float] | np.ndarray, dist: str = "norm", plot: Any = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float, float]]:
    arr = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = len(arr)
    quantiles = norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    slope, intercept = float(np.polyfit(quantiles, arr, 1)[0]), float(
        np.polyfit(quantiles, arr, 1)[1]
    )
    r = float(np.corrcoef(quantiles, arr)[0, 1])
    if plot is not None and hasattr(plot, "plot"):
        plot.plot(quantiles, arr, "bo", markersize=3)
        plot.plot(quantiles, slope * quantiles + intercept, "r-")
    return (quantiles, arr), (slope, intercept, r)


def expm(A: np.ndarray) -> np.ndarray:
    M = np.asarray(A, dtype=np.float64)
    norm_val = float(np.max(np.sum(np.abs(M), axis=1)))
    n_squarings = max(0, int(math.ceil(math.log2(norm_val + 1e-12))))
    M_scaled = M / (2.0**n_squarings)
    term: np.ndarray = np.eye(M.shape[0], dtype=np.float64)
    res = term.copy()
    for k in range(1, 20):
        term = (term @ M_scaled) / k
        res += term
        if np.max(np.abs(term)) < 1e-15:
            break
    for _ in range(n_squarings):
        res = res @ res
    return res


class OptimizeResult:
    def __init__(
        self, x: np.ndarray, fun: float, success: bool = True, nit: int = 0
    ) -> None:
        self.x = x
        self.fun = fun
        self.success = success
        self.nit = nit


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
) -> OptimizeResult:
    opts = options or {}
    maxiter = int(opts.get("maxiter", 150))
    ftol = float(opts.get("ftol", 1e-8))
    x = np.array(x0, dtype=np.float64).ravel()
    lr = 0.05
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    prev_loss = float("inf")

    for it in range(1, maxiter + 1):
        if jac is not None:
            grad = jac(x)
            loss = fun(x)
        else:
            loss = fun(x)
            grad = np.zeros_like(x)
            h = 1e-5
            for i in range(len(x)):
                x[i] += h
                grad[i] = (fun(x) - loss) / h
                x[i] -= h

        if abs(prev_loss - loss) < ftol:
            break
        prev_loss = loss

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**it)
        v_hat = v / (1.0 - beta2**it)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        if bounds is not None:
            for i, b in enumerate(bounds):
                if b is not None:
                    low, high = b
                    if low is not None:
                        x[i] = max(x[i], low)
                    if high is not None:
                        x[i] = min(x[i], high)

    return OptimizeResult(x=x, fun=float(fun(x)), success=True, nit=it)


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = np.array(cost_matrix, dtype=np.float64, copy=True)
    n_rows, n_cols = C.shape
    row_ind = np.arange(n_rows, dtype=np.int64)
    col_ind = np.zeros(n_rows, dtype=np.int64)
    assigned_cols: set[int] = set()

    for r in range(n_rows):
        sorted_cols = np.argsort(C[r])
        chosen = -1
        for c in sorted_cols:
            if c not in assigned_cols:
                chosen = int(c)
                break
        if chosen == -1:
            chosen = int(sorted_cols[0])
        assigned_cols.add(chosen)
        col_ind[r] = chosen

    return row_ind, col_ind
