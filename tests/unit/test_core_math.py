import math

import numpy as np

from chokkhu.core.math import (
    cdist,
    chi2,
    chi2_contingency,
    digamma,
    erf,
    erfc,
    expm,
    f_oneway,
    gamma,
    kruskal,
    kstest,
    kurtosis,
    lgamma,
    linear_sum_assignment,
    minimize,
    norm,
    probplot,
    shapiro,
    skew,
    ttest_ind,
    zscore,
)


def test_cdist_metrics():
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Euclidean
    d_euc = cdist(A, B, metric="euclidean")
    assert d_euc.shape == (2, 2)
    assert np.isclose(d_euc[0, 0], 1.0)
    assert np.isclose(d_euc[0, 1], 1.0)

    # Squared Euclidean
    d_sq = cdist(A, B, metric="sqeuclidean")
    assert np.allclose(d_sq, d_euc**2)

    # Manhattan
    d_man = cdist(A, B, metric="cityblock")
    assert np.isclose(d_man[0, 0], 1.0)
    assert np.isclose(d_man[1, 0], 1.0)

    # Cosine
    d_cos = cdist(A, B, metric="cosine")
    assert d_cos.shape == (2, 2)

    # Minkowski
    d_mink = cdist(A, B, metric="minkowski", p=3.0)
    assert d_mink.shape == (2, 2)


def test_special_functions():
    vals = np.array([0.0, 0.5, 1.0, 2.0])
    # erf & erfc
    for v in vals:
        assert np.isclose(erf(v) + erfc(v), 1.0)
    assert np.isclose(erf(0.0), 0.0)

    # gamma & lgamma
    assert np.isclose(gamma(5.0), 24.0)
    assert np.isclose(lgamma(5.0), math.log(24.0))

    # digamma
    # For integer n: psi(n) = -euler_mascheroni + sum_{k=1}^{n-1} 1/k
    euler_gamma = 0.5772156649015329
    assert np.isclose(digamma(1.0), -euler_gamma, atol=1e-5)
    assert np.isclose(digamma(2.0), 1.0 - euler_gamma, atol=1e-5)
    assert np.isclose(digamma(3.0), 1.5 - euler_gamma, atol=1e-5)


def test_norm_distribution():
    assert np.isclose(norm.cdf(0.0), 0.5)
    assert np.isclose(norm.sf(0.0), 0.5)
    assert np.isclose(norm.ppf(0.5), 0.0)
    assert np.isclose(norm.ppf(0.975), 1.95996, atol=1e-3)
    assert np.isclose(norm.pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))


def test_chi2_distribution():
    # Chi2(df=2) CDF is 1 - exp(-x/2)
    assert np.isclose(chi2.cdf(2.0, df=2), 1.0 - math.exp(-1.0), atol=1e-4)
    assert np.isclose(chi2.sf(2.0, df=2), math.exp(-1.0), atol=1e-4)
    ppf_val = chi2.ppf(0.5, df=2)
    assert ppf_val > 0.0


def test_statistical_tests():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    s = skew(data)
    assert abs(s) < 1e-10
    k = kurtosis(data)
    assert isinstance(k, float)
    z = zscore(data)
    assert np.isclose(np.mean(z), 0.0, atol=1e-10)
    assert np.isclose(np.std(z), 1.0, atol=1e-5)

    # ANOVA & t-test
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([4.0, 5.0, 6.0])
    f_stat, p_val = f_oneway(g1, g2)
    assert f_stat > 0
    assert 0.0 <= p_val <= 1.0

    t_stat, p_t = ttest_ind(g1, g2)
    assert t_stat < 0
    assert 0.0 <= p_t <= 1.0

    # Kruskal
    h_stat, p_k = kruskal(g1, g2)
    assert h_stat >= 0
    assert 0.0 <= p_k <= 1.0

    # Chi2 contingency
    obs = np.array([[10, 20], [20, 40]])
    c_stat, p_c, dof, exp = chi2_contingency(obs)
    assert np.isclose(c_stat, 0.0, atol=1e-5)

    # Shapiro & KS
    w, p_w = shapiro(data)
    assert 0.0 <= w <= 1.0
    d, p_d = kstest(z, "norm")
    assert 0.0 <= d <= 1.0

    # Probplot
    (q, arr), (slope, intercept, r) = probplot(data)
    assert len(q) == len(data)
    assert r > 0.95


def test_matrix_exponential():
    A = np.zeros((2, 2))
    assert np.allclose(expm(A), np.eye(2))

    diag = np.diag([1.0, 2.0])
    expected = np.diag([math.exp(1.0), math.exp(2.0)])
    assert np.allclose(expm(diag), expected)


def test_optimizer_minimize():
    def obj(x):
        return (x[0] - 2.0) ** 2 + (x[1] + 3.0) ** 2

    res = minimize(obj, x0=np.array([0.0, 0.0]), bounds=[(0.0, 5.0), (-5.0, 0.0)])
    assert np.isclose(res.x[0], 2.0, atol=1e-2)
    assert np.isclose(res.x[1], -3.0, atol=1e-2)


def test_linear_sum_assignment():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    r_ind, c_ind = linear_sum_assignment(cost)
    assert len(r_ind) == 3
    assert len(c_ind) == 3
    assert len(set(c_ind)) == 3
