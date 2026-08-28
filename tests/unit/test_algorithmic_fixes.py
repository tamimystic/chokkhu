import numpy as np
import pandas as pd
from chokkhu.transformation.features import LogTransformer
from chokkhu.cleaning.outliers import handle_outliers
from chokkhu.models.ml.gradient_boosting import GradientBoosting
from chokkhu.eda.tabular.univariate import UnivariateAnalyzer
from chokkhu.eda.tabular.multivariate import MultivariateAnalyzer


def test_log_transformer_no_double_shift():
    df = pd.DataFrame({"pos": [1.0, 10.0, 100.0], "neg": [-5.0, 0.0, 5.0]})
    transformer = LogTransformer()
    res = transformer.fit_transform(df)
    assert not res.isna().any().any()
    assert np.all(np.isfinite(res.values))
    # Monotonicity test
    assert (res["neg"].diff().dropna() > 0).all()


def test_isolation_forest_outliers():
    np.random.seed(42)
    # Normal cluster with a couple of obvious outliers
    data = np.random.randn(100, 2)
    data = np.vstack([data, [50.0, 50.0], [-50.0, -50.0]])
    df = pd.DataFrame(data, columns=["f1", "f2"])

    cleaned = handle_outliers(df, method="isolation", action="remove")
    assert len(cleaned) < len(df)
    assert len(cleaned) >= 90


def test_gradient_boosting_predict_proba():
    np.random.seed(42)
    X = np.random.randn(60, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    gbm = GradientBoosting(task="classification", n_estimators=5, random_state=42)
    gbm.fit(X, y)

    probs = gbm.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)


def test_categorical_inference():
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"] * 20,
            "education_level": ["HighSchool", "BSc", "MSc", "PhD"] * 10,
        }
    )
    types = UnivariateAnalyzer.infer_data_types(df)
    # gender should be nominal, education_level should be ordinal due to level keyword
    assert "gender" in types["categorical"]["nominal"]
    assert "education_level" in types["categorical"]["ordinal"]


def test_psi_drift_calculation():
    s1 = pd.Series(np.random.randn(200))
    s2 = pd.Series(np.random.randn(200) + 0.1)
    psi = MultivariateAnalyzer._psi(s1, s2)
    assert psi >= 0.0
    assert np.isfinite(psi)
