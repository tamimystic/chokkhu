import numpy as np
import pandas as pd
import chokkhu as ck
from chokkhu.preprocessing.scaling import PowerScaler, QuantileScaler
from chokkhu.preprocessing.feature_selection import RFESelector


def test_power_scaler():
    np.random.seed(42)
    X = np.exp(np.random.randn(100, 2))
    scaler = PowerScaler()
    X_trans = scaler.fit_transform(X)
    assert X_trans.shape == X.shape
    assert scaler.lambdas is not None
    assert len(scaler.lambdas) == 2


def test_quantile_scaler():
    np.random.seed(42)
    X = np.random.exponential(scale=2.0, size=(100, 2))
    scaler = QuantileScaler(output_distribution="uniform", n_quantiles=50)
    X_trans = scaler.fit_transform(X)
    assert np.all(X_trans >= 0.0) and np.all(X_trans <= 1.0)


def test_rfe_selector():
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "strong": np.random.randn(80) * 10,
            "noise1": np.random.randn(80),
            "noise2": np.random.randn(80),
            "noise3": np.random.randn(80),
        }
    )
    target = 5.0 * df["strong"] + np.random.randn(80) * 0.1

    rfe = RFESelector(k=1)
    df_selected = rfe.fit_transform(df, target)
    assert df_selected.shape[1] == 1
    assert "strong" in df_selected.columns


def test_preprocess_with_new_scalers_and_selectors():
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "val1": np.random.randn(60) * 5 + 10,
            "val2": np.random.randn(60) * 2,
            "cat": np.random.choice(["A", "B"], size=60),
            "target": np.random.choice([0, 1], size=60),
        }
    )
    processed_df, state = ck.preprocess(
        df,
        target="target",
        scale="power",
        encode="label",
        select_features="rfe",
        select_k=2,
        verbose=False,
    )
    assert processed_df.shape[0] == 60
    assert processed_df.shape[1] == 3  # 2 selected features + target
    assert "target" in processed_df.columns
