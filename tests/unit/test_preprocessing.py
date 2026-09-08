from __future__ import annotations

import numpy as np
import pandas as pd

import chokkhu
from chokkhu.preprocessing import (
    BinaryEncoder,
    FrequencyEncoder,
    L2Scaler,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


def test_scalers():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    std_scaler = StandardScaler()
    scaled_std = std_scaler.fit_transform(data)
    assert np.allclose(np.mean(scaled_std, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(scaled_std, axis=0), [1.0, 1.0])
    inv_std = std_scaler.inverse_transform(scaled_std)
    assert np.allclose(inv_std, data)

    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_mm = minmax_scaler.fit_transform(data)
    assert scaled_mm.min() == 0.0
    assert scaled_mm.max() == 1.0
    inv_mm = minmax_scaler.inverse_transform(scaled_mm)
    assert np.allclose(inv_mm, data)

    rob_scaler = RobustScaler()
    scaled_rob = rob_scaler.fit_transform(data)
    assert scaled_rob.shape == data.shape

    maxabs_scaler = MaxAbsScaler()
    scaled_ma = maxabs_scaler.fit_transform(data)
    assert scaled_ma.max() <= 1.0

    l2_scaler = L2Scaler()
    scaled_l2 = l2_scaler.fit_transform(data)
    norms = np.linalg.norm(scaled_l2, axis=1)
    assert np.allclose(norms, [1.0, 1.0, 1.0])


def test_encoders():
    s = pd.Series(["cat", "dog", "cat", "bird"])

    le = LabelEncoder()
    encoded_le = le.fit_transform(s)
    assert len(encoded_le) == 4
    assert encoded_le.iloc[0] == encoded_le.iloc[2]

    ohe = OneHotEncoder(drop_first=False)
    encoded_ohe = ohe.fit_transform(s, prefix="animal")
    assert encoded_ohe.shape == (4, 3)

    target = pd.Series([10.0, 20.0, 10.0, 30.0])
    te = TargetEncoder(smoothing=1)
    encoded_te = te.fit_transform(s, target)
    assert len(encoded_te) == 4

    fe = FrequencyEncoder()
    encoded_fe = fe.fit_transform(s)
    assert encoded_fe.iloc[0] == 0.5

    be = BinaryEncoder()
    encoded_be = be.fit_transform(s, prefix="bin")
    assert encoded_be.shape[1] >= 2

    oe = OrdinalEncoder()
    encoded_oe = oe.fit_transform(s)
    assert len(encoded_oe) == 4


def test_full_preprocess_and_state():
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "salary": [20000.0, 30000.0, 40000.0, 50000.0],
            "city": ["Dhaka", "Chittagong", "Dhaka", "Sylhet"],
            "target": [0, 1, 0, 1],
        }
    )

    processed_train, state = chokkhu.preprocess(
        train_df, target="target", scale="standard", encode="onehot"
    )

    assert "target" in processed_train.columns
    assert "age" in processed_train.columns
    assert "salary" in processed_train.columns
    assert (
        "city_Dhaka" in processed_train.columns
        or "city_Sylhet" in processed_train.columns
    )

    test_df = pd.DataFrame(
        {"age": [25, 35], "salary": [25000.0, 35000.0], "city": ["Dhaka", "Sylhet"]}
    )

    processed_test = state.transform(test_df)
    assert len(processed_test) == 2
    assert "age" in processed_test.columns
