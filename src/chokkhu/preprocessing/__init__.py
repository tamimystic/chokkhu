from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger

from .encoding import (
    BinaryEncoder,
    FrequencyEncoder,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from .feature_selection import (
    ANOVASelector,
    CorrelationFilterSelector,
    MutualInfoSelector,
    RFESelector,
    VarianceThresholdSelector,
)
from .scaling import (
    L2Scaler,
    MaxAbsScaler,
    MinMaxScaler,
    PowerScaler,
    QuantileScaler,
    RobustScaler,
    StandardScaler,
    get_scaler,
)


class PreprocessorState:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.target_col = None
        self.encoded_cols = []
        self.num_cols = []
        self.cat_cols = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, enc in self.encoders.items():
            if col not in df.columns:
                continue
            if isinstance(enc, OneHotEncoder):
                encoded_df = enc.transform(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            elif isinstance(enc, BinaryEncoder):
                encoded_df = enc.transform(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                df[col] = enc.transform(df[col])
        for col, sc in self.scalers.items():
            if col in df.columns:
                df[col] = sc.transform(df[[col]].values).flatten()
        if self.feature_selector is not None:
            df = self.feature_selector.transform(df)
        return df


def preprocess(
    data: pd.DataFrame,
    target: str = None,
    scale: str = None,
    encode: str = None,
    select_features: str = None,
    select_k: int = None,
    save_report: bool = False,
    report_dir: str = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, PreprocessorState]:
    df = data.copy()
    state = PreprocessorState()
    state.target_col = target

    target_series = df[target] if target is not None and target in df.columns else None
    if target is not None and target in df.columns:
        features_df = df.drop(columns=[target])
    else:
        features_df = df

    state.num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    state.cat_cols = features_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if encode is not None:
        for col in state.cat_cols:
            if encode == "label":
                enc: Any = LabelEncoder()
                features_df[col] = enc.fit_transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "onehot":
                enc = OneHotEncoder()
                encoded_df = enc.fit_transform(features_df[col], prefix=col)
                features_df = pd.concat(
                    [features_df.drop(columns=[col]), encoded_df], axis=1
                )
                state.encoders[col] = enc
            elif encode == "binary":
                enc = BinaryEncoder()
                encoded_df = enc.fit_transform(features_df[col], prefix=col)
                features_df = pd.concat(
                    [features_df.drop(columns=[col]), encoded_df], axis=1
                )
                state.encoders[col] = enc
            elif encode == "ordinal":
                enc = OrdinalEncoder()
                features_df[col] = enc.fit_transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "frequency":
                enc = FrequencyEncoder()
                features_df[col] = enc.fit_transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "target" and target_series is not None:
                enc = TargetEncoder()
                features_df[col] = enc.fit_transform(features_df[col], target_series)
                state.encoders[col] = enc

    if scale is not None:
        cols_to_scale = features_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols_to_scale:
            scaler = get_scaler(scale)
            features_df[col] = scaler.fit_transform(features_df[[col]].values).flatten()
            state.scalers[col] = scaler

    if select_features is not None:
        k = select_k if select_k is not None else 10
        if select_features == "variance":
            fs: Any = VarianceThresholdSelector()
            features_df = fs.fit_transform(features_df)
            state.feature_selector = fs
        elif select_features == "correlation":
            fs = CorrelationFilterSelector()
            features_df = fs.fit_transform(features_df, target_series)
            state.feature_selector = fs
        elif select_features == "mutual_info" and target_series is not None:
            fs = MutualInfoSelector(k=k)
            features_df = fs.fit_transform(features_df, target_series)
            state.feature_selector = fs
        elif select_features == "anova" and target_series is not None:
            fs = ANOVASelector(k=k)
            features_df = fs.fit_transform(features_df, target_series)
            state.feature_selector = fs
        elif select_features == "rfe" and target_series is not None:
            fs = RFESelector(k=k)
            features_df = fs.fit_transform(features_df, target_series)
            state.feature_selector = fs

    if target_series is not None and target is not None:
        features_df[target] = target_series

    if verbose:
        Logger.info(f"Preprocessed features: {data.shape} -> {features_df.shape}")

    return features_df, state


__all__ = [
    "preprocess",
    "PreprocessorState",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "L2Scaler",
    "PowerScaler",
    "QuantileScaler",
    "get_scaler",
    "LabelEncoder",
    "OneHotEncoder",
    "BinaryEncoder",
    "OrdinalEncoder",
    "FrequencyEncoder",
    "TargetEncoder",
    "VarianceThresholdSelector",
    "CorrelationFilterSelector",
    "MutualInfoSelector",
    "ANOVASelector",
    "RFESelector",
]
