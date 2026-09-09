from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger

from .splitter import kfold, time_series_split, train_test_split


def split(
    data: Union[pd.DataFrame, np.ndarray, dict],
    target: str = None,
    test_size: float = 0.2,
    val_size: float = None,
    stratify: bool = False,
    shuffle: bool = True,
    random_state: int = None,
    cv: int = None,
    cv_stratified: bool = False,
    time_series: bool = False,
    n_splits: int = 5,
    verbose: bool = True,
):
    if isinstance(data, dict) and "X" in data and ("y" in data):
        X, y = (data["X"], data["y"])
    elif (
        isinstance(data, pd.DataFrame)
        and target is not None
        and (target in data.columns)
    ):
        X, y = (data.drop(columns=[target]), data[target])
    else:
        X, y = (data, None)
    if cv is not None:
        if verbose:
            Logger.info(f"Generated {cv}-fold cross-validation generator.")
        return kfold(
            X,
            y,
            k=cv,
            stratified=cv_stratified,
            shuffle=shuffle,
            random_state=random_state,
        )
    elif time_series:
        if verbose:
            Logger.info(
                f"Generated {n_splits}-split time-series cross-validation generator."
            )
        return time_series_split(X, y, n_splits=n_splits)
    else:
        res = train_test_split(
            X,
            y,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
            stratify=stratify,
            random_state=random_state,
        )
        if verbose:
            Logger.info("Data split completed successfully.")
        return res


__all__ = ["split", "train_test_split", "kfold", "time_series_split"]
