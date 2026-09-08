from __future__ import annotations

from typing import Any, Generator

import numpy as np
import pandas as pd


def train_test_split(
    X: Any,
    y: Any = None,
    test_size: float = 0.2,
    val_size: float = None,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int = None,
):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    if stratify and y is not None:
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        train_list: list = []
        test_list: list = []
        val_list: list = []
        for c in classes:
            c_idx = np.where(y_arr == c)[0]
            if shuffle:
                np.random.shuffle(c_idx)
            n_c = len(c_idx)
            if test_size > 0 and n_c >= 2:
                n_test = min(n_c - 1, max(1, int(n_c * test_size)))
            else:
                n_test = int(n_c * test_size)
            if val_size is not None:
                if val_size > 0 and (n_c - n_test) >= 2:
                    n_val = min(n_c - n_test - 1, max(1, int(n_c * val_size)))
                else:
                    n_val = int(n_c * val_size)
                test_list.extend(c_idx[:n_test])
                val_list.extend(c_idx[n_test : n_test + n_val])
                train_list.extend(c_idx[n_test + n_val :])
            else:
                test_list.extend(c_idx[:n_test])
                train_list.extend(c_idx[n_test:])
        train_idx = np.array(train_list)
        test_idx = np.array(test_list)
        val_idx = np.array(val_list) if val_size is not None else None
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        if test_size > 0 and n >= 2:
            n_test = min(n - 1, max(1, int(n * test_size)))
        else:
            n_test = int(n * test_size)
        if val_size is not None:
            if val_size > 0 and (n - n_test) >= 2:
                n_val = min(n - n_test - 1, max(1, int(n * val_size)))
            else:
                n_val = int(n * val_size)
            test_idx = indices[:n_test]
            val_idx = indices[n_test : n_test + n_val]
            train_idx = indices[n_test + n_val :]
        else:
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            val_idx = None

    if is_df:
        X_train, X_test = (X.iloc[train_idx], X.iloc[test_idx])
        X_val = X.iloc[val_idx] if val_idx is not None else None
    else:
        X_train, X_test = (X[train_idx], X[test_idx])
        X_val = X[val_idx] if val_idx is not None else None
    if y is not None:
        if is_series:
            y_train, y_test = (y.iloc[train_idx], y.iloc[test_idx])
            y_val = y.iloc[val_idx] if val_idx is not None else None
        else:
            y_train, y_test = (y[train_idx], y[test_idx])
            y_val = y[val_idx] if val_idx is not None else None
        if val_size is not None:
            return (X_train, X_val, X_test, y_train, y_val, y_test)
        return (X_train, X_test, y_train, y_test)
    if val_size is not None:
        return (X_train, X_val, X_test)
    return (X_train, X_test)


def kfold(
    X: Any,
    y: Any = None,
    k: int = 5,
    stratified: bool = False,
    shuffle: bool = True,
    random_state: int = None,
) -> Generator:
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    if stratified and y is not None:
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        folds: list = [[] for _ in range(k)]
        for c in classes:
            c_idx = np.where(y_arr == c)[0]
            if shuffle:
                np.random.shuffle(c_idx)
            for i, idx in enumerate(c_idx):
                folds[i % k].append(idx)
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        folds = [list(indices[i::k]) for i in range(k)]
    for i in range(k):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([np.array(folds[j]) for j in range(k) if j != i])
        if is_df:
            X_tr, X_va = (X.iloc[train_idx], X.iloc[val_idx])
        else:
            X_tr, X_va = (X[train_idx], X[val_idx])
        if y is not None:
            if is_series:
                y_tr, y_va = (y.iloc[train_idx], y.iloc[val_idx])
            else:
                y_tr, y_va = (y[train_idx], y[val_idx])
            yield (X_tr, X_va, y_tr, y_va)
        else:
            yield (X_tr, X_va)


def time_series_split(X: Any, y: Any = None, n_splits: int = 5) -> Generator:
    n = len(X)
    fold_size = n // (n_splits + 1)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        if is_df:
            X_tr, X_va = (X.iloc[:train_end], X.iloc[val_start:val_end])
        else:
            X_tr, X_va = (X[:train_end], X[val_start:val_end])
        if y is not None:
            if is_series:
                y_tr, y_va = (y.iloc[:train_end], y.iloc[val_start:val_end])
            else:
                y_tr, y_va = (y[:train_end], y[val_start:val_end])
            yield (X_tr, X_va, y_tr, y_va)
        else:
            yield (X_tr, X_va)
