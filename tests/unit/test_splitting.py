from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chokkhu
from chokkhu.splitting import kfold, time_series_split, train_test_split


def test_train_test_split_basic():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_tr) == 40
    assert len(X_te) == 10
    assert len(y_tr) == 40
    assert len(y_te) == 10


def test_train_val_test_split():
    df = pd.DataFrame({"feat": range(100), "target": [0, 1] * 50})
    X_tr, X_va, X_te, y_tr, y_va, y_te = chokkhu.split(
        df, target="target", test_size=0.2, val_size=0.1, stratify=True, random_state=42
    )
    assert len(X_tr) == 70
    assert len(X_va) == 10
    assert len(X_te) == 20
    assert y_tr.mean() == pytest.approx(0.5, abs=0.1)
    assert y_te.mean() == pytest.approx(0.5, abs=0.1)


def test_kfold():
    X = np.arange(20)
    folds = list(kfold(X, k=4, shuffle=False))
    assert len(folds) == 4
    for tr, val in folds:
        assert len(tr) == 15
        assert len(val) == 5


def test_time_series_split():
    X = np.arange(30)
    splits = list(time_series_split(X, n_splits=3))
    assert len(splits) == 3
    for tr, val in splits:
        assert len(tr) > 0
        assert len(val) > 0
        assert tr[-1] < val[0]
