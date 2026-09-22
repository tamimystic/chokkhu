from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _knn_impute_col(
    df: pd.DataFrame, col: str, k: int = 5, fallback_val: object = 0.0
) -> pd.Series:
    s = df[col].copy()
    missing_mask = s.isna().to_numpy()
    if not missing_mask.any():
        return s
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != col]
    if not num_cols:
        med = s.median()
        return s.fillna(fallback_val if pd.isna(med) else med)
    complete_mask = ~df[num_cols].isna().any(axis=1).to_numpy() & ~missing_mask
    if not complete_mask.any():
        med = s.median()
        return s.fillna(fallback_val if pd.isna(med) else med)
    X_complete = np.array(
        df[num_cols].to_numpy(dtype=np.float64)[complete_mask], copy=True
    )
    y_complete = np.array(s.to_numpy(dtype=np.float64)[complete_mask], copy=True)
    s_arr = np.array(s.to_numpy(dtype=np.float64), copy=True)
    X_all = np.array(df[num_cols].to_numpy(dtype=np.float64), copy=True)
    missing_indices = np.where(missing_mask)[0]
    fallback_mean = (
        float(np.nanmean(y_complete))
        if len(y_complete) > 0 and not np.isnan(np.nanmean(y_complete))
        else float(fallback_val if isinstance(fallback_val, (int, float)) else 0.0)
    )
    for idx in missing_indices:
        row_feat = X_all[idx]
        if np.isnan(row_feat).any():
            s_arr[idx] = fallback_mean
            continue
        dists = np.linalg.norm(X_complete - row_feat, axis=1)
        k_nearest = np.argsort(dists)[: min(k, len(dists))]
        weights = 1.0 / (dists[k_nearest] + 1e-08)
        s_arr[idx] = float(np.average(y_complete[k_nearest], weights=weights))
    return pd.Series(s_arr, index=df.index, name=col)


def _iterative_impute(
    df: pd.DataFrame, max_iter: int = 10, fallback_val: object = 0.0
) -> pd.DataFrame:
    df_res = df.copy()
    num_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in num_cols if df_res[c].isna().any()]
    if not missing_cols:
        return df_res
    for c in num_cols:
        med = df_res[c].median()
        df_res[c] = df_res[c].fillna(fallback_val if pd.isna(med) else med)
    for _ in range(max_iter):
        for c in missing_cols:
            orig_missing = df[c].isna().to_numpy()
            predictors = [p for p in num_cols if p != c]
            if not predictors:
                continue
            X_train = df_res.loc[~orig_missing, predictors].to_numpy(dtype=np.float64)
            y_train = df_res.loc[~orig_missing, c].to_numpy(dtype=np.float64)
            X_missing = df_res.loc[orig_missing, predictors].to_numpy(dtype=np.float64)
            if len(X_train) == 0 or len(X_missing) == 0:
                continue
            X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
            X_missing_b = np.column_stack([np.ones(len(X_missing)), X_missing])
            w, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
            df_res.loc[orig_missing, c] = X_missing_b @ w
    return df_res


def handle_missing(
    data: pd.DataFrame,
    strategy: str = "median",
    threshold: float = 0.5,
    fill_value: object = 0,
    knn_k: int = 5,
    interpolate_method: str = "linear",
    interpolate_order: int = 2,
    iterative_max_iter: int = 10,
) -> pd.DataFrame:
    df = data.copy()
    if strategy is None:
        return df
    if strategy == "drop_cols":
        return df.loc[:, df.isna().mean() <= threshold]
    if strategy == "drop_rows":
        return df.dropna()
    if strategy == "iterative":
        return _iterative_impute(
            df, max_iter=iterative_max_iter, fallback_val=fill_value
        )
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in df.columns:
        if not df[col].isna().any():
            continue
        pct = df[col].isna().mean()
        if strategy == "auto":
            if pct > threshold:
                df = df.drop(columns=[col])
                continue
            if col in num_cols:
                series = df[col].dropna()
                is_normal = False
                if len(series) >= 8:
                    try:
                        _, p = stats.shapiro(series[:5000])
                        is_normal = p > 0.05
                    except Exception:
                        is_normal = False
                val = series.mean() if is_normal else series.median()
                if pd.isna(val):
                    val = fill_value
                df[col] = df[col].fillna(val)
            else:
                mode_s = df[col].mode()
                mode_val = (
                    mode_s.iloc[0]
                    if not mode_s.empty and pd.notna(mode_s.iloc[0])
                    else "missing"
                )
                df[col] = df[col].fillna(mode_val)
        elif strategy == "mean" and col in num_cols:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(fill_value if pd.isna(mean_val) else mean_val)
        elif strategy == "median" and col in num_cols:
            med_val = df[col].median()
            df[col] = df[col].fillna(fill_value if pd.isna(med_val) else med_val)
        elif strategy == "mode":
            mode_s = df[col].mode()
            mode_val = (
                mode_s.iloc[0]
                if not mode_s.empty and pd.notna(mode_s.iloc[0])
                else fill_value
            )
            df[col] = df[col].fillna(mode_val)
        elif strategy == "constant":
            df[col] = df[col].fillna(fill_value)
        elif strategy == "ffill":
            df[col] = df[col].ffill().fillna(fill_value)
        elif strategy == "bfill":
            df[col] = df[col].bfill().fillna(fill_value)
        elif strategy == "interpolate" and col in num_cols:
            if interpolate_method in ["polynomial", "spline"]:
                df[col] = (
                    df[col]
                    .interpolate(method=interpolate_method, order=interpolate_order)
                    .fillna(fill_value)
                )
            else:
                df[col] = df[col].interpolate(method="linear").fillna(fill_value)
        elif strategy == "knn" and col in num_cols:
            df[col] = _knn_impute_col(df, col, k=knn_k, fallback_val=fill_value)
        elif col in cat_cols:
            mode_s = df[col].mode()
            mode_val = (
                mode_s.iloc[0]
                if not mode_s.empty and pd.notna(mode_s.iloc[0])
                else "missing"
            )
            df[col] = df[col].fillna(mode_val)
    return df
