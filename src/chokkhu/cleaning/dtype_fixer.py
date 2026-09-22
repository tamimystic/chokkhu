from __future__ import annotations

import numpy as np
import pandas as pd


def fix_dtypes(
    data: pd.DataFrame, category_threshold: int = 20, date_formats: list = None
) -> pd.DataFrame:
    df = data.copy()
    for col in df.columns:
        if (
            df[col].dtype == object
            or pd.api.types.is_string_dtype(df[col].dtype)
            or pd.api.types.is_object_dtype(df[col].dtype)
        ):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            numeric_s = pd.to_numeric(s, errors="coerce")
            if numeric_s.notna().mean() > 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                continue
            try:
                date_s = pd.to_datetime(s, errors="coerce", format="mixed")
            except Exception:
                date_s = pd.to_datetime(s, errors="coerce")
            if date_s.notna().mean() > 0.8:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                except Exception:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                continue
            bool_map = {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
                "1": True,
                "0": False,
                "y": True,
                "n": False,
            }
            lower_s = s.astype(str).str.lower().str.strip()
            if len(lower_s) > 0 and lower_s.isin(bool_map.keys()).all():
                lower_full = df[col].astype(str).str.lower().str.strip()
                mapped = lower_full.map(bool_map)
                if df[col].isna().any():
                    mapped.loc[df[col].isna()] = np.nan
                    try:
                        df[col] = mapped.astype("boolean")
                    except Exception:
                        df[col] = mapped
                else:
                    df[col] = mapped.astype(bool)
                continue
            if df[col].nunique() <= category_threshold:
                df[col] = df[col].astype("category")
    return df
