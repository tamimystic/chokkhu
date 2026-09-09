from __future__ import annotations

import numpy as np
import pandas as pd


class TabularStats:

    @staticmethod
    def extract(df: pd.DataFrame) -> dict:
        results = {
            "shape": df.shape,
            "dtypes": df.dtypes,
            "missing": df.isnull().sum(),
            "numerical_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
        }
        if len(results["numerical_cols"]) > 0:
            results["correlation"] = df[results["numerical_cols"]].corr()
        else:
            results["correlation"] = pd.DataFrame()
        return results
