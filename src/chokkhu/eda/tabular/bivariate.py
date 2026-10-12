from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from chokkhu.core import math as stats
from chokkhu.eda.tabular.univariate import UnivariateAnalyzer


class BivariateAnalyzer:

    @staticmethod
    def calculate_iv_woe(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
        df_temp = df[[feature, target]].dropna()
        if df_temp.empty or df_temp[target].nunique() != 2:
            return {}
        t_max = df_temp[target].max()
        df_temp["target_bin"] = np.where(df_temp[target] == t_max, 1, 0)
        if (
            pd.api.types.is_numeric_dtype(df_temp[feature])
            and df_temp[feature].nunique() > 10
        ):
            df_temp["feature_bin"] = pd.qcut(
                df_temp[feature], q=10, duplicates="drop"
            ).astype(str)
        else:
            df_temp["feature_bin"] = df_temp[feature].astype(str)
        grouped = df_temp.groupby("feature_bin")["target_bin"].agg(["count", "sum"])
        grouped["events"] = grouped["sum"]
        grouped["non_events"] = grouped["count"] - grouped["sum"]
        total_events = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()
        if total_events == 0 or total_non_events == 0:
            return {}
        grouped["event_rate"] = grouped["events"] / total_events
        grouped["non_event_rate"] = grouped["non_events"] / total_non_events
        grouped["event_rate"] = grouped["event_rate"].replace(0, 0.0001)
        grouped["non_event_rate"] = grouped["non_event_rate"].replace(0, 0.0001)
        grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
        grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped[
            "woe"
        ]
        return {"iv": grouped["iv"].sum(), "woe_dict": grouped["woe"].to_dict()}

    @staticmethod
    def analyze(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        types = UnivariateAnalyzer.infer_data_types(df)
        categorical = types["categorical"]["ordinal"] + types["categorical"]["nominal"]
        numerical = types["numerical"]["discrete"] + types["numerical"]["continuous"]
        cat_vs_cat = {}
        if len(categorical) >= 2:
            for i in range(len(categorical)):
                for j in range(i + 1, len(categorical)):
                    c1, c2 = (categorical[i], categorical[j])
                    try:
                        crosstab = pd.crosstab(df[c1], df[c2])
                        if not crosstab.empty:
                            cat_vs_cat[f"{c1}_vs_{c2}"] = {
                                "c1": c1,
                                "c2": c2,
                                "crosstab": crosstab,
                            }
                    except Exception:
                        pass
        results["cat_vs_cat"] = cat_vs_cat
        cat_vs_num = {}
        if len(categorical) > 0 and len(numerical) > 0:
            for cat in categorical:
                for num in numerical:
                    groups = [
                        group[num].dropna().values
                        for name, group in df.groupby(cat)
                        if len(group) > 5
                    ]
                    if len(groups) >= 2:
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            cat_vs_num[f"{cat}_vs_{num}"] = {
                                "cat": cat,
                                "num": num,
                                "anova_p": p_val,
                            }
                        except Exception:
                            pass
        results["cat_vs_num"] = cat_vs_num
        num_vs_num = {}
        if len(numerical) >= 2:
            for i in range(len(numerical)):
                for j in range(i + 1, len(numerical)):
                    n1, n2 = (numerical[i], numerical[j])
                    try:
                        corr = df[[n1, n2]].corr().iloc[0, 1]
                        num_vs_num[f"{n1}_vs_{n2}"] = {
                            "n1": n1,
                            "n2": n2,
                            "pearson": corr,
                        }
                    except Exception:
                        pass
        results["num_vs_num"] = num_vs_num
        if target_col and target_col in df.columns:
            target_analysis = {}
            if df[target_col].nunique() == 2:
                iv_results = {}
                for col in df.columns:
                    if col == target_col:
                        continue
                    iv_data = BivariateAnalyzer.calculate_iv_woe(df, col, target_col)
                    if iv_data and "iv" in iv_data:
                        iv_results[col] = iv_data["iv"]
                target_analysis["information_value"] = iv_results
            if target_col in numerical:
                anova_res = {}
                for cat_col in categorical:
                    groups = [
                        group[target_col].dropna().values
                        for name, group in df.groupby(cat_col)
                        if len(group) > 5
                    ]
                    if len(groups) >= 2:
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            h_stat, kp_val = stats.kruskal(*groups)
                            anova_res[cat_col] = {"anova_p": p_val, "kruskal_p": kp_val}
                        except Exception:
                            pass
                target_analysis["categorical_vs_target_anova"] = anova_res
            if df[target_col].nunique() == 2:
                t_tests = {}
                for num_col in numerical:
                    if num_col == target_col:
                        continue
                    try:
                        t_stat, p_val = stats.ttest_ind(
                            df[df[target_col] == df[target_col].unique()[0]][
                                num_col
                            ].dropna(),
                            df[df[target_col] == df[target_col].unique()[1]][
                                num_col
                            ].dropna(),
                        )
                        t_tests[num_col] = {"t_stat": t_stat, "p_val": p_val}
                    except Exception:
                        pass
                target_analysis["numerical_vs_target_ttest"] = t_tests
            results["target_analysis"] = target_analysis
        else:
            results["target_analysis"] = {}
        return results
