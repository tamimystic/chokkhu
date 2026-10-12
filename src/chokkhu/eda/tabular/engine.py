from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger
from chokkhu.eda.tabular.bivariate import BivariateAnalyzer
from chokkhu.eda.tabular.global_eda import GlobalAnalyzer
from chokkhu.eda.tabular.multivariate import MultivariateAnalyzer
from chokkhu.eda.tabular.plotter import TabularPlotter
from chokkhu.eda.tabular.univariate import UnivariateAnalyzer


def tabular(
    df: pd.DataFrame = None,
    dataset_path: str = None,
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
    target_col: str = None,
) -> Dict[str, Any]:
    if isinstance(dataset_path, pd.DataFrame):
        df = dataset_path
        dataset_path = None
    if isinstance(df, str):
        dataset_path = df
        df = None
    if df is None and dataset_path is not None:
        if dataset_path.endswith(".csv"):
            import codecs

            try:
                with codecs.open(
                    dataset_path, "r", encoding="utf-8", errors="strict"
                ) as f:
                    f.read(1024)
            except UnicodeDecodeError:
                Logger.warning(
                    f"Encoding issue detected in {dataset_path}. Falling back to 'latin-1' or other standard encoding."
                )
                df = pd.read_csv(dataset_path, encoding="latin-1")
            else:
                df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".xlsx"):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(
                "Unsupported file format. Please provide a DataFrame or CSV/Excel/Parquet path."
            )
    return TabularEDAEngine(df, save_reports, save_dir, target_col).execute()


class TabularEDAEngine:

    def __init__(
        self,
        df: pd.DataFrame,
        save_reports: bool = False,
        save_dir: str = "chokkhu_reports",
        target_col: str = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Dataset must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        self.df = df
        self.save_reports = save_reports
        self.save_dir = save_dir
        self.target_col = target_col
        self.results: Dict[str, Any] = {}

    def execute(self):
        if self.save_reports:
            os.makedirs(self.save_dir, exist_ok=True)
            Logger.info(f"Reports will be saved to {self.save_dir}")
        Logger.info("Executing Phase 0: Global Dataset Profiling...")
        self.results["global_eda"] = GlobalAnalyzer.analyze(self.df)
        self.results["univariate"] = UnivariateAnalyzer.analyze(self.df)
        type_mapping = self.results["univariate"]["type_mapping"]
        from chokkhu.core.visualizer import PlotVisualizer

        md_text = (
            "## Data Type Classification\n"
            "**Categorical Data:**\n"
            f"- Ordinal: {type_mapping['categorical']['ordinal']}\n"
            f"- Nominal: {type_mapping['categorical']['nominal']}\n\n"
            "**Numerical Data:**\n"
            f"- Discrete: {type_mapping['numerical']['discrete']}\n"
            f"- Continuous: {type_mapping['numerical']['continuous']}\n\n"
            "**Specialized Data:**\n"
            f"- DateTime: {type_mapping['specialized']['datetime']}\n"
            f"- Text: {type_mapping['specialized']['text']}\n"
        )
        PlotVisualizer.display_markdown(md_text)
        self.results["bivariate"] = BivariateAnalyzer.analyze(self.df, self.target_col)
        self.results["multivariate"] = MultivariateAnalyzer.analyze(
            self.df, self.target_col
        )
        plotter = TabularPlotter(
            df=self.df,
            results=self.results,
            save_dir=self.save_dir,
            save_reports=self.save_reports,
            target_col=self.target_col,
        )
        plotter.plot_all()
        if self.save_reports:
            self._save_json()
            import os

            from chokkhu.reports.html_builder import HTMLReportBuilder

            html_path = os.path.join(self.save_dir, "chokkhu_report.html")
            HTMLReportBuilder.build(self.save_dir, title="Chokkhu Tabular EDA Report")

            # Display inline if inside a Jupyter Notebook
            try:
                from IPython.display import HTML, display

                with open(html_path, "r", encoding="utf-8") as f:
                    display(HTML(f.read()))
            except ImportError:
                pass

        Logger.info("Tabular EDA Pipeline completed successfully.")
        return self.results

    def _save_json(self):
        json_path = os.path.join(self.save_dir, "eda_summary.json")
        try:
            clean_results = self._sanitize_dict(self.results)
            with open(json_path, "w") as f:
                json.dump(clean_results, f, indent=4)
            Logger.info(f"Summary JSON saved to {json_path}")
        except Exception as e:
            Logger.warning(f"Could not save JSON summary: {e}")

    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                sanitized[k] = self._sanitize_dict(v)
            elif isinstance(v, (pd.DataFrame, pd.Series)):
                sanitized[k] = "Pandas Object Omitted"
            elif isinstance(v, (np.integer, np.floating)):
                sanitized[k] = float(v)
            elif isinstance(v, np.bool_):
                sanitized[k] = bool(v)
            elif isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                try:
                    json.dumps(v)
                    sanitized[k] = v
                except (TypeError, OverflowError):
                    sanitized[k] = str(v)
        return sanitized
