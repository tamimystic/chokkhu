from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chokkhu.core import math as sp_stats
from chokkhu.core.visualizer import PlotVisualizer

from .base_plotter import BasePlotter


class UnivariatePlotter(BasePlotter):

    def plot(self):
        univ = self.results.get("univariate", {})
        PlotVisualizer.display_markdown("## Dataset Global Overview")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=missing.index,
                y=missing.values,
                hue=missing.index,
                legend=False,
                palette="Reds",
                ax=ax,
            )
            ax.set_title("Missing Values per Column")
            ax.set_ylabel("Null Count")
            ax.tick_params(axis="x", rotation=90)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, "0_missing_values.png", self.save_dir, self.save_reports
            )
        else:
            print("    [INFO] No missing values found in the dataset.")
        desc = self.df.describe().T.round(2)
        if not desc.empty:
            fig, ax = plt.subplots(figsize=(14, max(4, len(desc) * 0.5)))
            ax.axis("tight")
            ax.axis("off")
            table = ax.table(
                cellText=desc.values,
                rowLabels=desc.index,
                colLabels=desc.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.set_title("Dataset Statistical Summary (df.describe)")
            PlotVisualizer.save_and_show(
                fig, "0_dataset_description.png", self.save_dir, self.save_reports
            )
        PlotVisualizer.display_markdown("### Categorical Data Analysis")
        ordinal = univ.get("ordinal_stats", {})
        if ordinal:
            PlotVisualizer.display_markdown("### Ordinal Features Analysis")
        for col, stats in ordinal.items():
            freq = stats.get("frequencies", {})
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(freq.keys()),
                y=list(freq.values()),
                hue=list(freq.keys()),
                legend=False,
                palette="pastel",
                ax=ax,
            )
            ax.set_title(f"Frequencies in Ordinal Feature: {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, f"1_ordinal_{col}.png", self.save_dir, self.save_reports
            )
        nominal = univ.get("nominal_stats", {})
        if nominal:
            PlotVisualizer.display_markdown("### Nominal Features Analysis")
            for col, stats in nominal.items():
                series = self.df[col].dropna()
                top_15 = series.value_counts().head(15)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    x=top_15.index,
                    y=top_15.values,
                    hue=top_15.index,
                    legend=False,
                    palette="Set3",
                    ax=ax,
                )
                ax.set_title(f"Top 15 Categories in Nominal Feature: {col}")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=90)
                self._add_bar_labels(ax, fmt="%d")
                PlotVisualizer.save_and_show(
                    fig, f"1_nominal_{col}.png", self.save_dir, self.save_reports
                )
        else:
            print(
                "    [INFO] No Nominal features found (no categorical columns with 20-100 unique values)."
            )
        PlotVisualizer.display_markdown("### Numerical Data Analysis")
        discrete = univ.get("discrete_stats", {})
        if discrete:
            PlotVisualizer.display_markdown("### Discrete Features Analysis")
        for col, stats in discrete.items():
            freq = stats.get("frequencies", {})
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(freq.keys()),
                y=list(freq.values()),
                hue=list(freq.keys()),
                legend=False,
                palette="deep",
                ax=ax,
            )
            ax.set_title(f"Distribution of Discrete Feature: {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, f"2_discrete_{col}.png", self.save_dir, self.save_reports
            )
        continuous = univ.get("continuous_stats", {})
        if continuous:
            PlotVisualizer.display_markdown("### Continuous Features Analysis")
        for col, stats in continuous.items():
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0, 0], color="teal")
            axes[0, 0].set_title(f"Distribution of {col}")
            sns.boxplot(y=self.df[col].dropna(), ax=axes[0, 1], color="salmon")
            axes[0, 1].set_title(f"Outliers in {col} (Vertical Boxplot)")
            sns.violinplot(y=self.df[col].dropna(), ax=axes[1, 0], color="skyblue")
            axes[1, 0].set_title(f"Density & Spread in {col} (Violin Plot)")
            sp_stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f"Q-Q Plot for Normal Distribution of {col}")
            plt.tight_layout()
            PlotVisualizer.save_and_show(
                fig, f"2_continuous_{col}.png", self.save_dir, self.save_reports
            )
        PlotVisualizer.display_markdown("### Specialized Data Analysis")
        datetime = univ.get("datetime_stats", {})
        if datetime:
            PlotVisualizer.display_markdown("### DateTime Features Analysis")
        for col, stats in datetime.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            series = pd.to_datetime(self.df[col], errors="coerce").dropna()
            monthly_raw = series.groupby(series.dt.to_period("M")).size()
            monthly_raw.plot(
                ax=ax, marker="o", color="navy", label="Raw Counts", alpha=0.5
            )
            trend = stats.get("monthly_trend", {})
            if trend:
                trend_series = pd.Series(trend)
                trend_series.index = pd.PeriodIndex(trend_series.index, freq="M")
                trend_series.plot(
                    ax=ax, color="red", linewidth=2, label="Trend (3-Month MA)"
                )
            stat_text = (
                "Stationary" if stats.get("pseudo_stationary") else "Non-Stationary"
            )
            ax.set_title(f"Time Series Trend (Monthly) for {col} [{stat_text}]")
            ax.set_ylabel("Count")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, f"3_datetime_{col}.png", self.save_dir, self.save_reports
            )
        text_stats = univ.get("text_stats", {})
        if text_stats:
            PlotVisualizer.display_markdown("### Text Features Analysis (N-Grams)")
        for col, stats in text_stats.items():
            unigrams = stats.get("top_unigrams", {})
            bigrams = stats.get("top_bigrams", {})
            if unigrams:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                sns.barplot(
                    x=list(unigrams.values()),
                    y=list(unigrams.keys()),
                    ax=axes[0],
                    palette="Blues_d",
                )
                axes[0].set_title(f"Top 10 Unigrams: {col}")
                if bigrams:
                    sns.barplot(
                        x=list(bigrams.values()),
                        y=list(bigrams.keys()),
                        ax=axes[1],
                        palette="Greens_d",
                    )
                    axes[1].set_title(f"Top 10 Bigrams: {col}")
                plt.tight_layout()
                PlotVisualizer.save_and_show(
                    fig, f"3_text_{col}_ngrams.png", self.save_dir, self.save_reports
                )
