from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chokkhu.core.visualizer import PlotVisualizer

from .base_plotter import BasePlotter


class BivariatePlotter(BasePlotter):

    def plot(self):
        biv = self.results.get("bivariate", {})
        cat_vs_cat = biv.get("cat_vs_cat", {})
        if cat_vs_cat:
            PlotVisualizer.display_markdown("### Categorical vs Categorical Analysis")
        for pair_name, data in cat_vs_cat.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            col1 = data["c1"]
            col2 = data["c2"]
            if col1 in self.df.columns and col2 in self.df.columns:
                sns.countplot(data=self.df, x=col1, hue=col2, ax=ax, palette="Set2")
                ax.set_title(f"{pair_name}")
                ax.tick_params(axis="x", rotation=45)
                self._add_bar_labels(ax, fmt="%d")
                plt.tight_layout()
                PlotVisualizer.save_and_show(
                    fig,
                    f"4_cat_vs_cat_{pair_name}.png",
                    self.save_dir,
                    self.save_reports,
                )
        cat_vs_num = biv.get("cat_vs_num", {})
        if cat_vs_num:
            PlotVisualizer.display_markdown("### Categorical vs Numerical Analysis")
        for pair_name, data in cat_vs_num.items():
            cat = data["cat"]
            num = data["num"]
            fig, ax = plt.subplots(figsize=(10, 6))
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                sns.kdeplot(
                    data=self.df,
                    x=num,
                    hue=cat,
                    fill=False,
                    common_norm=False,
                    palette="tab10",
                    linewidth=2.5,
                    ax=ax,
                )
            ax.set_title(f"{pair_name}")
            PlotVisualizer.save_and_show(
                fig, f"4_cat_vs_num_{pair_name}.png", self.save_dir, self.save_reports
            )
        num_vs_num = biv.get("num_vs_num", {})
        if num_vs_num:
            PlotVisualizer.display_markdown("### Numerical vs Numerical Analysis")
        hue_col = None
        univ = self.results.get("univariate", {})
        if (
            self.target_col
            and self.target_col in self.df.columns
            and (self.df[self.target_col].nunique() < 10)
        ):
            hue_col = self.target_col
        else:
            ordinal_cols = list(univ.get("ordinal_stats", {}).keys())
            if ordinal_cols:
                hue_col = ordinal_cols[0]
        # Sort pairs by absolute correlation and plot top-10 to prevent O(N^2) plot explosion
        sorted_pairs = sorted(
            num_vs_num.items(),
            key=lambda item: abs(item[1].get("pearson", 0.0) or 0.0),
            reverse=True,
        )
        max_biv_plots = min(10, len(sorted_pairs))
        for pair_name, data in sorted_pairs[:max_biv_plots]:
            n1 = data["n1"]
            n2 = data["n2"]
            if hue_col and hue_col not in [n1, n2]:
                g = sns.jointplot(
                    data=self.df,
                    x=n1,
                    y=n2,
                    hue=hue_col,
                    height=8,
                    palette="tab10",
                    alpha=0.7,
                )
            else:
                g = sns.jointplot(
                    data=self.df,
                    x=n1,
                    y=n2,
                    kind="reg",
                    height=8,
                    scatter_kws={"alpha": 0.5, "color": "purple"},
                    line_kws={"color": "red"},
                )
            g.fig.suptitle(f"{pair_name}", y=1.02)
            PlotVisualizer.save_and_show(
                g.fig, f"4_num_vs_num_{pair_name}.png", self.save_dir, self.save_reports
            )
        target_analysis = biv.get("target_analysis", {})
        if target_analysis:
            PlotVisualizer.display_markdown(
                "### Target vs All Features Analysis (IV, WoE, T-Test)"
            )
        iv_res = target_analysis.get("information_value", {})
        if iv_res:
            iv_df = pd.DataFrame(
                list(iv_res.items()), columns=["Feature", "IV"]
            ).sort_values("IV", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=iv_df,
                x="Feature",
                y="IV",
                hue="Feature",
                legend=False,
                palette="viridis",
                ax=ax,
            )
            ax.set_title(
                f"Predictive Power (Information Value) vs Target: {self.target_col}"
            )
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            PlotVisualizer.save_and_show(
                fig, "4_iv_predictive_power.png", self.save_dir, self.save_reports
            )
        anova = target_analysis.get("categorical_vs_target_anova", {})
        if anova:
            features = list(anova.keys())
            p_vals = [anova[f]["anova_p"] for f in features]
            log_p = [-np.log10(p + 1e-09) for p in p_vals]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=features, y=log_p, hue=features, legend=False, palette="magma", ax=ax
            )
            ax.set_title(
                f"ANOVA Significance (-log10 P-Value) vs Target: {self.target_col}"
            )
            ax.set_ylabel("-log10(p-value)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            ax.axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "4_anova_significance.png", self.save_dir, self.save_reports
            )
        t_tests = target_analysis.get("numerical_vs_target_ttest", {})
        if t_tests:
            features = list(t_tests.keys())
            p_vals = [t_tests[f]["p_val"] for f in features]
            log_p = [-np.log10(p + 1e-09) for p in p_vals]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=features,
                y=log_p,
                hue=features,
                legend=False,
                palette="coolwarm",
                ax=ax,
            )
            ax.set_title(
                f"T-Test Significance (-log10 P-Value) vs Target: {self.target_col}"
            )
            ax.set_ylabel("-log10(p-value)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            ax.axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "4_ttest_significance.png", self.save_dir, self.save_reports
            )
