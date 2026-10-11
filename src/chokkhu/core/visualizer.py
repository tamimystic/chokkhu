from __future__ import annotations

import os

import matplotlib.pyplot as plt


class PlotVisualizer:

    @staticmethod
    def setup_theme() -> None:
        plt.switch_backend("Agg")
        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid")
        except (ImportError, Exception):
            pass
        plt.rcParams.update(
            {
                "axes.grid": True,
                "grid.color": "#e0e0e0",
                "grid.linestyle": "--",
                "font.size": 12,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
            }
        )

    @staticmethod
    def save_and_show(fig, filename: str, save_dir: str, save_reports: bool) -> None:
        plt.tight_layout()
        if save_reports:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            pass

        plt.close(fig)

    @staticmethod
    def display_markdown(text: str) -> None:
        try:
            from IPython.display import Markdown, display

            display(Markdown(text))
        except ImportError:
            print(f"\n{text.replace('#', '').strip()}\n")
