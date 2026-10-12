from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger, get_progress_bar
from chokkhu.core.visualizer import PlotVisualizer

from .color import ColorAnalyzer
from .duplicates import DuplicateDetector
from .metadata import MetadataExtractor
from .objects import ObjectDetector
from .plotter import ImagePlotter
from .quality import QualityAnalyzer
from .texture import TextureAnalyzer


class ImageEDA:

    def __init__(
        self,
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports",
    ):
        self.dataset_path = dataset_path
        self.save_reports = save_reports
        self.save_dir = save_dir
        self.results: dict = {}
        self.class_paths: list = []
        if self.save_reports:
            os.makedirs(self.save_dir, exist_ok=True)
        PlotVisualizer.setup_theme()
        self._perform_eda()

    def _collect_paths(self):
        for root, _, files in os.walk(self.dataset_path):
            if any(
                (f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) for f in files)
            ):
                self.class_paths.append(root)

    def _perform_eda(self):
        Logger.info(f"Executing Image EDA for: {self.dataset_path}")
        self._collect_paths()
        if not self.class_paths:
            Logger.error("No valid images found in the specified path.")
            return
        self.results = self._analyze_data()
        plotter = ImagePlotter(
            self.results["df_metrics"], self.results, self.save_dir, self.save_reports
        )
        Logger.info("Rendering High-Quality Reports...")
        plotter.plot_all()
        if self.save_reports:
            csv_path = os.path.join(self.save_dir, "image_metrics.csv")
            self.results["df_metrics"].to_csv(csv_path, index=False)
            from chokkhu.reports.html_builder import HTMLReportBuilder

            html_path = os.path.join(self.save_dir, "chokkhu_report.html")
            HTMLReportBuilder.build(self.save_dir, title="Chokkhu Image EDA Report")

            # Display inline if inside a Jupyter Notebook
            try:
                from IPython.display import HTML, display

                with open(html_path, "r", encoding="utf-8") as f:
                    display(HTML(f.read()))
            except ImportError:
                pass

            Logger.info(f"Image EDA Complete! All reports saved at: {self.save_dir}")

    def _analyze_data(self) -> dict:
        exts = (".png", ".jpg", ".jpeg", ".webp")
        metrics_list = []
        total_rgb_hist = np.zeros((256, 3))
        processed_count = 0
        avg_images = {}
        for path in self.class_paths:
            class_name = os.path.basename(path)
            Logger.info(f"Running Analysis for Class: {class_name}")
            files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
            sum_img = None
            count_img = 0
            for img_name in get_progress_bar(files, desc=f"Analyzing {class_name}"):
                img_path = os.path.join(path, img_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                resized_for_avg = cv2.resize(img_rgb, (128, 128)).astype(np.float64)
                if sum_img is None:
                    sum_img = resized_for_avg
                else:
                    sum_img += resized_for_avg
                count_img += 1
                meta_features = MetadataExtractor.extract(img_path, img_bgr)
                color_features = ColorAnalyzer.extract(gray)
                texture_features = TextureAnalyzer.extract(gray)
                quality_features = QualityAnalyzer.extract(
                    gray, color_features["Brightness"], color_features["Contrast"]
                )
                duplicate_features = DuplicateDetector.extract(gray)
                object_features = ObjectDetector.extract(gray)
                combined = {
                    "Class": class_name,
                    "Image": img_name,
                    **meta_features,
                    **color_features,
                    **texture_features,
                    **quality_features,
                    **duplicate_features,
                    **object_features,
                }
                metrics_list.append(combined)
                for i in range(3):
                    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                    total_rgb_hist[:, i] += hist.flatten()
                processed_count += 1
            if count_img > 0:
                avg_images[class_name] = (sum_img / count_img).astype(np.uint8)
        df_metrics = pd.DataFrame(metrics_list)
        avg_hist = (
            total_rgb_hist / processed_count if processed_count > 0 else total_rgb_hist
        )
        return {
            "df_metrics": df_metrics,
            "avg_rgb_hist": avg_hist,
            "avg_images": avg_images,
        }
