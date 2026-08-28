from __future__ import annotations

__version__ = "0.8.0"
__author__ = "tamimystic"

from .cleaning import clean
from .eda import ImageEDA, image
from .eda import tabular as tabular_fn
from .io import load, save
from .preprocessing import (
    PowerScaler,
    QuantileScaler,
    RFESelector,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    preprocess,
)
from .splitting import split
from .transformation import (
    LDA,
    PCA,
    SMOTE,
    TSNE,
    ImageAugmenter,
    LinearDiscriminantAnalysis,
    PolynomialFeatures,
    transform,
)
from .models import NeuralNetwork, train
from .evaluation import (
    evaluate,
    accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    pr_auc_score,
)
from .explainability import ExplanationResult, explain
from .pipeline import PipelineResult, TransformationState, pipeline


class EDAWrapper:
    @staticmethod
    def image(dataset_path: str, save_reports: bool = False, save_dir: str = None):
        if save_reports and save_dir is None:
            save_dir = "chokkhu_outputs/image_reports"
        return ImageEDA(
            dataset_path=dataset_path, save_reports=save_reports, save_dir=save_dir
        )

    @staticmethod
    def tabular(
        dataset_path: str,
        save_reports: bool = False,
        save_dir: str = None,
        target_col: str = None,
    ):
        if save_reports and save_dir is None:
            save_dir = "chokkhu_outputs/tabular_reports"
        return tabular_fn(
            dataset_path=dataset_path,
            target_col=target_col,
            save_reports=save_reports,
            save_dir=save_dir,
        )


eda = EDAWrapper()

__all__ = [
    "eda",
    "load",
    "save",
    "clean",
    "preprocess",
    "transform",
    "split",
    "PCA",
    "LDA",
    "LinearDiscriminantAnalysis",
    "TSNE",
    "SMOTE",
    "ImageAugmenter",
    "PolynomialFeatures",
    "train",
    "evaluate",
    "explain",
    "ExplanationResult",
    "pipeline",
    "PipelineResult",
    "TransformationState",
    "NeuralNetwork",
    "PowerScaler",
    "QuantileScaler",
    "RFESelector",
    "accuracy_score",
    "roc_auc_score",
    "pr_auc_score",
    "log_loss",
    "mean_squared_error",
    "r2_score",
]
