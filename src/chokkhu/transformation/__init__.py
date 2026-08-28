from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger

from .augmentation import ImageAugmenter
from .features import BinningTransformer, LogTransformer, PolynomialFeatures
from .lda import LinearDiscriminantAnalysis
from .pca import PCA, TruncatedSVD
from .resampling import (
    ADASYN,
    SMOTE,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTETomek,
)
from .tsne import TSNE

LDA = LinearDiscriminantAnalysis


def transform(
    data: pd.DataFrame | str | dict[str, Any],
    target: str | None = None,
    pca: int | None = None,
    pca_variance: float | None = None,
    pca_whiten: bool = False,
    svd: int | None = None,
    lda: int | None = None,
    tsne: int | None = None,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float = 200.0,
    tsne_n_iter: int = 500,
    resample: str | None = None,
    resample_ratio: float = 1.0,
    smote_k: int = 5,
    random_state: int | None = None,
    augment: bool = False,
    augment_techniques: list[str] | None = None,
    augment_factor: int = 1,
    rotate_range: tuple[int, int] = (-30, 30),
    brightness_range: tuple[float, float] = (0.7, 1.3),
    noise_std: float = 0.05,
    polynomial: int | None = None,
    interaction_only: bool = False,
    include_bias: bool = False,
    log_features: list[str] | None = None,
    binning_bins: int | None = None,
    binning_strategy: str = "uniform",
    binning_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame | dict[str, Any] | tuple[pd.DataFrame, pd.Series]:
    if isinstance(data, dict) and ("images" in data or "image" in data):
        img_list = data.get("images", data.get("image", []))
        lbl_list = data.get("labels", None)
        if augment:
            augmenter = ImageAugmenter(
                techniques=augment_techniques,
                augment_factor=augment_factor,
                rotate_range=rotate_range,
                brightness_range=brightness_range,
                noise_std=noise_std,
                random_state=random_state,
            )
            aug_imgs, aug_lbls = augmenter.augment_dataset(img_list, lbl_list)
            res_dict = dict(data)
            res_dict["images"] = aug_imgs
            if aug_lbls is not None:
                res_dict["labels"] = aug_lbls
            if verbose:
                Logger.info(
                    f"Augmented image dataset: {len(img_list)} -> {len(aug_imgs)} images"
                )
            return res_dict
        return data

    if isinstance(data, str):
        from chokkhu.io import load

        data = load(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame or image dictionary.")

    df = data.copy()
    initial_shape = df.shape

    if log_features is not None:
        df = LogTransformer(columns=log_features).fit_transform(df)

    if binning_bins is not None:
        df = BinningTransformer(
            n_bins=binning_bins,
            strategy=binning_strategy,
            columns=binning_columns,
        ).fit_transform(df)

    if polynomial is not None and polynomial >= 2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target is not None and target in num_cols:
            num_cols.remove(target)
        if len(num_cols) > 0:
            poly = PolynomialFeatures(
                degree=polynomial,
                interaction_only=interaction_only,
                include_bias=include_bias,
            )
            poly_arr = poly.fit_transform(df[num_cols].to_numpy())
            poly_cols = [f"poly_{i}" for i in range(poly_arr.shape[1])]
            poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)
            non_num_cols = [c for c in df.columns if c not in num_cols]
            df = pd.concat([df[non_num_cols], poly_df], axis=1)

    if resample is not None and target is not None and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[num_cols].to_numpy(dtype=np.float64)

        sampler: Any = None
        if resample == "smote":
            sampler = SMOTE(
                k_neighbors=smote_k,
                ratio=resample_ratio,
                random_state=random_state,
            )
            X_res, y_res = sampler.fit_resample(X_num, y)
        elif resample == "adasyn":
            sampler = ADASYN(
                k_neighbors=smote_k,
                ratio=resample_ratio,
                random_state=random_state,
            )
            X_res, y_res = sampler.fit_resample(X_num, y)
        elif resample == "random_oversample":
            sampler = RandomOverSampler(ratio=resample_ratio, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_num, y)
        elif resample == "random_undersample":
            sampler = RandomUnderSampler(
                ratio=resample_ratio, random_state=random_state
            )
            X_res, y_res = sampler.fit_resample(X_num, y)
        elif resample == "smote_tomek":
            sampler = SMOTETomek(
                k_neighbors=smote_k,
                ratio=resample_ratio,
                random_state=random_state,
            )
            X_res, y_res = sampler.fit_resample(X_num, y)
        else:
            X_res, y_res = X_num, y.to_numpy()

        res_df = pd.DataFrame(X_res, columns=num_cols)
        res_df[target] = y_res
        df = res_df

    if pca is not None or pca_variance is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target is not None and target in num_cols:
            num_cols.remove(target)
        if len(num_cols) > 0:
            pca_model = PCA(
                n_components=pca,
                variance_ratio=pca_variance,
                whiten=pca_whiten,
            )
            pca_arr = pca_model.fit_transform(df[num_cols].to_numpy())
            pca_cols = [f"pca_{i}" for i in range(pca_arr.shape[1])]
            pca_df = pd.DataFrame(pca_arr, columns=pca_cols, index=df.index)
            non_num_cols = [c for c in df.columns if c not in num_cols]
            df = pd.concat([df[non_num_cols], pca_df], axis=1)

    if svd is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target is not None and target in num_cols:
            num_cols.remove(target)
        if len(num_cols) > 0:
            svd_model = TruncatedSVD(n_components=svd)
            svd_arr = svd_model.fit_transform(df[num_cols].to_numpy())
            svd_cols = [f"svd_{i}" for i in range(svd_arr.shape[1])]
            svd_df = pd.DataFrame(svd_arr, columns=svd_cols, index=df.index)
            non_num_cols = [c for c in df.columns if c not in num_cols]
            df = pd.concat([df[non_num_cols], svd_df], axis=1)

    if lda is not None and target is not None and target in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        if len(num_cols) > 0:
            lda_model = LinearDiscriminantAnalysis(n_components=lda)
            lda_arr = lda_model.fit_transform(
                df[num_cols].to_numpy(), df[target].to_numpy()
            )
            lda_cols = [f"lda_{i}" for i in range(lda_arr.shape[1])]
            lda_df = pd.DataFrame(lda_arr, columns=lda_cols, index=df.index)
            non_num_cols = [c for c in df.columns if c not in num_cols]
            df = pd.concat([df[non_num_cols], lda_df], axis=1)

    if tsne is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target is not None and target in num_cols:
            num_cols.remove(target)
        if len(num_cols) > 0:
            tsne_model = TSNE(
                n_components=tsne,
                perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate,
                n_iter=tsne_n_iter,
                random_state=random_state,
            )
            tsne_arr = tsne_model.fit_transform(df[num_cols].to_numpy())
            tsne_cols = [f"tsne_{i}" for i in range(tsne_arr.shape[1])]
            tsne_df = pd.DataFrame(tsne_arr, columns=tsne_cols, index=df.index)
            non_num_cols = [c for c in df.columns if c not in num_cols]
            df = pd.concat([df[non_num_cols], tsne_df], axis=1)

    if verbose:
        Logger.info(f"Transformed dataset: {initial_shape} -> {df.shape}")

    return df


__all__ = [
    "transform",
    "PCA",
    "TruncatedSVD",
    "LinearDiscriminantAnalysis",
    "LDA",
    "TSNE",
    "SMOTE",
    "ADASYN",
    "RandomOverSampler",
    "RandomUnderSampler",
    "SMOTETomek",
    "ImageAugmenter",
    "PolynomialFeatures",
    "LogTransformer",
    "BinningTransformer",
]
