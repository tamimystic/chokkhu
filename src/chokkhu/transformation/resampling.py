from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _pairwise_distances_batched(
    X: np.ndarray, Y: np.ndarray | None = None, batch_size: int = 500
) -> np.ndarray:
    """Computes pairwise Euclidean distance matrix in batches to avoid MemoryError on large datasets."""
    n_x = len(X)
    if Y is None:
        Y = X
    n_y = len(Y)

    dists: np.ndarray = np.empty((n_x, n_y), dtype=np.float64)
    Y_sq = np.sum(Y**2, axis=1)

    for i in range(0, n_x, batch_size):
        end_i = min(i + batch_size, n_x)
        X_batch = X[i:end_i]
        X_sq = np.sum(X_batch**2, axis=1)[:, np.newaxis]
        dot = np.dot(X_batch, Y.T)
        d_sq = np.maximum(X_sq + Y_sq[np.newaxis, :] - 2.0 * dot, 0.0)
        dists[i:end_i] = np.sqrt(d_sq)

    return dists


class RandomOverSampler:
    def __init__(self, ratio: float = 1.0, random_state: int | None = None) -> None:
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        max_count = int(np.max(counts))
        target_count = int(max_count * self.ratio)

        X_res = list(X_arr)
        y_res = list(y_arr)

        for c, count in zip(classes, counts):
            if count >= target_count:
                continue
            n_samples = target_count - count
            c_indices = np.where(y_arr == c)[0]
            sampled_idx = np.random.choice(c_indices, size=n_samples, replace=True)
            X_res.extend(X_arr[sampled_idx])
            y_res.extend(y_arr[sampled_idx])

        X_res_arr = np.array(X_res)
        y_res_arr = np.array(y_res)

        if is_df:
            X_out = pd.DataFrame(X_res_arr, columns=X.columns)
        else:
            X_out = X_res_arr

        if is_series:
            y_out = pd.Series(y_res_arr, name=y.name)
        else:
            y_out = y_res_arr

        return X_out, y_out


class RandomUnderSampler:
    def __init__(self, ratio: float = 1.0, random_state: int | None = None) -> None:
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        min_count = int(np.min(counts))
        target_count = max(1, int(min_count / self.ratio))

        X_res = []
        y_res = []

        for c, count in zip(classes, counts):
            c_indices = np.where(y_arr == c)[0]
            if count > target_count:
                sampled_idx = np.random.choice(
                    c_indices, size=target_count, replace=False
                )
            else:
                sampled_idx = c_indices
            X_res.extend(X_arr[sampled_idx])
            y_res.extend(y_arr[sampled_idx])

        X_res_arr = np.array(X_res)
        y_res_arr = np.array(y_res)

        if is_df:
            X_out = pd.DataFrame(X_res_arr, columns=X.columns)
        else:
            X_out = X_res_arr

        if is_series:
            y_out = pd.Series(y_res_arr, name=y.name)
        else:
            y_out = y_res_arr

        return X_out, y_out


class SMOTE:
    def __init__(
        self, k_neighbors: int = 5, ratio: float = 1.0, random_state: int | None = None
    ) -> None:
        self.k_neighbors = k_neighbors
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        max_count = int(np.max(counts))
        target_count = int(max_count * self.ratio)

        X_res = list(X_arr)
        y_res = list(y_arr)

        for c, count in zip(classes, counts):
            if count >= target_count:
                continue
            n_synthetic = target_count - count
            X_minority = X_arr[y_arr == c]
            n_min = len(X_minority)
            if n_min <= 1:
                continue

            k = min(self.k_neighbors, n_min - 1)

            # Memory-efficient distance matrix (O(N^2) 2D matrix without 3D tensors)
            dists = _pairwise_distances_batched(X_minority)
            np.fill_diagonal(dists, np.inf)

            nearest_indices = np.argsort(dists, axis=1)[:, :k]

            for _ in range(n_synthetic):
                idx = np.random.randint(0, n_min)
                sample = X_minority[idx]
                nn_idx = nearest_indices[idx, np.random.randint(0, k)]
                neighbor = X_minority[nn_idx]
                lam = np.random.uniform(0.0, 1.0)
                synthetic = sample + lam * (neighbor - sample)
                X_res.append(synthetic)
                y_res.append(c)

        X_res_arr = np.array(X_res)
        y_res_arr = np.array(y_res)

        if is_df:
            X_out = pd.DataFrame(X_res_arr, columns=X.columns)
        else:
            X_out = X_res_arr

        if is_series:
            y_out = pd.Series(y_res_arr, name=y.name)
        else:
            y_out = y_res_arr

        return X_out, y_out


class ADASYN:
    def __init__(
        self, k_neighbors: int = 5, ratio: float = 1.0, random_state: int | None = None
    ) -> None:
        self.k_neighbors = k_neighbors
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        majority_idx = int(np.argmax(counts))
        majority_class = classes[majority_idx]
        max_count = int(counts[majority_idx])

        X_res = list(X_arr)
        y_res = list(y_arr)

        for c, count in zip(classes, counts):
            if c == majority_class:
                continue
            G = int((max_count - count) * self.ratio)
            if G <= 0:
                continue

            X_minority = X_arr[y_arr == c]
            if len(X_minority) <= 1:
                continue

            k = min(self.k_neighbors, len(X_arr) - 1)

            # Memory-efficient batched distances
            dists_all = _pairwise_distances_batched(X_minority, X_arr)

            r = np.zeros(len(X_minority))
            for i in range(len(X_minority)):
                nearest_all = np.argsort(dists_all[i])[1 : k + 1]
                maj_count: float = float(np.sum(y_arr[nearest_all] != c))
                r[i] = maj_count / float(k)

            sum_r: float = float(np.sum(r))
            if sum_r == 0:
                r_norm = np.ones(len(X_minority)) / float(len(X_minority))
            else:
                r_norm = r / sum_r

            g_samples = np.random.multinomial(G, r_norm)

            dists_min = _pairwise_distances_batched(X_minority)
            np.fill_diagonal(dists_min, np.inf)
            k_min = min(self.k_neighbors, len(X_minority) - 1)
            nearest_min = np.argsort(dists_min, axis=1)[:, :k_min]

            for i, n_gen in enumerate(g_samples):
                sample = X_minority[i]
                for _ in range(n_gen):
                    nn_idx = nearest_min[i, np.random.randint(0, k_min)]
                    neighbor = X_minority[nn_idx]
                    lam = np.random.uniform(0.0, 1.0)
                    synthetic = sample + lam * (neighbor - sample)
                    X_res.append(synthetic)
                    y_res.append(c)

        X_res_arr = np.array(X_res)
        y_res_arr = np.array(y_res)

        if is_df:
            X_out = pd.DataFrame(X_res_arr, columns=X.columns)
        else:
            X_out = X_res_arr

        if is_series:
            y_out = pd.Series(y_res_arr, name=y.name)
        else:
            y_out = y_res_arr

        return X_out, y_out


class SMOTETomek:
    def __init__(
        self, k_neighbors: int = 5, ratio: float = 1.0, random_state: int | None = None
    ) -> None:
        self.smote = SMOTE(
            k_neighbors=k_neighbors, ratio=ratio, random_state=random_state
        )

    def fit_resample(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_sm, y_sm = self.smote.fit_resample(X, y)
        X_arr = np.asarray(X_sm, dtype=np.float64)
        y_arr = np.asarray(y_sm)

        n = len(X_arr)
        if n < 2:
            return X_sm, y_sm

        dists = _pairwise_distances_batched(X_arr)
        np.fill_diagonal(dists, np.inf)

        nearest_1 = np.argmin(dists, axis=1)

        tomek_mask: np.ndarray = np.ones(n, dtype=bool)
        for i in range(n):
            j = nearest_1[i]
            if nearest_1[j] == i and y_arr[i] != y_arr[j]:
                tomek_mask[i] = False
                tomek_mask[j] = False

        X_out_arr = X_arr[tomek_mask]
        y_out_arr = y_arr[tomek_mask]

        if is_df:
            X_out = pd.DataFrame(X_out_arr, columns=X.columns)
        else:
            X_out = X_out_arr

        if is_series:
            y_out = pd.Series(y_out_arr, name=y.name)
        else:
            y_out = y_out_arr

        return X_out, y_out
