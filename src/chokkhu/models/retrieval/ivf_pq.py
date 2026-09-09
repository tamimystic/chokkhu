"""Inverted File with Product Quantization (IVF-PQ) Index.

Pure NumPy implementation of IVF-PQ for memory-compact and ultra-fast
approximate nearest neighbor search with Asymmetric Distance Computation (ADC).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class IVFPQIndex:
    """Inverted File with Product Quantization (IVF-PQ) Index.

    Parameters
    ----------
    dim : int
        Vector dimensionality. Must be divisible by n_subvectors (m).
    n_lists : int, default=32
        Number of coarse Voronoi clusters (inverted lists).
    n_subvectors : int, default=8
        Number of sub-vectors for Product Quantization. dim must be divisible by n_subvectors.
    n_bits : int, default=8
        Number of bits per sub-quantizer. Typically 8 (yielding 2^8 = 256 centroids).
    metric : str, default='euclidean'
        Distance metric: 'euclidean' or 'dot'.
    seed : Optional[int], default=42
        Random seed for K-Means cluster initialization.
    """

    def __init__(
        self,
        dim: int,
        n_lists: int = 32,
        n_subvectors: int = 8,
        n_bits: int = 8,
        metric: str = "euclidean",
        seed: Optional[int] = 42,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % n_subvectors != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by n_subvectors ({n_subvectors})"
            )
        if n_bits > 16:
            raise ValueError(f"n_bits ({n_bits}) too large; maximum supported is 16")

        self.dim = dim
        self.n_lists = n_lists
        self.m = n_subvectors
        self.d_sub = dim // n_subvectors
        self.k_sub = 1 << n_bits
        self.metric = metric.lower()
        self.rng = np.random.RandomState(seed)

        # Coarse Quantizer (Voronoi centroids): shape (n_lists, dim)
        self.coarse_centroids: Optional[np.ndarray] = None

        # PQ Codebooks: shape (m, k_sub, d_sub)
        self.codebooks: Optional[np.ndarray] = None

        # Inverted Lists: inverted_lists[list_id] = list of encoded bytecodes (N_c, m)
        self.inverted_lists: Dict[int, List[np.ndarray]] = {}
        # Inverted IDs: inverted_ids[list_id] = list of user IDs
        self.inverted_ids: Dict[int, List[Union[int, str]]] = {}

        self.is_trained: bool = False
        self.total_vectors: int = 0

    def _simple_kmeans(
        self, X: np.ndarray, k: int, n_iter: int = 20
    ) -> np.ndarray:
        """Lightweight Lloyd's K-Means clustering in pure NumPy."""
        n_samples = X.shape[0]
        if n_samples <= k:
            centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
            centroids[:n_samples] = X
            return centroids

        # K-Means++ initialization
        init_idx = [int(self.rng.randint(0, n_samples))]
        for _ in range(1, k):
            dist_sq = np.min(
                np.sum((X[:, None, :] - X[init_idx, :]) ** 2, axis=2), axis=1
            )
            dist_sq = np.maximum(0.0, dist_sq)
            prob = dist_sq / max(1e-12, np.sum(dist_sq))
            next_idx = int(self.rng.choice(n_samples, p=prob))
            init_idx.append(next_idx)

        centroids = X[init_idx].astype(np.float32)

        for _ in range(n_iter):
            # Assign points to nearest centroid
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[j] = X[self.rng.randint(0, n_samples)]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return centroids

    def train(self, X: Union[np.ndarray, List[List[float]]], n_iter: int = 25) -> "IVFPQIndex":
        """Train the coarse quantizer and product quantizer codebooks on representative vectors.

        Parameters
        ----------
        X : np.ndarray of shape (N, dim)
            Training vectors.
        n_iter : int, default=25
            Number of K-Means iterations.
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Feature dim {arr.shape[1]} does not match index dim {self.dim}")

        n_samples = arr.shape[0]
        n_clusters = min(self.n_lists, n_samples)
        self.n_lists = n_clusters

        # 1. Train Coarse Voronoi Centroids
        self.coarse_centroids = self._simple_kmeans(arr, n_clusters, n_iter=n_iter)

        # 2. Compute Residuals
        dists = np.sum(
            (arr[:, None, :] - self.coarse_centroids[None, :, :]) ** 2, axis=2
        )
        coarse_labels = np.argmin(dists, axis=1)
        residuals = arr - self.coarse_centroids[coarse_labels]

        # 3. Train Sub-space PQ Codebooks on Residuals
        self.codebooks = np.zeros((self.m, self.k_sub, self.d_sub), dtype=np.float32)
        for sub_i in range(self.m):
            start_col = sub_i * self.d_sub
            end_col = start_col + self.d_sub
            sub_residuals = residuals[:, start_col:end_col]
            k_sub_actual = min(self.k_sub, n_samples)
            cb = self._simple_kmeans(sub_residuals, k_sub_actual, n_iter=n_iter)
            if k_sub_actual < self.k_sub:
                self.codebooks[sub_i, :k_sub_actual] = cb
            else:
                self.codebooks[sub_i] = cb

        # Initialize inverted lists
        for c_id in range(self.n_lists):
            self.inverted_lists[c_id] = []
            self.inverted_ids[c_id] = []

        self.is_trained = True
        return self

    def _quantize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Quantize residuals into m sub-vector code indices."""
        n = residuals.shape[0]
        dtype = np.uint8 if self.k_sub <= 256 else np.uint16
        codes = np.zeros((n, self.m), dtype=dtype)

        for sub_i in range(self.m):
            start_col = sub_i * self.d_sub
            end_col = start_col + self.d_sub
            sub_res = residuals[:, start_col:end_col]
            cb = self.codebooks[sub_i]  # shape (k_sub, d_sub)

            # Distances from sub_res (N, d_sub) to cb (k_sub, d_sub)
            sub_dists = np.sum(
                (sub_res[:, None, :] - cb[None, :, :]) ** 2, axis=2
            )
            codes[:, sub_i] = np.argmin(sub_dists, axis=1)

        return codes

    def add(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        ids: Optional[List[Union[int, str]]] = None,
    ) -> None:
        """Encode and insert vectors into the IVF-PQ index.

        Parameters
        ----------
        vectors : np.ndarray of shape (N, dim)
            Feature vectors.
        ids : Optional[list of ID], default=None
            Identifiers for the vectors.
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors. Call .train(X) first.")

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.dim:
            raise ValueError(f"Vector dim {arr.shape[1]} does not match index dim {self.dim}")

        n_samples = arr.shape[0]
        if ids is None:
            curr_len = self.total_vectors
            assigned_ids = list(range(curr_len, curr_len + n_samples))
        else:
            if len(ids) != n_samples:
                raise ValueError(f"Length of ids ({len(ids)}) must match vectors ({n_samples})")
            assigned_ids = ids

        # 1. Assign to Coarse Centroids
        dists = np.sum(
            (arr[:, None, :] - self.coarse_centroids[None, :, :]) ** 2, axis=2
        )
        coarse_labels = np.argmin(dists, axis=1)
        residuals = arr - self.coarse_centroids[coarse_labels]

        # 2. Product Quantize Residuals
        codes = self._quantize_residuals(residuals)

        # 3. Store in Inverted Lists
        for i in range(n_samples):
            c_label = int(coarse_labels[i])
            self.inverted_lists[c_label].append(codes[i])
            self.inverted_ids[c_label].append(assigned_ids[i])

        self.total_vectors += n_samples

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        k: int = 10,
        n_probe: int = 4,
    ) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """Search for top-k approximate nearest neighbors using Asymmetric Distance Computation (ADC).

        Parameters
        ----------
        query : np.ndarray of shape (dim,)
            Query vector.
        k : int, default=10
            Number of top nearest neighbors.
        n_probe : int, default=4
            Number of closest Voronoi cells to inspect.

        Returns
        -------
        distances : np.ndarray of shape (k,)
            Approximate squared L2 distances.
        matched_ids : list of IDs
            IDs of the nearest neighbors.
        """
        if not self.is_trained or self.total_vectors == 0:
            return np.empty(0, dtype=np.float32), []

        q_vec = np.asarray(query, dtype=np.float32).ravel()
        if q_vec.shape[0] != self.dim:
            raise ValueError(f"Query dim {q_vec.shape[0]} does not match index dim {self.dim}")

        n_probe = min(max(1, n_probe), self.n_lists)

        # 1. Find top n_probe coarse centroids
        coarse_dists = np.sum((self.coarse_centroids - q_vec) ** 2, axis=1)
        probe_lists = np.argsort(coarse_dists)[:n_probe]

        candidate_dists: List[float] = []
        candidate_ids: List[Union[int, str]] = []

        for list_id in probe_lists:
            codes_list = self.inverted_lists[list_id]
            if not codes_list:
                continue

            ids_list = self.inverted_ids[list_id]
            coarse_centroid = self.coarse_centroids[list_id]
            query_residual = q_vec - coarse_centroid

            # Precompute ADC Lookup Table for this probed list: shape (m, k_sub)
            lut = np.zeros((self.m, self.k_sub), dtype=np.float32)
            for sub_i in range(self.m):
                start_col = sub_i * self.d_sub
                end_col = start_col + self.d_sub
                q_sub = query_residual[start_col:end_col]
                cb = self.codebooks[sub_i]
                lut[sub_i] = np.sum((cb - q_sub) ** 2, axis=1)

            codes_arr = np.array(codes_list)  # (N_c, m)
            # Vectorized ADC lookup: sum over m sub-spaces
            sub_indices = np.arange(self.m)
            approx_dists = np.sum(lut[sub_indices, codes_arr], axis=1)

            candidate_dists.extend(approx_dists.tolist())
            candidate_ids.extend(ids_list)

        if not candidate_dists:
            return np.empty(0, dtype=np.float32), []

        cand_dists_arr = np.array(candidate_dists, dtype=np.float32)
        top_k_indices = np.argsort(cand_dists_arr)[:k]

        return cand_dists_arr[top_k_indices], [candidate_ids[i] for i in top_k_indices]

    def __len__(self) -> int:
        return self.total_vectors

    def __repr__(self) -> str:
        return (
            f"IVFPQIndex(dim={self.dim}, n_lists={self.n_lists}, "
            f"m={self.m}, size={self.total_vectors}, trained={self.is_trained})"
        )
