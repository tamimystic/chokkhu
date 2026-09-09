"""Locality-Sensitive Hashing (LSH) Index.

Pure NumPy implementation of Random Hyperplanes LSH (for Cosine/Angular similarity)
and MinHash LSH (for Jaccard set similarity).
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class RandomHyperplaneLSH:
    """Random Hyperplanes Locality-Sensitive Hashing for Cosine/Angular similarity.

    Parameters
    ----------
    dim : int
        Dimensionality of the feature vectors.
    n_tables : int, default=8
        Number of independent hash tables.
    n_bits : int, default=16
        Number of random hyperplane bits per hash table.
    seed : Optional[int], default=42
        Random seed for hyperplane generation.
    """

    def __init__(
        self,
        dim: int,
        n_tables: int = 8,
        n_bits: int = 16,
        seed: Optional[int] = 42,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if n_bits <= 0 or n_bits > 64:
            raise ValueError(f"n_bits must be in [1, 64], got {n_bits}")

        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits

        rng = np.random.RandomState(seed)
        # Hyperplanes: shape (n_tables, n_bits, dim)
        self.hyperplanes = rng.randn(n_tables, n_bits, dim).astype(np.float32)

        # Hash tables: tables[table_idx][bucket_hash] = list of vector_idx
        self.tables: List[Dict[int, List[int]]] = [{} for _ in range(n_tables)]

        self.vectors: List[np.ndarray] = []
        self.ids: List[Union[int, str]] = []

    def _hash_vector(self, vec: np.ndarray, table_idx: int) -> int:
        """Compute integer bucket hash from random hyperplane projections."""
        projs = np.dot(self.hyperplanes[table_idx], vec)
        bits = (projs >= 0).astype(int)
        # Convert bit array to integer
        powers = 1 << np.arange(self.n_bits, dtype=np.int64)
        return int(np.sum(bits * powers))

    def add(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        ids: Optional[List[Union[int, str]]] = None,
    ) -> None:
        """Insert vectors into the LSH index."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {arr.shape[1]} does not match index dim {self.dim}"
            )

        n_samples = arr.shape[0]
        assigned_ids: List[Union[int, str]]
        if ids is None:
            curr_len = len(self.vectors)
            assigned_ids = [int(i) for i in range(curr_len, curr_len + n_samples)]
        else:
            if len(ids) != n_samples:
                raise ValueError(
                    f"Length of ids ({len(ids)}) must match vectors ({n_samples})"
                )
            assigned_ids = list(ids)

        for i in range(n_samples):
            vec = arr[i]
            idx = len(self.vectors)
            self.vectors.append(vec)
            self.ids.append(assigned_ids[i])

            for t_idx in range(self.n_tables):
                h = self._hash_vector(vec, t_idx)
                if h not in self.tables[t_idx]:
                    self.tables[t_idx][h] = []
                self.tables[t_idx][h].append(idx)

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        k: int = 10,
    ) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """Search for approximate nearest neighbors via multi-table candidate retrieval.

        Parameters
        ----------
        query : np.ndarray of shape (dim,)
            Query vector.
        k : int, default=10
            Number of top nearest neighbors.

        Returns
        -------
        distances : np.ndarray
            Cosine distances (1 - cosine_similarity).
        matched_ids : list
            Matching vector IDs.
        """
        if len(self.vectors) == 0:
            return np.empty(0, dtype=np.float32), []

        q_vec = np.asarray(query, dtype=np.float32).ravel()
        if q_vec.shape[0] != self.dim:
            raise ValueError(
                f"Query dim {q_vec.shape[0]} does not match index dim {self.dim}"
            )

        # Collect candidate indices across all tables
        candidates: Set[int] = set()
        for t_idx in range(self.n_tables):
            h = self._hash_vector(q_vec, t_idx)
            bucket = self.tables[t_idx].get(h, [])
            candidates.update(bucket)

        if not candidates:
            # Fallback to random sample or empty if no collision
            return np.empty(0, dtype=np.float32), []

        cand_list = list(candidates)
        cand_vecs = np.array([self.vectors[i] for i in cand_list], dtype=np.float32)

        # Compute exact cosine distance over candidates
        norm_q = np.linalg.norm(q_vec)
        norm_c = np.linalg.norm(cand_vecs, axis=1)
        norm_c = np.where(norm_c < 1e-12, 1e-12, norm_c)
        cos_sim: np.ndarray
        if norm_q < 1e-12:
            cos_sim = np.zeros(len(cand_list), dtype=np.float32)
        else:
            cos_sim = np.dot(cand_vecs, q_vec) / (norm_c * norm_q)

        dists = np.maximum(0.0, 1.0 - cos_sim)
        top_k_indices = np.argsort(dists)[:k]

        return dists[top_k_indices], [self.ids[cand_list[i]] for i in top_k_indices]

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return (
            f"RandomHyperplaneLSH(dim={self.dim}, n_tables={self.n_tables}, "
            f"n_bits={self.n_bits}, size={len(self.vectors)})"
        )


class MinHashLSH:
    """MinHash Locality-Sensitive Hashing for Jaccard Set Similarity.

    Parameters
    ----------
    n_permutations : int, default=128
        Number of MinHash hash functions.
    threshold : float, default=0.5
        Jaccard similarity threshold for candidate pairing.
    n_bands : Optional[int], default=None
        Number of bands for multi-table banding technique. If None, chosen automatically based on threshold.
    seed : Optional[int], default=42
        Random seed for hash parameters.
    """

    def __init__(
        self,
        n_permutations: int = 128,
        threshold: float = 0.5,
        n_bands: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.num_perm = n_permutations
        self.threshold = threshold

        if n_bands is not None:
            if n_permutations % n_bands != 0:
                raise ValueError(
                    f"n_permutations ({n_permutations}) must be divisible by n_bands ({n_bands})"
                )
            self.b = n_bands
            self.r = n_permutations // n_bands
        else:
            # Auto-tune b and r for target threshold
            best_b = 1
            best_r = n_permutations
            best_diff = float("inf")
            for b in range(1, n_permutations + 1):
                if n_permutations % b == 0:
                    r = n_permutations // b
                    # Approx S-curve threshold: (1/b)^(1/r)
                    t_est = (1.0 / b) ** (1.0 / r)
                    diff = abs(t_est - threshold)
                    if diff < best_diff:
                        best_diff = diff
                        best_b = b
                        best_r = r
            self.b = best_b
            self.r = best_r

        rng = np.random.RandomState(seed)
        self.prime = 2147483647  # 2^31 - 1 Mersenne prime
        self.a = rng.randint(1, self.prime, size=n_permutations, dtype=np.int64)
        self.b_coeffs = rng.randint(0, self.prime, size=n_permutations, dtype=np.int64)

        # Hash tables: tables[band_idx][band_hash] = list of doc_id
        self.tables: List[Dict[int, List[Union[int, str]]]] = [
            {} for _ in range(self.b)
        ]
        self.signatures: Dict[Union[int, str], np.ndarray] = {}

    def compute_signature(
        self, set_indices: Union[Set[int], List[int], np.ndarray]
    ) -> np.ndarray:
        """Compute MinHash signature vector for a set of token/shingle integer IDs."""
        if not set_indices:
            return np.full(self.num_perm, self.prime, dtype=np.int64)

        elems = np.asarray(list(set_indices), dtype=np.int64)[:, None]  # (N, 1)
        # Universal hash function: (a * x + b) % prime
        hashes = (elems * self.a[None, :] + self.b_coeffs[None, :]) % self.prime
        sig = np.min(hashes, axis=0)
        return sig

    def add(
        self,
        doc_id: Union[int, str],
        set_indices: Union[Set[int], List[int], np.ndarray],
    ) -> None:
        """Add a set document to the MinHash LSH index."""
        sig = self.compute_signature(set_indices)
        self.signatures[doc_id] = sig

        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_chunk = tuple(sig[start:end])
            band_hash = hash(band_chunk)

            if band_hash not in self.tables[band_idx]:
                self.tables[band_idx][band_hash] = []
            self.tables[band_idx][band_hash].append(doc_id)

    def search(
        self, set_indices: Union[Set[int], List[int], np.ndarray], k: int = 10
    ) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """Query similar sets with estimated Jaccard similarity >= threshold."""
        q_sig = self.compute_signature(set_indices)
        candidates: Set[Union[int, str]] = set()

        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_chunk = tuple(q_sig[start:end])
            band_hash = hash(band_chunk)

            bucket = self.tables[band_idx].get(band_hash, [])
            candidates.update(bucket)

        if not candidates:
            return np.empty(0, dtype=np.float32), []

        cand_ids = list(candidates)
        cand_sims: List[float] = []

        for cid in cand_ids:
            cand_sig = self.signatures[cid]
            sim = float(np.mean(q_sig == cand_sig))
            cand_sims.append(sim)

        sims_arr = np.array(cand_sims, dtype=np.float32)
        top_indices = np.argsort(-sims_arr)[:k]

        return sims_arr[top_indices], [cand_ids[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self.signatures)

    def __repr__(self) -> str:
        return (
            f"MinHashLSH(n_permutations={self.num_perm}, n_bands={self.b}, "
            f"size={len(self.signatures)})"
        )
