"""Dense Vector Retriever & Hybrid Search Engine.

Provides unified dense vector retrieval with pluggable ANN index backends (HNSW, IVF-PQ, Exact)
and Reciprocal Rank Fusion (RRF) for hybrid sparse-dense information retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from chokkhu.models.retrieval.hnsw import HNSWIndex
from chokkhu.models.retrieval.ivf_pq import IVFPQIndex


def reciprocal_rank_fusion(
    ranked_lists: List[List[Union[int, str]]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Tuple[Union[int, str], float]]:
    """Combine multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{m in rankings} 1 / (k + rank_m(d))

    Parameters
    ----------
    ranked_lists : list of list of IDs
        Multiple ranked result lists from different search engines (e.g. [sparse_results, dense_results]).
    k : int, default=60
        Smoothing constant preventing high ranks from dominating.
    top_n : Optional[int], default=None
        Number of top merged items to return.

    Returns
    -------
    fused_results : list of (doc_id, score) sorted by descending RRF score.
    """
    scores: Dict[Union[int, str], float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]
    return sorted_items


class DenseRetriever:
    """Unified Dense Vector Retriever with metadata storage and pluggable index backend.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    backend : str, default='hnsw'
        ANN backend: 'hnsw', 'ivf_pq', or 'exact'.
    metric : str, default='cosine'
        Distance metric: 'cosine', 'euclidean', or 'dot'.
    index_params : Optional[dict], default=None
        Custom parameters passed to the underlying index.
    """

    def __init__(
        self,
        dim: int,
        backend: str = "hnsw",
        metric: str = "cosine",
        index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dim = dim
        self.backend = backend.lower()
        self.metric = metric.lower()
        self.params = index_params or {}

        self.documents: Dict[Union[int, str], str] = {}
        self.metadata: Dict[Union[int, str], Dict[str, Any]] = {}
        self.vectors: List[np.ndarray] = []
        self.doc_ids: List[Union[int, str]] = []

        if self.backend == "hnsw":
            self.index = HNSWIndex(
                dim=dim,
                metric=metric,
                m=self.params.get("m", 16),
                ef_construction=self.params.get("ef_construction", 64),
                ef_search=self.params.get("ef_search", 32),
                seed=self.params.get("seed", 42),
            )
        elif self.backend == "ivf_pq":
            self.index = IVFPQIndex(
                dim=dim,
                n_lists=self.params.get("n_lists", 32),
                n_subvectors=self.params.get("n_subvectors", 8),
                metric=metric,
                seed=self.params.get("seed", 42),
            )
        elif self.backend == "exact":
            self.index = None
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Must be 'hnsw', 'ivf_pq', or 'exact'."
            )

    def add_documents(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        documents: Optional[List[str]] = None,
        ids: Optional[List[Union[int, str]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embedding vectors and associated document texts/metadata to the retriever."""
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_samples = arr.shape[0]
        if ids is None:
            curr_len = len(self.doc_ids)
            assigned_ids = list(range(curr_len, curr_len + n_samples))
        else:
            if len(ids) != n_samples:
                raise ValueError(f"Length of ids ({len(ids)}) must match embeddings ({n_samples})")
            assigned_ids = ids

        for i in range(n_samples):
            doc_id = assigned_ids[i]
            self.doc_ids.append(doc_id)
            self.vectors.append(arr[i])
            if documents is not None:
                self.documents[doc_id] = documents[i]
            if metadatas is not None:
                self.metadata[doc_id] = metadatas[i]

        if self.backend == "hnsw":
            self.index.add(arr, assigned_ids)
        elif self.backend == "ivf_pq":
            if not self.index.is_trained:
                self.index.train(arr)
            self.index.add(arr, assigned_ids)

    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant documents for a query vector.

        Returns
        -------
        results : list of dicts containing 'id', 'score', 'document', 'metadata'.
        """
        q_vec = np.asarray(query_embedding, dtype=np.float32).ravel()

        if self.backend in ("hnsw", "ivf_pq"):
            dists, ids = self.index.search(q_vec, k=k)
        else:
            # Exact brute force
            all_vecs = np.array(self.vectors, dtype=np.float32)
            if self.metric == "cosine":
                norm_q = np.linalg.norm(q_vec)
                norm_v = np.linalg.norm(all_vecs, axis=1)
                norm_v = np.where(norm_v < 1e-12, 1e-12, norm_v)
                cos_sim = np.dot(all_vecs, q_vec) / (norm_v * max(1e-12, norm_q))
                dists = np.maximum(0.0, 1.0 - cos_sim)
            elif self.metric == "euclidean":
                dists = np.sqrt(np.sum((all_vecs - q_vec) ** 2, axis=1))
            else:
                dists = -np.dot(all_vecs, q_vec)

            top_k_idx = np.argsort(dists)[:k]
            dists = dists[top_k_idx]
            ids = [self.doc_ids[i] for i in top_k_idx]

        results = []
        for dist, doc_id in zip(dists, ids):
            results.append(
                {
                    "id": doc_id,
                    "distance": float(dist),
                    "document": self.documents.get(doc_id, ""),
                    "metadata": self.metadata.get(doc_id, {}),
                }
            )
        return results

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(dim={self.dim}, backend='{self.backend}', "
            f"metric='{self.metric}', size={len(self.doc_ids)})"
        )
