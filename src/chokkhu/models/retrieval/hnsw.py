"""Hierarchical Navigable Small World (HNSW) Graph Index.

Pure NumPy implementation of HNSW for high-dimensional approximate nearest
neighbor (ANN) vector retrieval with logarithmic search complexity.
"""

from typing import Dict, List, Optional, Tuple, Union
import heapq
import numpy as np


class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) Index.

    Parameters
    ----------
    dim : int
        Dimensionality of the input vectors.
    metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean' (L2), 'cosine', 'dot' (inner product).
    m : int, default=16
        Maximum number of outgoing connections per node in layers > 0.
        Layer 0 has max connections M0 = 2 * M.
    ef_construction : int, default=64
        Size of the dynamic candidate list during graph construction.
    ef_search : int, default=32
        Size of the dynamic candidate list during query search.
    seed : Optional[int], default=42
        Random seed for reproducible layer assignment.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "euclidean",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        seed: Optional[int] = 42,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if m <= 1:
            raise ValueError(f"m must be > 1, got {m}")

        self.dim = dim
        self.metric = metric.lower()
        if self.metric not in ("euclidean", "cosine", "dot", "ip"):
            raise ValueError(
                f"Unsupported metric: {metric}. Must be 'euclidean', 'cosine', or 'dot'."
            )

        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / np.log(m)

        self.rng = np.random.RandomState(seed)

        # Storage
        self.vectors: List[np.ndarray] = []
        self.ids: List[Union[int, str]] = []
        self.id_map: Dict[Union[int, str], int] = {}
        self.node_levels: List[int] = []

        # Graph structure: graphs[level][node_idx] = set of neighbor node_idx
        self.graphs: List[Dict[int, List[int]]] = []
        self.enter_point: Optional[int] = None
        self.max_level: int = -1

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two 1D vectors."""
        if self.metric == "euclidean":
            diff = a - b
            return float(np.dot(diff, diff) ** 0.5)
        elif self.metric == "cosine":
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 1.0
            cos_sim = np.dot(a, b) / (norm_a * norm_b)
            return float(max(0.0, 1.0 - cos_sim))
        elif self.metric in ("dot", "ip"):
            return float(-np.dot(a, b))
        return 0.0

    def _batch_distance(
        self, query: np.ndarray, target_indices: List[int]
    ) -> np.ndarray:
        """Vectorized distance between a query vector and a list of target indices."""
        if not target_indices:
            return np.empty(0, dtype=np.float32)

        targets = np.array([self.vectors[i] for i in target_indices], dtype=np.float32)
        if self.metric == "euclidean":
            diff = targets - query
            return np.sqrt(np.sum(diff * diff, axis=1))
        elif self.metric == "cosine":
            norm_q = np.linalg.norm(query)
            norm_t = np.linalg.norm(targets, axis=1)
            norm_t = np.where(norm_t < 1e-12, 1e-12, norm_t)
            if norm_q < 1e-12:
                return np.ones(len(target_indices), dtype=np.float32)
            sim = np.dot(targets, query) / (norm_t * norm_q)
            return np.maximum(0.0, 1.0 - sim)
        elif self.metric in ("dot", "ip"):
            return -np.dot(targets, query)
        return np.zeros(len(target_indices), dtype=np.float32)

    def _get_random_level(self) -> int:
        """Generate a random layer level using exponential distribution."""
        unif = max(1e-10, self.rng.uniform(0.0, 1.0))
        return int(np.floor(-np.log(unif) * self.ml))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_candidates: int,
        level: int,
    ) -> List[Tuple[float, int]]:
        """Greedy beam search in a single layer."""
        visited = set(entry_points)
        candidates: List[Tuple[float, int]] = []  # Min-heap (dist, node)
        nearest_set: List[Tuple[float, int]] = []  # Max-heap (-dist, node)

        for ep in entry_points:
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(nearest_set, (-dist, ep))

        graph_level = self.graphs[level]

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_dist = -nearest_set[0][0]

            if c_dist > furthest_dist:
                break

            neighbors = graph_level.get(c_node, [])
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            for n in unvisited_neighbors:
                visited.add(n)

            if unvisited_neighbors:
                dists = self._batch_distance(query, unvisited_neighbors)
                for n_node, n_dist in zip(unvisited_neighbors, dists):
                    furthest_dist = -nearest_set[0][0]
                    if n_dist < furthest_dist or len(nearest_set) < num_candidates:
                        heapq.heappush(candidates, (n_dist, n_node))
                        heapq.heappush(nearest_set, (-n_dist, n_node))
                        if len(nearest_set) > num_candidates:
                            heapq.heappop(nearest_set)

        results = [(-neg_d, node) for neg_d, node in nearest_set]
        results.sort(key=lambda x: x[0])
        return results

    def _select_neighbors(
        self, candidates: List[Tuple[float, int]], max_connections: int
    ) -> List[int]:
        """Simple heuristic: select top-M closest nodes."""
        candidates.sort(key=lambda x: x[0])
        return [node for _, node in candidates[:max_connections]]

    def add(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        ids: Optional[List[Union[int, str]]] = None,
    ) -> None:
        """Insert vectors into the HNSW index.

        Parameters
        ----------
        vectors : np.ndarray or list of shape (N, dim)
            Feature vectors to insert.
        ids : Optional[list of ID], default=None
            Custom identifiers for the vectors. If None, 0-indexed integers are used.
        """
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
            custom_id = assigned_ids[i]
            self._insert_single(vec, custom_id)

    def _insert_single(self, vector: np.ndarray, custom_id: Union[int, str]) -> int:
        """Insert a single vector into the hierarchical graph."""
        node_idx = len(self.vectors)
        self.vectors.append(vector)
        self.ids.append(custom_id)
        self.id_map[custom_id] = node_idx

        node_level = self._get_random_level()
        self.node_levels.append(node_level)

        # Extend graphs list if node_level exceeds current max_level
        while len(self.graphs) <= node_level:
            self.graphs.append({})

        if self.enter_point is None:
            self.enter_point = node_idx
            self.max_level = node_level
            for lvl in range(node_level + 1):
                self.graphs[lvl][node_idx] = []
            return node_idx

        curr_ep = [self.enter_point]
        max_l = self.max_level

        # 1. Greedy search from max_level down to node_level + 1
        for lvl in range(max_l, node_level, -1):
            nearest = self._search_layer(vector, curr_ep, 1, lvl)
            curr_ep = [nearest[0][1]]

        # 2. From min(node_level, max_l) down to 0: search and connect
        top_level = min(node_level, max_l)
        for lvl in range(top_level, -1, -1):
            candidates = self._search_layer(vector, curr_ep, self.ef_construction, lvl)
            max_conn = self.m0 if lvl == 0 else self.m
            neighbors = self._select_neighbors(candidates, max_conn)

            self.graphs[lvl][node_idx] = neighbors
            for neighbor in neighbors:
                neigh_conns = self.graphs[lvl].get(neighbor, [])
                if node_idx not in neigh_conns:
                    neigh_conns.append(node_idx)
                    # Prune neighbor connections if exceeding max_conn
                    if len(neigh_conns) > max_conn:
                        n_vec = self.vectors[neighbor]
                        pair_dists = [
                            (self._distance(n_vec, self.vectors[c]), c)
                            for c in neigh_conns
                        ]
                        pair_dists.sort(key=lambda x: x[0])
                        self.graphs[lvl][neighbor] = [
                            c for _, c in pair_dists[:max_conn]
                        ]

            curr_ep = [c for _, c in candidates]

        if node_level > self.max_level:
            self.max_level = node_level
            self.enter_point = node_idx

        return node_idx

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """Search for the top-k nearest neighbors of the query vector.

        Parameters
        ----------
        query : np.ndarray or list of shape (dim,)
            Query feature vector.
        k : int, default=10
            Number of nearest neighbors to retrieve.
        ef_search : Optional[int], default=None
            Search candidate pool size. If None, uses self.ef_search.

        Returns
        -------
        distances : np.ndarray of shape (k,)
            Distances of retrieved neighbors.
        matched_ids : list of IDs
            Corresponding vector identifiers.
        """
        if self.enter_point is None or len(self.vectors) == 0:
            return np.empty(0, dtype=np.float32), []

        q_vec = np.asarray(query, dtype=np.float32).ravel()
        if q_vec.shape[0] != self.dim:
            raise ValueError(
                f"Query dim {q_vec.shape[0]} does not match index dim {self.dim}"
            )

        ef = ef_search or self.ef_search
        ef = max(ef, k)

        curr_ep = [self.enter_point]
        for lvl in range(self.max_level, 0, -1):
            nearest = self._search_layer(q_vec, curr_ep, 1, lvl)
            curr_ep = [nearest[0][1]]

        candidates = self._search_layer(q_vec, curr_ep, ef, 0)
        top_k = candidates[:k]

        distances = np.array([dist for dist, _ in top_k], dtype=np.float32)
        matched_ids = [self.ids[node] for _, node in top_k]

        return distances, matched_ids

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.dim}, metric='{self.metric}', m={self.m}, "
            f"size={len(self.vectors)}, max_level={self.max_level})"
        )
