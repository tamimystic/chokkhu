"""Vietoris-Rips Simplicial Complex and Persistent Homology computation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class UnionFind:
    """Disjoint-set data structure for 0-dimensional persistent homology (H0)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.birth_times = [0.0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(
        self, i: int, j: int, current_time: float
    ) -> Optional[Tuple[float, float]]:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Elder rule: younger component dies
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i

            death_pair = (self.birth_times[root_j], current_time)
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return death_pair
        return None


class VietorisRipsComplex:
    """Vietoris-Rips Simplicial Complex and filtration builder.

    Computes 0D (connected components) and 1D (loops/holes) persistence pairs.

    Parameters
    ----------
    max_edge_length : float
        Maximum filtration radius epsilon to construct edges.
    max_dimension : int
        Maximum homology dimension (0 for H0, 1 for H0 and H1).
    """

    def __init__(
        self, max_edge_length: float = float("inf"), max_dimension: int = 1
    ) -> None:
        self.max_edge_length = float(max_edge_length)
        self.max_dimension = int(max_dimension)

    def fit_transform(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute persistence diagrams for dimensions 0 and 1 from point cloud X."""
        pts = np.asarray(X, dtype=np.float64)
        n = len(pts)

        # Pairwise distance matrix
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        # Collect all edges
        edges: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dists[i, j])
                if d <= self.max_edge_length:
                    edges.append((d, i, j))

        edges.sort(key=lambda x: x[0])

        # 1. Dimension 0 (H0) via Union-Find
        uf = UnionFind(n)
        h0_pairs: List[Tuple[float, float]] = []

        for d, u, v in edges:
            death_info = uf.union(u, v, d)
            if death_info is not None:
                b, death = death_info
                if death > b:
                    h0_pairs.append((b, death))

        # Essential feature that lives to infinity
        h0_pairs.append((0.0, float("inf")))

        # 2. Dimension 1 (H1) persistence via cycle tracking
        h1_pairs: List[Tuple[float, float]] = []
        if self.max_dimension >= 1 and len(edges) >= 3:
            # Triangles (2-simplices)
            triangles: List[Tuple[float, int, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    d_ij = dists[i, j]
                    for k in range(j + 1, n):
                        d_jk = dists[j, k]
                        d_ik = dists[i, k]
                        max_d = max(d_ij, d_jk, d_ik)
                        if max_d <= self.max_edge_length:
                            triangles.append((float(max_d), i, j, k))

            triangles.sort(key=lambda x: x[0])

            for d_tri, u, v, w in triangles:
                # Triangle uvw kills the cycle formed by edges (u,v), (v,w), (u,w)
                birth = max(dists[u, v], dists[v, w], dists[u, w])
                death = d_tri
                if death > birth:
                    h1_pairs.append((float(birth), float(death)))

        h0_arr = np.array(h0_pairs, dtype=np.float64) if h0_pairs else np.empty((0, 2))
        h1_arr = np.array(h1_pairs, dtype=np.float64) if h1_pairs else np.empty((0, 2))

        return {0: h0_arr, 1: h1_arr}
