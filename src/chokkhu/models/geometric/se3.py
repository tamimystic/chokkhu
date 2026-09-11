"""SE(3) Equivariant Steerable Point Convolutions and Spherical Harmonics.

Formulated from first principles using real spherical harmonics basis expansions Y_l^m
and continuous radial basis filter steerability in pure NumPy.
"""

from typing import Optional

import numpy as np


class SphericalHarmonics:
    """Real Spherical Harmonics Basis Functions Y_l^m up to degree l=2.

    Parameters
    ----------
    max_l : int, default=2
        Maximum harmonic band degree l in {0, 1, 2}.
    """

    def __init__(self, max_l: int = 2) -> None:
        self.max_l = min(2, max(0, int(max_l)))

    @staticmethod
    def compute(vectors: np.ndarray, max_l: int = 2) -> np.ndarray:
        """Compute real spherical harmonics for a batch of 3D direction vectors.

        Parameters
        ----------
        vectors : np.ndarray of shape (N, 3) or (3,)
            Input 3D cartesian coordinates [x, y, z].
        max_l : int, default=2
            Maximum spherical harmonic degree.

        Returns
        -------
        harmonics : np.ndarray of shape (N, (max_l + 1)^2)
            Spherical harmonic evaluations.
        """
        vecs = np.atleast_2d(np.asarray(vectors, dtype=float))
        n_points = vecs.shape[0]

        x = vecs[:, 0]
        y = vecs[:, 1]
        z = vecs[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12

        # Normalized coordinates on unit sphere
        nx = x / r
        ny = y / r
        nz = z / r

        # Total basis components: (max_l + 1)^2
        # l=0: 1 component (m=0)
        # l=1: 3 components (m=-1, 0, 1)
        # l=2: 5 components (m=-2, -1, 0, 1, 2)
        total_dim = (max_l + 1) ** 2
        Y = np.zeros((n_points, total_dim), dtype=float)

        # l = 0, m = 0
        Y[:, 0] = 0.5 * np.sqrt(1.0 / np.pi)

        if max_l >= 1:
            c1 = np.sqrt(3.0 / (4.0 * np.pi))
            Y[:, 1] = c1 * ny  # l=1, m=-1
            Y[:, 2] = c1 * nz  # l=1, m=0
            Y[:, 3] = c1 * nx  # l=1, m=1

        if max_l >= 2:
            c2_15 = 0.5 * np.sqrt(15.0 / np.pi)
            c2_5 = 0.25 * np.sqrt(5.0 / np.pi)
            Y[:, 4] = c2_15 * (nx * ny)  # l=2, m=-2
            Y[:, 5] = c2_15 * (ny * nz)  # l=2, m=-1
            Y[:, 6] = c2_5 * (2.0 * nz**2 - nx**2 - ny**2)  # l=2, m=0
            Y[:, 7] = c2_15 * (nx * nz)  # l=2, m=1
            Y[:, 8] = 0.25 * np.sqrt(15.0 / np.pi) * (nx**2 - ny**2)  # l=2, m=2

        return Y if vectors.ndim == 2 else Y[0]


class SE3EquivariantConv:
    """SE(3) Equivariant Steerable Point Convolution.

    Applies continuous equivariant filtering over 3D point clouds and molecular graphs:
    f_out(i) = sum_{j in N(i)} W_radial(||r_ij||) * (Y_l^m(r_ij / ||r_ij||) tensor f_in(j)) + b

    Parameters
    ----------
    in_dim : int
        Input scalar/vector feature dimension.
    out_dim : int
        Output feature dimension.
    max_l : int, default=1
        Maximum spherical harmonic degree for steerable angular filter.
    num_radial_bases : int, default=8
        Number of Gaussian Radial Basis Functions (RBF).
    cutoff_radius : float, default=5.0
        Neighbor interaction cutoff radius.
    seed : int, default=42
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_l: int = 1,
        num_radial_bases: int = 8,
        cutoff_radius: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.max_l = min(2, max(0, int(max_l)))
        self.num_radial_bases = max(2, int(num_radial_bases))
        self.cutoff_radius = float(cutoff_radius)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.harmonic_dim = (self.max_l + 1) ** 2
        self.rbf_centers = np.linspace(0.1, self.cutoff_radius, self.num_radial_bases)
        self.rbf_gamma = 1.0 / (
            (self.cutoff_radius / self.num_radial_bases) ** 2 + 1e-6
        )

        # Learnable filter weights: (num_radial_bases, in_dim * harmonic_dim, out_dim)
        scale = 1.0 / np.sqrt(self.in_dim * self.harmonic_dim)
        self.radial_weights = (
            self.rng.randn(
                self.num_radial_bases,
                self.in_dim * self.harmonic_dim,
                self.out_dim,
            )
            * scale
        )
        self.bias = np.zeros(self.out_dim)

    def _radial_basis(self, distances: np.ndarray) -> np.ndarray:
        """Compute Gaussian RBF activations with cosine cutoff envelope."""
        d = distances[..., None]
        rbf = np.exp(-self.rbf_gamma * ((d - self.rbf_centers) ** 2))
        # Cosine smooth cutoff envelope: 0.5 * (cos(pi * d / r_cut) + 1) for d <= r_cut
        envelope = 0.5 * (
            np.cos(np.clip(np.pi * d / self.cutoff_radius, 0.0, np.pi)) + 1.0
        )
        envelope = np.where(d <= self.cutoff_radius, envelope, 0.0)
        return rbf * envelope

    def forward(
        self,
        pos: np.ndarray,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass for SE(3) steerable point convolution.

        Parameters
        ----------
        pos : np.ndarray of shape (N, 3)
            3D Cartesian coordinates of nodes/atoms.
        node_features : np.ndarray of shape (N, in_dim)
            Input node attribute features.
        edge_index : Optional[np.ndarray] of shape (2, E), default=None
            Graph adjacency connectivity (source_idx, target_idx). If None, all pairs within cutoff are connected.

        Returns
        -------
        out_features : np.ndarray of shape (N, out_dim)
            Updated node representations.
        """
        P = np.asarray(pos, dtype=float)
        H = np.asarray(node_features, dtype=float)
        N = P.shape[0]

        if edge_index is None:
            # Build all-pairs within cutoff
            diff = P[:, None, :] - P[None, :, :]  # (N, N, 3): r_ij = pos_j - pos_i
            dist = np.linalg.norm(diff, axis=-1)  # (N, N)
            mask = (dist <= self.cutoff_radius) & (dist > 1e-8)
            src_nodes, tgt_nodes = np.where(mask)
        else:
            tgt_nodes, src_nodes = edge_index[0], edge_index[1]
            diff = P[src_nodes] - P[tgt_nodes]
            dist = np.linalg.norm(diff, axis=-1)

        out_features = np.zeros((N, self.out_dim), dtype=float)

        if len(src_nodes) == 0:
            return out_features + self.bias

        # 1. Spherical Harmonics of relative displacement vectors
        rel_vecs = diff[src_nodes, tgt_nodes] if edge_index is None else diff
        r_ij = dist[src_nodes, tgt_nodes] if edge_index is None else dist
        Y_harmonics = SphericalHarmonics.compute(
            rel_vecs, max_l=self.max_l
        )  # (E, harmonic_dim)

        # 2. Radial RBF activations
        R_bases = self._radial_basis(r_ij)  # (E, num_radial_bases)

        # 3. Kernel construction per edge: W(r_ij) = sum_k R_k(r_ij) * W_k -> (E, in_dim * harmonic_dim, out_dim)
        W_edges = np.einsum("ek,kio->eio", R_bases, self.radial_weights)

        # 4. Steerable tensor product: (Y_l^m tensor h_src)
        h_src = H[src_nodes]  # (E, in_dim)
        # Kronecker / outer product per edge: (E, in_dim * harmonic_dim)
        steerable_input = np.einsum("ei,eh->eih", h_src, Y_harmonics).reshape(
            len(src_nodes), self.in_dim * self.harmonic_dim
        )

        # 5. Message passing and aggregation into target nodes
        messages = np.einsum("ei,eio->eo", steerable_input, W_edges)

        np.add.at(out_features, tgt_nodes, messages)
        out_features = out_features + self.bias
        return out_features
