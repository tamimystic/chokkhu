"""Protein Residue Contact Map Prediction via Direct Coupling Analysis (DCA) in pure NumPy."""

from __future__ import annotations


import numpy as np


class ProteinContactMap:
    r"""Co-Evolutionary Sequence Covariance and Direct Coupling Analysis (DCA) in pure NumPy (Marks et al. 2011).

    Analyzes Multiple Sequence Alignments (MSA) to compute residue-residue co-evolutionary
    direct information (DI) and predict 2D contact maps with Average Product Correction (APC).

    Parameters
    ----------
    pseudocount_weight : float, default=0.5
        Pseudocount shrinkage weight $\theta$ regularizing empirical frequencies.
    apc : bool, default=True
        Apply Average Product Correction (APC) to remove phylogenetic background bias.
    """

    def __init__(self, pseudocount_weight: float = 0.5, apc: bool = True) -> None:
        self.pseudocount_weight = float(pseudocount_weight)
        self.apc = apc

    def compute_contact_map(self, msa_matrix: np.ndarray) -> np.ndarray:
        """Computes 2D residue contact probability/coupling matrix from encoded MSA.

        Parameters
        ----------
        msa_matrix : np.ndarray
            Integer-encoded Multiple Sequence Alignment of shape $(N_{seq}, L)$,
            where elements are amino acid alphabet indices $0..20$.

        Returns
        -------
        np.ndarray
            Symmetric residue contact score matrix of shape $(L, L)$.
        """
        msa = np.asarray(msa_matrix, dtype=np.int32)
        N_seq, L = msa.shape
        q = 21  # 20 standard amino acids + 1 gap token

        # 1. Single-site and Pairwise joint empirical frequencies with pseudocount
        weight = self.pseudocount_weight
        f_i = np.zeros((L, q), dtype=np.float64)
        for i in range(L):
            counts = np.bincount(msa[:, i], minlength=q)[:q]
            f_i[i] = (1.0 - weight) * (counts / N_seq) + weight / q

        # Pairwise joint frequencies f_ij(a, b)
        coupling_matrix = np.zeros((L, L), dtype=np.float64)

        for i in range(L):
            for j in range(i + 1, L):
                # Joint frequency
                joint_counts: np.ndarray = np.zeros((q, q), dtype=np.float64)
                for s in range(N_seq):
                    a_i = msa[s, i]
                    a_j = msa[s, j]
                    if a_i < q and a_j < q:
                        joint_counts[a_i, a_j] += 1.0

                f_ij = (1.0 - weight) * (joint_counts / N_seq) + weight / (q * q)
                # Covariance: C_ij = f_ij - f_i x f_j
                cov = f_ij - np.outer(f_i[i], f_i[j])
                # Direct information Frobenius norm
                di_norm = np.sqrt(np.sum(cov**2))
                coupling_matrix[i, j] = di_norm
                coupling_matrix[j, i] = di_norm

        # 2. Average Product Correction (APC)
        if self.apc:
            row_means = np.mean(coupling_matrix, axis=1, keepdims=True)
            col_means = np.mean(coupling_matrix, axis=0, keepdims=True)
            total_mean = np.mean(coupling_matrix)
            if total_mean > 1e-12:
                apc_matrix = np.matmul(row_means, col_means) / total_mean
                coupling_matrix = np.maximum(0.0, coupling_matrix - apc_matrix)

        np.fill_diagonal(coupling_matrix, 0.0)
        return coupling_matrix
