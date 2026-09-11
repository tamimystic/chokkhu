"""RegMean (Regression Mean Matrix Fusion) for Neural Weight Fusion.

Reference:
    Jin et al., "Dataless Knowledge Fusion by Merging Weights of Language Models", ICLR 2023.
"""

from typing import Dict, List, Union, Any, Optional
import numpy as np


class RegMean:
    """RegMean: Optimal linear layer weight fusion minimizing expected activation error.

    Closed-form solution:
        W* = (sum_k G_k + eps * I)^{-1} (sum_k G_k W_k)
    where G_k = X_k^T X_k is the input Gram matrix for model k.
    """

    def __init__(self, eps: float = 1e-5, diagonal_approx: bool = False) -> None:
        """Initialize RegMean.

        Args:
            eps: Regularization damping coefficient for matrix inversion.
            diagonal_approx: Whether to use diagonal Gram approximation for high dimensions.
        """
        self.eps = float(eps)
        self.diagonal_approx = bool(diagonal_approx)

    @staticmethod
    def compute_gram_matrix(activations: np.ndarray) -> np.ndarray:
        """Compute Gram matrix G = X^T X from input activations array (N, d_in)."""
        X = activations.astype(np.float64)
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
        return X.T @ X

    def merge_layers(
        self,
        weights_list: List[np.ndarray],
        gram_matrices_list: List[np.ndarray],
        eps: Optional[float] = None,
        diagonal_approx: Optional[bool] = None,
    ) -> np.ndarray:
        """Merge a list of linear layer weight matrices using their corresponding input Gram matrices.

        Args:
            weights_list: List of weight matrices [W_1, ..., W_K].
                          Standard shape: (d_in, d_out) or (d_out, d_in).
            gram_matrices_list: List of Gram matrices [G_1, ..., G_K] of shape (d_in, d_in).
            eps: Regularization constant.
            diagonal_approx: Override diagonal approximation flag.

        Returns:
            Optimal merged weight matrix.
        """
        if len(weights_list) != len(gram_matrices_list):
            raise ValueError(
                f"Length mismatch: {len(weights_list)} weights vs {len(gram_matrices_list)} grams"
            )
        if not weights_list:
            raise ValueError("weights_list cannot be empty")

        regularizer = self.eps if eps is None else float(eps)
        diag = self.diagonal_approx if diagonal_approx is None else bool(diagonal_approx)

        K = len(weights_list)
        W0 = weights_list[0].astype(np.float64)
        G0 = gram_matrices_list[0].astype(np.float64)
        d_in = G0.shape[0]

        # Determine orientation: (d_in, d_out) or (d_out, d_in)
        transposed = False
        if W0.shape[0] != d_in and W0.shape[1] == d_in:
            # W is (d_out, d_in) -> transpose to (d_in, d_out)
            transposed = True
            weights_list = [W.astype(np.float64).T for W in weights_list]
        else:
            weights_list = [W.astype(np.float64) for W in weights_list]

        if diag:
            # Diagonal approximation
            sum_diag_G = np.zeros(d_in, dtype=np.float64)
            sum_GW = np.zeros_like(weights_list[0], dtype=np.float64)
            for W, G in zip(weights_list, gram_matrices_list):
                dG = np.diag(G.astype(np.float64))
                sum_diag_G += dG
                sum_GW += dG[:, np.newaxis] * W

            W_star = sum_GW / (sum_diag_G[:, np.newaxis] + regularizer)
        else:
            # Exact closed-form matrix inversion
            sum_G = np.zeros((d_in, d_in), dtype=np.float64)
            sum_GW = np.zeros_like(weights_list[0], dtype=np.float64)
            for W, G in zip(weights_list, gram_matrices_list):
                G_f = G.astype(np.float64)
                sum_G += G_f
                sum_GW += G_f @ W

            # Regularized inversion
            inv_sum_G = np.linalg.pinv(sum_G + regularizer * np.eye(d_in))
            W_star = inv_sum_G @ sum_GW

        if transposed:
            W_star = W_star.T

        return W_star.astype(weights_list[0].dtype)
