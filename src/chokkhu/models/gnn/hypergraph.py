from __future__ import annotations

import numpy as np


class HypergraphConvolution:
    """
    Hypergraph Spectral Convolution (HGNN) Layer.
    Processes multi-way complex relationships using hyperedge incidence matrix H.

    Formula:
    X^(l+1) = sigma(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X^(l) Theta)

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = np.random.RandomState(seed)
        self.Theta: np.ndarray = rng.randn(in_dim, out_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / in_dim)
        self.bias: np.ndarray = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        X: np.ndarray,
        H: np.ndarray,
        W: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray
            Node feature matrix of shape [N, in_dim].
        H : np.ndarray
            Hypergraph incidence matrix of shape [N, E_hyper], where H[v, e] = 1 if v in e.
        W : np.ndarray | None
            Hyperedge weights of shape [E_hyper] (defaults to ones).

        Returns
        -------
        out : np.ndarray
            Updated features of shape [N, out_dim].
        """
        N, E_hyper = H.shape

        W_mat: np.ndarray = (
            np.eye(E_hyper, dtype=np.float32)
            if W is None
            else np.diag(np.asarray(W, dtype=np.float32))
        )

        # Node degree: d(v) = sum_e W(e) * H(v, e)
        d_v = np.sum(np.dot(H, W_mat), axis=1)
        d_v_inv_sqrt = np.zeros_like(d_v, dtype=np.float32)
        mask_v = d_v > 0
        d_v_inv_sqrt[mask_v] = np.power(d_v[mask_v], -0.5)
        D_v_inv_sqrt = np.diag(d_v_inv_sqrt)

        # Hyperedge degree: d(e) = sum_v H(v, e)
        d_e = np.sum(H, axis=0)
        d_e_inv = np.zeros_like(d_e, dtype=np.float32)
        mask_e = d_e > 0
        d_e_inv[mask_e] = np.power(d_e[mask_e], -1.0)
        D_e_inv = np.diag(d_e_inv)

        # Filter: A_hyper = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
        HT_Dv = np.dot(H.T, D_v_inv_sqrt)
        propagator = np.dot(
            np.dot(np.dot(D_v_inv_sqrt, H), np.dot(W_mat, D_e_inv)), HT_Dv
        )

        out = np.dot(np.dot(propagator, X), self.Theta) + self.bias
        return np.maximum(0.0, out)  # ReLU


class HGNN:
    """
    Deep Hypergraph Neural Network Classifier.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension.
    out_dim : int
        Output number of classes.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42
    ) -> None:
        self.conv1 = HypergraphConvolution(in_dim, hidden_dim, seed=seed)
        self.conv2 = HypergraphConvolution(hidden_dim, out_dim, seed=seed + 1)

    def forward(self, X: np.ndarray, H: np.ndarray) -> np.ndarray:
        h = self.conv1.forward(X, H)
        out = self.conv2.forward(h, H)
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=1, keepdims=True)
