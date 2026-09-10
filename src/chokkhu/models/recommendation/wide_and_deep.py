"""Wide & Deep Learning Recommender.

Pure NumPy implementation of Cheng et al. (2016) combining linear memorization
and deep neural generalization for click-through rate (CTR) prediction and recommendation.
"""

from typing import List, Optional
import numpy as np


class WideAndDeep:
    r"""Wide & Deep Learning Model for Recommendation & CTR Prediction.

    .. math::
        P(Y=1|x) = \sigma(w_{	ext{wide}}^T x_{	ext{wide}} + w_{	ext{deep}}^T a^{(L)} + b)

    Parameters
    ----------
    sparse_cardinalities : List[int]
        Cardinalities (vocabulary sizes) of categorical feature columns.
    embedding_dim : int, default=16
        Dimension of latent embeddings for sparse categorical columns.
    deep_hidden_units : List[int], default=[64, 32]
        Hidden layer sizes for the deep MLP feedforward network.
    lr : float, default=0.01
        Learning rate.
    n_epochs : int, default=20
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        sparse_cardinalities: Optional[List[int]] = None,
        embedding_dim: int = 16,
        deep_hidden_units: Optional[List[int]] = None,
        lr: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 64,
        seed: Optional[int] = 42,
    ) -> None:
        self.sparse_cardinalities = sparse_cardinalities or []
        self.embedding_dim = embedding_dim
        self.deep_hidden_units = deep_hidden_units or [64, 32]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

        self.dense_dim: int = 0
        self.wide_weights: Optional[np.ndarray] = None
        self.embeddings: List[np.ndarray] = []
        self.mlp_weights: List[np.ndarray] = []
        self.mlp_biases: List[np.ndarray] = []
        self.deep_head_w: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        X_dense: Optional[np.ndarray],
        X_sparse: Optional[np.ndarray],
        y: np.ndarray,
    ) -> "WideAndDeep":
        """Fit Wide & Deep model on dense and sparse feature arrays."""
        rng = np.random.RandomState(self.seed)
        n_samples = len(y)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.dense_dim = X_dense.shape[1] if X_dense is not None else 0
        n_sparse = X_sparse.shape[1] if X_sparse is not None else 0

        if not self.sparse_cardinalities and n_sparse > 0:
            assert X_sparse is not None
            self.sparse_cardinalities = [
                int(np.max(X_sparse[:, col])) + 1 for col in range(n_sparse)
            ]

        # Wide weights: dense features + one-hot sum of sparse
        wide_dim = self.dense_dim + sum(self.sparse_cardinalities)
        self.wide_weights = (rng.randn(wide_dim, 1) * 0.01).astype(np.float32)

        # Deep embeddings
        self.embeddings = [
            (rng.randn(card, self.embedding_dim) * 0.05).astype(np.float32)
            for card in self.sparse_cardinalities
        ]

        # Deep MLP
        deep_in_dim = self.dense_dim + n_sparse * self.embedding_dim
        self.mlp_weights = []
        self.mlp_biases = []
        cur_in = deep_in_dim

        for units in self.deep_hidden_units:
            w = (rng.randn(cur_in, units) * np.sqrt(2.0 / cur_in)).astype(np.float32)
            bias_vec: np.ndarray = np.zeros(units, dtype=np.float32)
            self.mlp_weights.append(w)
            self.mlp_biases.append(bias_vec)
            cur_in = units

        self.deep_head_w = (rng.randn(cur_in, 1) * np.sqrt(2.0 / cur_in)).astype(
            np.float32
        )
        self.bias = 0.0

        for _ in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            n_batches = int(np.ceil(n_samples / self.batch_size))

            for batch_idx in range(n_batches):
                idx = perm[
                    batch_idx
                    * self.batch_size : min(
                        n_samples, (batch_idx + 1) * self.batch_size
                    )
                ]
                batch_n = len(idx)
                b_y = y_arr[idx]

                b_dense: np.ndarray = (
                    X_dense[idx]
                    if X_dense is not None
                    else np.empty((batch_n, 0), dtype=np.float32)
                )
                b_sparse: np.ndarray = (
                    X_sparse[idx]
                    if X_sparse is not None
                    else np.empty((batch_n, 0), dtype=np.int32)
                )

                # 1. Forward Wide
                wide_feats = [b_dense]
                for col_idx, card in enumerate(self.sparse_cardinalities):
                    col_vals = b_sparse[:, col_idx]
                    one_hot: np.ndarray = np.zeros((batch_n, card), dtype=np.float32)
                    one_hot[np.arange(batch_n), col_vals] = 1.0
                    wide_feats.append(one_hot)

                wide_x = np.concatenate(wide_feats, axis=1)
                wide_out = np.dot(wide_x, self.wide_weights)

                # 2. Forward Deep
                deep_feats = [b_dense]
                for col_idx in range(n_sparse):
                    emb_vals = self.embeddings[col_idx][b_sparse[:, col_idx]]
                    deep_feats.append(emb_vals)

                deep_in = np.concatenate(deep_feats, axis=1)

                mlp_acts = [deep_in]
                mlp_pre = []
                cur = deep_in
                for w, bias in zip(self.mlp_weights, self.mlp_biases):
                    z = np.dot(cur, w) + bias
                    mlp_pre.append(z)
                    cur = np.maximum(0.0, z)
                    mlp_acts.append(cur)

                deep_out = np.dot(cur, self.deep_head_w)

                logits = wide_out + deep_out + self.bias
                preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

                # Backward
                dlogits = (preds - b_y) / batch_n

                # Wide updates
                dwide_w = np.dot(wide_x.T, dlogits)
                self.wide_weights -= self.lr * dwide_w
                self.bias -= float(self.lr * np.sum(dlogits))

                # Deep updates
                ddeep_head_w = np.dot(cur.T, dlogits)
                dcur = np.dot(dlogits, self.deep_head_w.T)
                self.deep_head_w -= self.lr * ddeep_head_w

                for layer_idx in reversed(range(len(self.mlp_weights))):
                    dz = dcur * (mlp_pre[layer_idx] > 0).astype(np.float32)
                    dw = np.dot(mlp_acts[layer_idx].T, dz)
                    db = np.sum(dz, axis=0)
                    dcur = np.dot(dz, self.mlp_weights[layer_idx].T)
                    self.mlp_weights[layer_idx] -= self.lr * dw
                    self.mlp_biases[layer_idx] -= self.lr * db

                # Embedding updates
                offset = self.dense_dim
                for col_idx in range(n_sparse):
                    demb = dcur[:, offset : offset + self.embedding_dim]
                    offset += self.embedding_dim
                    np.add.at(
                        self.embeddings[col_idx], b_sparse[:, col_idx], -self.lr * demb
                    )

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X_dense: Optional[np.ndarray] = None,
        X_sparse: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict click probability for input samples."""
        if not self.is_fitted or self.wide_weights is None or self.deep_head_w is None:
            raise RuntimeError("Model must be fitted before predict_proba.")

        n_samples = (
            len(X_dense)
            if X_dense is not None
            else (len(X_sparse) if X_sparse is not None else 0)
        )
        b_dense: np.ndarray = (
            X_dense
            if X_dense is not None
            else np.empty((n_samples, 0), dtype=np.float32)
        )
        b_sparse: np.ndarray = (
            X_sparse
            if X_sparse is not None
            else np.empty((n_samples, 0), dtype=np.int32)
        )
        n_sparse = b_sparse.shape[1]

        # Wide forward
        wide_feats = [b_dense]
        for col_idx, card in enumerate(self.sparse_cardinalities):
            col_vals = b_sparse[:, col_idx]
            one_hot: np.ndarray = np.zeros((n_samples, card), dtype=np.float32)
            one_hot[np.arange(n_samples), col_vals] = 1.0
            wide_feats.append(one_hot)

        wide_x = np.concatenate(wide_feats, axis=1)
        wide_out = np.dot(wide_x, self.wide_weights)

        # Deep forward
        deep_feats = [b_dense]
        for col_idx in range(n_sparse):
            emb_vals = self.embeddings[col_idx][b_sparse[:, col_idx]]
            deep_feats.append(emb_vals)

        cur = np.concatenate(deep_feats, axis=1)
        for w, bias in zip(self.mlp_weights, self.mlp_biases):
            cur = np.maximum(0.0, np.dot(cur, w) + bias)

        deep_out = np.dot(cur, self.deep_head_w)
        logits = wide_out + deep_out + self.bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return preds.ravel()

    def predict(
        self,
        X_dense: Optional[np.ndarray] = None,
        X_sparse: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X_dense, X_sparse) >= threshold).astype(np.int32)
