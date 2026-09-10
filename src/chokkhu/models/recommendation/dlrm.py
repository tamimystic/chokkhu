"""Deep Learning Recommendation Model (DLRM).

Pure NumPy implementation of Naumov et al. (2019, Meta AI) combining bottom MLP
for dense numerical features, embedding tables for sparse categorical features,
explicit dot-product feature interaction, and top MLP for click-through rate prediction.
"""

from typing import List, Optional
import numpy as np


class DLRM:
    r"""Deep Learning Recommendation Model (DLRM).

    Meta's state-of-the-art recommendation architecture:
    1. Dense bottom MLP maps continuous numerical features to embedding dimension :math:`D`.
    2. Categorical embeddings map :math:`S` sparse features to vectors in :math:`\mathbb{R}^D`.
    3. Explicit pairwise dot-product feature interaction computes all :math:`\mathbf{v}_i^T \mathbf{v}_j`.
    4. Top MLP maps concatenated interaction representation to final probability score.

    Parameters
    ----------
    embedding_dim : int, default=16
        Common latent vector dimensionality for both dense projection and sparse embeddings.
    sparse_cardinalities : List[int]
        Vocabulary sizes for each categorical column.
    bottom_mlp_units : List[int], default=[32, 16]
        Hidden layers for bottom continuous feature MLP. The last unit MUST equal embedding_dim.
    top_mlp_units : List[int], default=[64, 32]
        Hidden layers for top classification MLP.
    lr : float, default=0.01
        Learning rate.
    n_epochs : int, default=20
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        sparse_cardinalities: Optional[List[int]] = None,
        bottom_mlp_units: Optional[List[int]] = None,
        top_mlp_units: Optional[List[int]] = None,
        lr: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 64,
        seed: Optional[int] = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.sparse_cardinalities = sparse_cardinalities or []
        self.bottom_mlp_units = bottom_mlp_units or [32, embedding_dim]
        self.top_mlp_units = top_mlp_units or [64, 32]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

        self.dense_dim: int = 0
        self.embeddings: List[np.ndarray] = []

        # Bottom MLP weights
        self.b_weights: List[np.ndarray] = []
        self.b_biases: List[np.ndarray] = []

        # Top MLP weights
        self.t_weights: List[np.ndarray] = []
        self.t_biases: List[np.ndarray] = []
        self.top_head_w: Optional[np.ndarray] = None
        self.top_head_b: float = 0.0

        self.is_fitted: bool = False

    def fit(
        self,
        X_dense: Optional[np.ndarray],
        X_sparse: np.ndarray,
        y: np.ndarray,
    ) -> "DLRM":
        """Fit DLRM model on dense and sparse feature arrays."""
        rng = np.random.RandomState(self.seed)
        X_sp = np.asarray(X_sparse, dtype=np.int32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples, n_sparse = X_sp.shape
        self.dense_dim = X_dense.shape[1] if X_dense is not None else 0

        if not self.sparse_cardinalities:
            self.sparse_cardinalities = [
                int(np.max(X_sp[:, col])) + 1 for col in range(n_sparse)
            ]

        # Sparse Embedding Tables
        self.embeddings = [
            (rng.randn(card, self.embedding_dim) * 0.05).astype(np.float32)
            for card in self.sparse_cardinalities
        ]

        # Bottom MLP (if dense features present)
        self.b_weights = []
        self.b_biases = []
        if self.dense_dim > 0:
            cur_in = self.dense_dim
            for units in self.bottom_mlp_units:
                w = (rng.randn(cur_in, units) * np.sqrt(2.0 / cur_in)).astype(
                    np.float32
                )
                bias_vec: np.ndarray = np.zeros(units, dtype=np.float32)
                self.b_weights.append(w)
                self.b_biases.append(bias_vec)
                cur_in = units

        # Interaction Dimension: num_vectors = (1 if dense else 0) + n_sparse
        num_vecs = (1 if self.dense_dim > 0 else 0) + n_sparse
        num_interactions = (num_vecs * (num_vecs - 1)) // 2
        top_in_dim = self.embedding_dim + num_interactions

        # Top MLP
        self.t_weights = []
        self.t_biases = []
        cur_top = top_in_dim
        for units in self.top_mlp_units:
            w = (rng.randn(cur_top, units) * np.sqrt(2.0 / cur_top)).astype(np.float32)
            bias_vec = np.zeros(units, dtype=np.float32)
            self.t_weights.append(w)
            self.t_biases.append(bias_vec)
            cur_top = units

        self.top_head_w = (rng.randn(cur_top, 1) * np.sqrt(2.0 / cur_top)).astype(
            np.float32
        )
        self.top_head_b = 0.0

        triu_indices = np.triu_indices(num_vecs, k=1)

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
                b_sparse = X_sp[idx]
                b_y = y_arr[idx]

                vectors: List[np.ndarray] = []

                # Bottom MLP
                b_acts = []
                b_pres = []
                if self.dense_dim > 0:
                    assert X_dense is not None
                    b_dense = X_dense[idx]
                    cur = b_dense
                    b_acts.append(cur)
                    for w, bias in zip(self.b_weights, self.b_biases):
                        z = np.dot(cur, w) + bias
                        b_pres.append(z)
                        cur = np.maximum(0.0, z)
                        b_acts.append(cur)
                    v0 = cur
                    vectors.append(v0)

                # Sparse Embeddings
                for f in range(n_sparse):
                    vectors.append(self.embeddings[f][b_sparse[:, f]])

                # Pairwise Dot Products
                V = np.stack(vectors, axis=1)
                Gram = np.matmul(V, V.transpose(0, 2, 1))
                interactions = Gram[:, triu_indices[0], triu_indices[1]]

                first_vec = vectors[0]
                top_in = np.concatenate([first_vec, interactions], axis=1)

                # Forward Top MLP
                t_acts = [top_in]
                t_pres = []
                cur = top_in
                for w, bias in zip(self.t_weights, self.t_biases):
                    z = np.dot(cur, w) + bias
                    t_pres.append(z)
                    cur = np.maximum(0.0, z)
                    t_acts.append(cur)

                logits = np.dot(cur, self.top_head_w) + self.top_head_b
                preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

                # Backward
                dlogits = (preds - b_y) / batch_n

                # Top Head
                dtop_head_w = np.dot(cur.T, dlogits)
                self.top_head_w -= self.lr * dtop_head_w
                self.top_head_b -= float(self.lr * np.sum(dlogits))

                dcur = np.dot(dlogits, self.top_head_w.T)
                for layer_idx in reversed(range(len(self.t_weights))):
                    dz = dcur * (t_pres[layer_idx] > 0).astype(np.float32)
                    dw = np.dot(t_acts[layer_idx].T, dz)
                    db = np.sum(dz, axis=0)
                    dcur = np.dot(dz, self.t_weights[layer_idx].T)
                    self.t_weights[layer_idx] -= self.lr * dw
                    self.t_biases[layer_idx] -= self.lr * db

                dfirst_vec = dcur[:, : self.embedding_dim]
                dinteractions = dcur[:, self.embedding_dim :]

                dGram = np.zeros_like(Gram)
                dGram[:, triu_indices[0], triu_indices[1]] = dinteractions
                dGram[:, triu_indices[1], triu_indices[0]] = dinteractions

                dV = 2.0 * np.matmul(dGram, V)
                dV[:, 0, :] += dfirst_vec

                # Update Embeddings
                vec_offset = 1 if self.dense_dim > 0 else 0
                for f in range(n_sparse):
                    np.add.at(
                        self.embeddings[f],
                        b_sparse[:, f],
                        -self.lr * dV[:, vec_offset + f, :],
                    )

                # Update Bottom MLP
                if self.dense_dim > 0:
                    dcur_b = dV[:, 0, :]
                    for layer_idx in reversed(range(len(self.b_weights))):
                        dz = dcur_b * (b_pres[layer_idx] > 0).astype(np.float32)
                        dw = np.dot(b_acts[layer_idx].T, dz)
                        db = np.sum(dz, axis=0)
                        dcur_b = np.dot(dz, self.b_weights[layer_idx].T)
                        self.b_weights[layer_idx] -= self.lr * dw
                        self.b_biases[layer_idx] -= self.lr * db

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X_dense: Optional[np.ndarray],
        X_sparse: np.ndarray,
    ) -> np.ndarray:
        """Predict interaction probabilities for input query."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X_sp = np.asarray(X_sparse, dtype=np.int32)
        n_samples, n_sparse = X_sp.shape
        num_vecs = (1 if self.dense_dim > 0 else 0) + n_sparse
        triu_indices = np.triu_indices(num_vecs, k=1)

        vectors: List[np.ndarray] = []
        if self.dense_dim > 0:
            assert X_dense is not None
            cur = X_dense
            for w, bias in zip(self.b_weights, self.b_biases):
                cur = np.maximum(0.0, np.dot(cur, w) + bias)
            vectors.append(cur)

        for f in range(n_sparse):
            vectors.append(self.embeddings[f][X_sp[:, f]])

        V = np.stack(vectors, axis=1)
        Gram = np.matmul(V, V.transpose(0, 2, 1))
        interactions = Gram[:, triu_indices[0], triu_indices[1]]
        first_vec = vectors[0]

        cur = np.concatenate([first_vec, interactions], axis=1)
        for w, bias in zip(self.t_weights, self.t_biases):
            cur = np.maximum(0.0, np.dot(cur, w) + bias)

        logits = np.dot(cur, self.top_head_w) + self.top_head_b
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return preds.ravel()

    def predict(
        self,
        X_dense: Optional[np.ndarray],
        X_sparse: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary click labels."""
        return (self.predict_proba(X_dense, X_sparse) >= threshold).astype(np.int32)
