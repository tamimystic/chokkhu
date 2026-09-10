"""DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.

Pure NumPy implementation of Guo et al. (2017) integrating Factorization Machines
and Deep Neural Networks for end-to-end 1st-order, 2nd-order, and higher-order feature interactions.
"""

from typing import List, Optional
import numpy as np


class DeepFM:
    r"""DeepFM Recommender Architecture.

        Jointly learns low- and high-order feature interactions without manual feature engineering:

        .. math::
            \hat{y} = \sigma(y_{	ext{FM1}} + y_{	ext{FM2}} + y_{	ext{Deep}})

        where:
        - :math:`y_{	ext{FM1}} = \langle w, x
    angle + w_0`
        - :math:`y_{	ext{FM2}} =
    rac{1}{2} \sum_{f=1}^K \left[ \left(\sum_{i=1}^m v_{i,f}
    ight)^2 - \sum_{i=1}^m v_{i,f}^2
    ight]`
        - :math:`y_{	ext{Deep}} = W_{	ext{out}} a^{(L)} + b_{	ext{out}}`

        Parameters
        ----------
        field_cardinalities : List[int]
            Number of unique categories for each sparse categorical field.
        embedding_dim : int, default=16
            Dimension of feature latent vectors :math:`K`.
        mlp_layers : List[int], default=[64, 32]
            Hidden layer dimensions for deep component.
        lr : float, default=0.01
            Learning rate.
        n_epochs : int, default=20
            Number of training epochs.
        batch_size : int, default=64
            Mini-batch size.
        seed : Optional[int], default=42
            Random seed.
    """

    def __init__(
        self,
        field_cardinalities: Optional[List[int]] = None,
        embedding_dim: int = 16,
        mlp_layers: Optional[List[int]] = None,
        lr: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 64,
        seed: Optional[int] = 42,
    ) -> None:
        self.field_cardinalities = field_cardinalities or []
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers or [64, 32]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

        # FM 1st order linear weights per field
        self.fm_1st_weights: List[np.ndarray] = []
        self.fm_1st_bias: float = 0.0

        # FM 2nd order / Deep shared embeddings per field
        self.embeddings: List[np.ndarray] = []

        # Deep MLP
        self.mlp_weights: List[np.ndarray] = []
        self.mlp_biases: List[np.ndarray] = []
        self.deep_head_w: Optional[np.ndarray] = None
        self.deep_head_b: float = 0.0

        self.is_fitted: bool = False

    def fit(
        self,
        X_sparse: np.ndarray,
        y: np.ndarray,
    ) -> "DeepFM":
        """Fit DeepFM on categorical sparse field matrix."""
        rng = np.random.RandomState(self.seed)
        X_arr = np.asarray(X_sparse, dtype=np.int32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples, n_fields = X_arr.shape

        if not self.field_cardinalities:
            self.field_cardinalities = [
                int(np.max(X_arr[:, col])) + 1 for col in range(n_fields)
            ]

        # 1. FM 1st Order Linear Weights
        self.fm_1st_weights = [
            (rng.randn(card, 1) * 0.01).astype(np.float32)
            for card in self.field_cardinalities
        ]
        self.fm_1st_bias = 0.0

        # 2. Shared Embeddings
        self.embeddings = [
            (rng.randn(card, self.embedding_dim) * 0.05).astype(np.float32)
            for card in self.field_cardinalities
        ]

        # 3. Deep MLP
        deep_in = n_fields * self.embedding_dim
        self.mlp_weights = []
        self.mlp_biases = []
        cur_in = deep_in

        for units in self.mlp_layers:
            w = (rng.randn(cur_in, units) * np.sqrt(2.0 / cur_in)).astype(np.float32)
            bias_vec: np.ndarray = np.zeros(units, dtype=np.float32)
            self.mlp_weights.append(w)
            self.mlp_biases.append(bias_vec)
            cur_in = units

        self.deep_head_w = (rng.randn(cur_in, 1) * np.sqrt(2.0 / cur_in)).astype(
            np.float32
        )
        self.deep_head_b = 0.0

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
                b_X = X_arr[idx]
                b_y = y_arr[idx]

                # 1. FM 1st Order Forward
                y_fm1: np.ndarray = np.full(
                    (batch_n, 1), self.fm_1st_bias, dtype=np.float32
                )
                for f in range(n_fields):
                    y_fm1 += self.fm_1st_weights[f][b_X[:, f]]

                # 2. FM 2nd Order Forward
                field_embs = np.stack(
                    [self.embeddings[f][b_X[:, f]] for f in range(n_fields)], axis=1
                )
                sum_emb = np.sum(field_embs, axis=1)
                sum_sq_emb = np.sum(field_embs**2, axis=1)
                sq_sum_emb = sum_emb**2
                y_fm2 = 0.5 * np.sum(sq_sum_emb - sum_sq_emb, axis=1, keepdims=True)

                # 3. Deep Forward
                deep_in_act = field_embs.reshape(batch_n, -1)
                mlp_acts = [deep_in_act]
                mlp_pre = []
                cur = deep_in_act

                for w, bias in zip(self.mlp_weights, self.mlp_biases):
                    z = np.dot(cur, w) + bias
                    mlp_pre.append(z)
                    cur = np.maximum(0.0, z)
                    mlp_acts.append(cur)

                y_deep = np.dot(cur, self.deep_head_w) + self.deep_head_b

                # Joint Logits
                logits = y_fm1 + y_fm2 + y_deep
                preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

                # Backward BCE
                dlogits = (preds - b_y) / batch_n

                # 1. Update FM 1st
                self.fm_1st_bias -= float(self.lr * np.sum(dlogits))
                for f in range(n_fields):
                    np.add.at(self.fm_1st_weights[f], b_X[:, f], -self.lr * dlogits)

                # 2. FM 2nd Gradients
                dfm2_embs = dlogits[:, :, None] * (sum_emb[:, None, :] - field_embs)

                # 3. Deep Gradients
                ddeep_head_w = np.dot(cur.T, dlogits)
                self.deep_head_w -= self.lr * ddeep_head_w
                self.deep_head_b -= float(self.lr * np.sum(dlogits))

                dcur = np.dot(dlogits, self.deep_head_w.T)
                for layer_idx in reversed(range(len(self.mlp_weights))):
                    dz = dcur * (mlp_pre[layer_idx] > 0).astype(np.float32)
                    dw = np.dot(mlp_acts[layer_idx].T, dz)
                    db = np.sum(dz, axis=0)
                    dcur = np.dot(dz, self.mlp_weights[layer_idx].T)
                    self.mlp_weights[layer_idx] -= self.lr * dw
                    self.mlp_biases[layer_idx] -= self.lr * db

                ddeep_embs = dcur.reshape(batch_n, n_fields, self.embedding_dim)

                # Total Embedding Gradients
                d_total_embs = dfm2_embs + ddeep_embs
                for f in range(n_fields):
                    np.add.at(
                        self.embeddings[f], b_X[:, f], -self.lr * d_total_embs[:, f, :]
                    )

        self.is_fitted = True
        return self

    def predict_proba(self, X_sparse: np.ndarray) -> np.ndarray:
        """Predict CTR / interaction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X_arr = np.asarray(X_sparse, dtype=np.int32)
        n_samples, n_fields = X_arr.shape

        y_fm1: np.ndarray = np.full((n_samples, 1), self.fm_1st_bias, dtype=np.float32)
        for f in range(n_fields):
            y_fm1 += self.fm_1st_weights[f][X_arr[:, f]]

        field_embs = np.stack(
            [self.embeddings[f][X_arr[:, f]] for f in range(n_fields)], axis=1
        )
        sum_emb = np.sum(field_embs, axis=1)
        sum_sq_emb = np.sum(field_embs**2, axis=1)
        sq_sum_emb = sum_emb**2
        y_fm2 = 0.5 * np.sum(sq_sum_emb - sum_sq_emb, axis=1, keepdims=True)

        cur = field_embs.reshape(n_samples, -1)
        for w, bias in zip(self.mlp_weights, self.mlp_biases):
            cur = np.maximum(0.0, np.dot(cur, w) + bias)

        y_deep = np.dot(cur, self.deep_head_w) + self.deep_head_b
        logits = y_fm1 + y_fm2 + y_deep
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return preds.ravel()

    def predict(self, X_sparse: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classification labels."""
        return (self.predict_proba(X_sparse) >= threshold).astype(np.int32)
