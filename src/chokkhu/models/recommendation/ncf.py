"""Neural Collaborative Filtering (NCF / NeuMF).

Pure NumPy implementation of He et al. (2017) fusing Generalized Matrix Factorization (GMF)
and Multi-Layer Perceptron (MLP) for implicit feedback recommendation.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class NeuralCollaborativeFiltering:
    r"""Neural Collaborative Filtering (NeuMF) Architecture.

    Combines linearity of Matrix Factorization with non-linear multi-layer perceptron:

    .. math::
        \phi^{	ext{GMF}} = p_u^G \odot q_i^G
        \phi^{	ext{MLP}} = a_L(W_L \dots a_1(W_1 [p_u^M, q_i^M] + b_1) \dots + b_L)
        \hat{y}_{ui} = \sigma(h^T [\phi^{	ext{GMF}}, \phi^{	ext{MLP}}] + b_{out})

    Parameters
    ----------
    n_factors_gmf : int, default=16
        Dimension of GMF latent factors.
    n_factors_mlp : int, default=32
        Dimension of initial MLP user/item embeddings.
    mlp_layers : List[int], default=[64, 32, 16]
        Hidden layer dimensions for the MLP branch.
    lr : float, default=0.01
        Learning rate.
    n_epochs : int, default=20
        Number of training epochs.
    batch_size : int, default=128
        Mini-batch size.
    n_negatives : int, default=4
        Number of negative items sampled per positive interaction.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        n_factors_gmf: int = 16,
        n_factors_mlp: int = 32,
        mlp_layers: Optional[List[int]] = None,
        lr: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 128,
        n_negatives: int = 4,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_factors_gmf = n_factors_gmf
        self.n_factors_mlp = n_factors_mlp
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.seed = seed

        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_user_map: Dict[int, Union[int, str]] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}
        self.user_pos_sets: Dict[int, Set[int]] = {}

        # GMF Parameters
        self.gmf_user_emb: Optional[np.ndarray] = None
        self.gmf_item_emb: Optional[np.ndarray] = None

        # MLP Parameters
        self.mlp_user_emb: Optional[np.ndarray] = None
        self.mlp_item_emb: Optional[np.ndarray] = None
        self.mlp_weights: List[np.ndarray] = []
        self.mlp_biases: List[np.ndarray] = []

        # Prediction Head
        self.head_w: Optional[np.ndarray] = None
        self.head_b: float = 0.0
        self.is_fitted: bool = False

    def _init_weights(self, n_users: int, n_items: int) -> None:
        rng = np.random.RandomState(self.seed)
        self.gmf_user_emb = (rng.randn(n_users, self.n_factors_gmf) * 0.05).astype(
            np.float32
        )
        self.gmf_item_emb = (rng.randn(n_items, self.n_factors_gmf) * 0.05).astype(
            np.float32
        )
        self.mlp_user_emb = (rng.randn(n_users, self.n_factors_mlp) * 0.05).astype(
            np.float32
        )
        self.mlp_item_emb = (rng.randn(n_items, self.n_factors_mlp) * 0.05).astype(
            np.float32
        )

        self.mlp_weights = []
        self.mlp_biases = []
        in_dim = self.n_factors_mlp * 2

        for out_dim in self.mlp_layers:
            w = (rng.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)).astype(np.float32)
            bias_vec: np.ndarray = np.zeros(out_dim, dtype=np.float32)
            self.mlp_weights.append(w)
            self.mlp_biases.append(bias_vec)
            in_dim = out_dim

        head_in = self.n_factors_gmf + self.mlp_layers[-1]
        self.head_w = (rng.randn(head_in, 1) * np.sqrt(2.0 / head_in)).astype(
            np.float32
        )
        self.head_b = 0.0

    def fit(
        self,
        user_ids: Union[List[Union[int, str]], np.ndarray],
        item_ids: Union[List[Union[int, str]], np.ndarray],
    ) -> "NeuralCollaborativeFiltering":
        """Train NeuMF using mini-batch SGD with Adam optimizer."""
        u_arr = list(user_ids)
        i_arr = list(item_ids)

        if len(u_arr) != len(i_arr):
            raise ValueError("user_ids and item_ids must have identical lengths.")

        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_pos_sets = {}

        for u, i in zip(u_arr, i_arr):
            if u not in self.user_map:
                u_idx = len(self.user_map)
                self.user_map[u] = u_idx
                self.reverse_user_map[u_idx] = u
                self.user_pos_sets[u_idx] = set()
            if i not in self.item_map:
                i_idx = len(self.item_map)
                self.item_map[i] = i_idx
                self.reverse_item_map[i_idx] = i

            self.user_pos_sets[self.user_map[u]].add(self.item_map[i])

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self._init_weights(n_users, n_items)

        rng = np.random.RandomState(self.seed)

        # Adam optimizer state variables
        m_head_w = np.zeros_like(self.head_w)
        v_head_w = np.zeros_like(self.head_w)
        m_head_b = 0.0
        v_head_b = 0.0

        m_mw = [np.zeros_like(w) for w in self.mlp_weights]
        v_mw = [np.zeros_like(w) for w in self.mlp_weights]
        m_mb = [np.zeros_like(b) for b in self.mlp_biases]
        v_mb = [np.zeros_like(b) for b in self.mlp_biases]

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0

        pos_u_indices = np.array([self.user_map[u] for u in u_arr], dtype=np.int32)
        pos_i_indices = np.array([self.item_map[i] for i in i_arr], dtype=np.int32)

        for _ in range(self.n_epochs):
            train_u: List[int] = []
            train_i: List[int] = []
            train_y: List[float] = []

            for u_idx, i_idx in zip(pos_u_indices, pos_i_indices):
                train_u.append(u_idx)
                train_i.append(i_idx)
                train_y.append(1.0)

                pos_set = self.user_pos_sets[u_idx]
                for _ in range(self.n_negatives):
                    neg_i = rng.randint(0, n_items)
                    while neg_i in pos_set and len(pos_set) < n_items:
                        neg_i = rng.randint(0, n_items)
                    train_u.append(u_idx)
                    train_i.append(neg_i)
                    train_y.append(0.0)

            train_u_arr = np.array(train_u, dtype=np.int32)
            train_i_arr = np.array(train_i, dtype=np.int32)
            train_y_arr = np.array(train_y, dtype=np.float32)

            perm = rng.permutation(len(train_u_arr))
            train_u_arr = train_u_arr[perm]
            train_i_arr = train_i_arr[perm]
            train_y_arr = train_y_arr[perm]

            n_batches = int(np.ceil(len(train_u_arr) / self.batch_size))

            for batch_idx in range(n_batches):
                t += 1
                start = batch_idx * self.batch_size
                end = min(len(train_u_arr), (batch_idx + 1) * self.batch_size)

                b_u = train_u_arr[start:end]
                b_i = train_i_arr[start:end]
                b_y = train_y_arr[start:end, None]
                batch_n = len(b_u)

                # Forward GMF
                gmf_u = self.gmf_user_emb[b_u]
                gmf_i = self.gmf_item_emb[b_i]
                phi_gmf = gmf_u * gmf_i

                # Forward MLP
                mlp_u = self.mlp_user_emb[b_u]
                mlp_i = self.mlp_item_emb[b_i]
                phi_mlp_in = np.concatenate([mlp_u, mlp_i], axis=1)

                mlp_acts = [phi_mlp_in]
                mlp_pre_acts = []
                cur_act = phi_mlp_in

                for w, bias in zip(self.mlp_weights, self.mlp_biases):
                    z = np.dot(cur_act, w) + bias
                    mlp_pre_acts.append(z)
                    cur_act = np.maximum(0.0, z)
                    mlp_acts.append(cur_act)

                phi_mlp = cur_act

                # Fusion
                phi_neu = np.concatenate([phi_gmf, phi_mlp], axis=1)
                logits = np.dot(phi_neu, self.head_w) + self.head_b
                preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

                # Backward BCE Loss
                dlogits = (preds - b_y) / batch_n

                # Head gradients
                dhead_w = np.dot(phi_neu.T, dlogits)
                dhead_b = float(np.sum(dlogits))
                dphi_neu = np.dot(dlogits, self.head_w.T)

                dphi_gmf = dphi_neu[:, : self.n_factors_gmf]
                dphi_mlp = dphi_neu[:, self.n_factors_gmf :]

                dgmf_u = dphi_gmf * gmf_i
                dgmf_i = dphi_gmf * gmf_u

                dcur_act = dphi_mlp
                dmw_list: List[np.ndarray] = []
                dmb_list: List[np.ndarray] = []

                for layer_idx in reversed(range(len(self.mlp_weights))):
                    dz = dcur_act * (mlp_pre_acts[layer_idx] > 0).astype(np.float32)
                    dw = np.dot(mlp_acts[layer_idx].T, dz)
                    db = np.sum(dz, axis=0)
                    dmw_list.insert(0, dw)
                    dmb_list.insert(0, db)
                    dcur_act = np.dot(dz, self.mlp_weights[layer_idx].T)

                dmlp_u = dcur_act[:, : self.n_factors_mlp]
                dmlp_i = dcur_act[:, self.n_factors_mlp :]

                # Adam updates
                m_head_w = beta1 * m_head_w + (1 - beta1) * dhead_w
                v_head_w = beta2 * v_head_w + (1 - beta2) * (dhead_w**2)
                m_hat = m_head_w / (1 - beta1**t)
                v_hat = v_head_w / (1 - beta2**t)
                self.head_w -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                m_head_b = beta1 * m_head_b + (1 - beta1) * dhead_b
                v_head_b = beta2 * v_head_b + (1 - beta2) * (dhead_b**2)
                self.head_b -= (
                    self.lr
                    * (m_head_b / (1 - beta1**t))
                    / (np.sqrt(v_head_b / (1 - beta2**t)) + eps)
                )

                for layer_idx in range(len(self.mlp_weights)):
                    m_mw[layer_idx] = (
                        beta1 * m_mw[layer_idx] + (1 - beta1) * dmw_list[layer_idx]
                    )
                    v_mw[layer_idx] = beta2 * v_mw[layer_idx] + (1 - beta2) * (
                        dmw_list[layer_idx] ** 2
                    )
                    self.mlp_weights[layer_idx] -= (
                        self.lr
                        * (m_mw[layer_idx] / (1 - beta1**t))
                        / (np.sqrt(v_mw[layer_idx] / (1 - beta2**t)) + eps)
                    )

                    m_mb[layer_idx] = (
                        beta1 * m_mb[layer_idx] + (1 - beta1) * dmb_list[layer_idx]
                    )
                    v_mb[layer_idx] = beta2 * v_mb[layer_idx] + (1 - beta2) * (
                        dmb_list[layer_idx] ** 2
                    )
                    self.mlp_biases[layer_idx] -= (
                        self.lr
                        * (m_mb[layer_idx] / (1 - beta1**t))
                        / (np.sqrt(v_mb[layer_idx] / (1 - beta2**t)) + eps)
                    )

                np.add.at(self.gmf_user_emb, b_u, -self.lr * dgmf_u)
                np.add.at(self.gmf_item_emb, b_i, -self.lr * dgmf_i)
                np.add.at(self.mlp_user_emb, b_u, -self.lr * dmlp_u)
                np.add.at(self.mlp_item_emb, b_i, -self.lr * dmlp_i)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> float:
        """Predict interaction probability for user-item pair."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(item_id)

        if u_idx is None or i_idx is None:
            return 0.5

        gmf_u = self.gmf_user_emb[u_idx : u_idx + 1]
        gmf_i = self.gmf_item_emb[i_idx : i_idx + 1]
        phi_gmf = gmf_u * gmf_i

        mlp_u = self.mlp_user_emb[u_idx : u_idx + 1]
        mlp_i = self.mlp_item_emb[i_idx : i_idx + 1]
        cur_act = np.concatenate([mlp_u, mlp_i], axis=1)

        for w, b in zip(self.mlp_weights, self.mlp_biases):
            cur_act = np.maximum(0.0, np.dot(cur_act, w) + b)

        phi_neu = np.concatenate([phi_gmf, cur_act], axis=1)
        logit = float(np.dot(phi_neu, self.head_w).ravel()[0] + self.head_b)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
        filter_interacted: bool = True,
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend top-K items for a given user."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend.")

        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            return []

        n_items = len(self.item_map)
        all_u: np.ndarray = np.full(n_items, u_idx, dtype=np.int32)
        all_i: np.ndarray = np.arange(n_items, dtype=np.int32)

        gmf_u = self.gmf_user_emb[all_u]
        gmf_i = self.gmf_item_emb[all_i]
        phi_gmf = gmf_u * gmf_i

        mlp_u = self.mlp_user_emb[all_u]
        mlp_i = self.mlp_item_emb[all_i]
        cur_act = np.concatenate([mlp_u, mlp_i], axis=1)

        for w, b in zip(self.mlp_weights, self.mlp_biases):
            cur_act = np.maximum(0.0, np.dot(cur_act, w) + b)

        phi_neu = np.concatenate([phi_gmf, cur_act], axis=1)
        logits = (np.dot(phi_neu, self.head_w) + self.head_b).ravel()
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

        if filter_interacted and u_idx in self.user_pos_sets:
            for i in self.user_pos_sets[u_idx]:
                scores[i] = -float("inf")

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.reverse_item_map[i], float(scores[i]))
            for i in top_indices
            if scores[i] != -float("inf")
        ]
