"""Sequential & Session-Based Recommendation Models.

Pure NumPy implementations of:
- SASRec: Self-Attention Sequential Recommendation (Kang & McAuley, 2018)
- GRU4Rec: Session-Based Recommendation with Recurrent Neural Networks (Hidasi et al., 2015)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class SASRec:
    r"""Self-Attention Sequential Recommendation (SASRec).

    Applies causal multi-head self-attention with learned positional embeddings
    and feed-forward networks to capture long-range dependencies in user action sequences:

    .. math::
        \mathbf{E} = [\mathbf{e}_{s_1} + \mathbf{p}_1, \dots, \mathbf{e}_{s_T} + \mathbf{p}_T]
        \mathbf{H} = 	ext{SASRecBlocks}(\mathbf{E})
        \hat{r}_{i} = \mathbf{h}_T^T \mathbf{e}_i

    Parameters
    ----------
    max_len : int, default=20
        Maximum sequence history length.
    hidden_dim : int, default=32
        Hidden embedding dimensionality.
    n_heads : int, default=2
        Number of self-attention heads.
    n_layers : int, default=2
        Number of transformer blocks.
    lr : float, default=0.01
        Learning rate.
    n_epochs : int, default=20
        Training epochs.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        max_len: int = 20,
        hidden_dim: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        lr: float = 0.01,
        n_epochs: int = 20,
        seed: Optional[int] = 42,
    ) -> None:
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed

        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}
        self.item_embeddings: Optional[np.ndarray] = None
        self.pos_embeddings: Optional[np.ndarray] = None

        self.attn_w: List[Dict[str, np.ndarray]] = []
        self.is_fitted: bool = False

    def fit(
        self,
        sessions: List[List[Union[int, str]]],
    ) -> "SASRec":
        """Train SASRec on sequence history sessions."""
        if not sessions:
            raise ValueError("Training sessions list cannot be empty.")

        self.item_map = {}
        self.reverse_item_map = {}
        for sess in sessions:
            for item in sess:
                if item not in self.item_map:
                    idx = len(self.item_map) + 1
                    self.item_map[item] = idx
                    self.reverse_item_map[idx] = item

        n_items = len(self.item_map)
        rng = np.random.RandomState(self.seed)

        self.item_embeddings = (rng.randn(n_items + 1, self.hidden_dim) * 0.05).astype(
            np.float32
        )
        self.item_embeddings[0] = 0.0
        self.pos_embeddings = (rng.randn(self.max_len, self.hidden_dim) * 0.05).astype(
            np.float32
        )

        self.attn_w = []
        for _ in range(self.n_layers):
            w_qkv = (
                rng.randn(self.hidden_dim, self.hidden_dim * 3)
                * np.sqrt(2.0 / self.hidden_dim)
            ).astype(np.float32)
            w_ffn1 = (
                rng.randn(self.hidden_dim, self.hidden_dim * 2)
                * np.sqrt(2.0 / self.hidden_dim)
            ).astype(np.float32)
            w_ffn2 = (
                rng.randn(self.hidden_dim * 2, self.hidden_dim)
                * np.sqrt(1.0 / self.hidden_dim)
            ).astype(np.float32)
            self.attn_w.append({"qkv": w_qkv, "ffn1": w_ffn1, "ffn2": w_ffn2})

        for _ in range(self.n_epochs):
            for sess in sessions:
                if len(sess) < 2:
                    continue

                seq = [self.item_map[it] for it in sess[:-1]][-self.max_len :]
                target_pos = self.item_map[sess[-1]]

                target_neg = rng.randint(1, n_items + 1)
                while target_neg == target_pos and n_items > 1:
                    target_neg = rng.randint(1, n_items + 1)

                pad_len = self.max_len - len(seq)
                padded_seq = [0] * pad_len + seq
                seq_arr = np.array(padded_seq, dtype=np.int32)

                x = self.item_embeddings[seq_arr] + self.pos_embeddings

                for layer in self.attn_w:
                    qkv = np.dot(x, layer["qkv"])
                    q, k, v = np.split(qkv, 3, axis=-1)

                    scores = np.dot(q, k.T) / np.sqrt(self.hidden_dim)
                    mask: np.ndarray = np.triu(
                        np.ones((self.max_len, self.max_len), dtype=bool), k=1
                    )
                    scores[mask] = -1e9

                    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                    attn_probs = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-9)
                    attn_out = np.dot(attn_probs, v)
                    x = x + attn_out

                    ffn_mid = np.maximum(0.0, np.dot(x, layer["ffn1"]))
                    ffn_out = np.dot(ffn_mid, layer["ffn2"])
                    x = x + ffn_out

                final_h = x[-1]

                pos_e = self.item_embeddings[target_pos]
                neg_e = self.item_embeddings[target_neg]

                pos_score = float(np.dot(final_h, pos_e))
                neg_score = float(np.dot(final_h, neg_e))

                p_pos = 1.0 / (1.0 + np.exp(-np.clip(pos_score, -30.0, 30.0)))
                p_neg = 1.0 / (1.0 + np.exp(-np.clip(neg_score, -30.0, 30.0)))

                d_pos = p_pos - 1.0
                d_neg = p_neg - 0.0

                self.item_embeddings[target_pos] -= self.lr * d_pos * final_h
                self.item_embeddings[target_neg] -= self.lr * d_neg * final_h
                d_final_h = d_pos * pos_e + d_neg * neg_e

                for idx in seq:
                    self.item_embeddings[idx] -= self.lr * 0.1 * d_final_h

        self.is_fitted = True
        return self

    def predict_next(
        self,
        session: List[Union[int, str]],
        top_k: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Predict top-K next item candidates given recent session history."""
        if not self.is_fitted or self.item_embeddings is None:
            raise RuntimeError("Model must be fitted before calling predict_next.")

        seq = [self.item_map[it] for it in session if it in self.item_map][
            -self.max_len :
        ]
        if not seq:
            return []

        pad_len = self.max_len - len(seq)
        padded_seq = [0] * pad_len + seq
        seq_arr = np.array(padded_seq, dtype=np.int32)

        x = self.item_embeddings[seq_arr] + self.pos_embeddings

        for layer in self.attn_w:
            qkv = np.dot(x, layer["qkv"])
            q, k, v = np.split(qkv, 3, axis=-1)
            scores = np.dot(q, k.T) / np.sqrt(self.hidden_dim)
            mask: np.ndarray = np.triu(
                np.ones((self.max_len, self.max_len), dtype=bool), k=1
            )
            scores[mask] = -1e9
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_probs = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-9)
            x = x + np.dot(attn_probs, v)
            ffn_mid = np.maximum(0.0, np.dot(x, layer["ffn1"]))
            x = x + np.dot(ffn_mid, layer["ffn2"])

        final_h = x[-1]
        all_item_embs = self.item_embeddings[1:]
        scores = np.dot(all_item_embs, final_h)

        top_indices = np.argsort(-scores)[:top_k]
        return [(self.reverse_item_map[i + 1], float(scores[i])) for i in top_indices]


class GRU4Rec:
    r"""Session-based Recommendation with Gated Recurrent Units (GRU4Rec).

    Models sequential item transitions in anonymous sessions using recurrent gating:

    .. math::
        \mathbf{h}_t = 	ext{GRU}(\mathbf{h}_{t-1}, \mathbf{e}_{s_t})
        \hat{y} = \mathbf{h}_T^T \mathbf{E}

    Parameters
    ----------
    hidden_dim : int, default=32
        Hidden state dimension.
    lr : float, default=0.01
        Learning rate.
    n_epochs : int, default=20
        Number of training epochs.
    seed : Optional[int], default=42
        Random seed.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 0.01,
        n_epochs: int = 20,
        seed: Optional[int] = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed

        self.item_map: Dict[Union[int, str], int] = {}
        self.reverse_item_map: Dict[int, Union[int, str]] = {}
        self.embeddings: Optional[np.ndarray] = None

        # GRU Weights: Update Gate (z), Reset Gate (r), Candidate (h)
        self.W_z: Optional[np.ndarray] = None
        self.U_z: Optional[np.ndarray] = None
        self.b_z: Optional[np.ndarray] = None

        self.W_r: Optional[np.ndarray] = None
        self.U_r: Optional[np.ndarray] = None
        self.b_r: Optional[np.ndarray] = None

        self.W_h: Optional[np.ndarray] = None
        self.U_h: Optional[np.ndarray] = None
        self.b_h: Optional[np.ndarray] = None

        self.is_fitted: bool = False

    def _init_weights(self, n_items: int) -> None:
        rng = np.random.RandomState(self.seed)
        d = self.hidden_dim
        scale = 1.0 / np.sqrt(d)

        self.embeddings = (rng.randn(n_items, d) * 0.05).astype(np.float32)

        self.W_z = (rng.randn(d, d) * scale).astype(np.float32)
        self.U_z = (rng.randn(d, d) * scale).astype(np.float32)
        self.b_z = np.zeros(d, dtype=np.float32)

        self.W_r = (rng.randn(d, d) * scale).astype(np.float32)
        self.U_r = (rng.randn(d, d) * scale).astype(np.float32)
        self.b_r = np.zeros(d, dtype=np.float32)

        self.W_h = (rng.randn(d, d) * scale).astype(np.float32)
        self.U_h = (rng.randn(d, d) * scale).astype(np.float32)
        self.b_h = np.zeros(d, dtype=np.float32)

    def fit(
        self,
        sessions: List[List[Union[int, str]]],
    ) -> "GRU4Rec":
        """Train GRU4Rec on session item sequences."""
        if not sessions:
            raise ValueError("Sessions list cannot be empty.")

        self.item_map = {}
        self.reverse_item_map = {}
        for sess in sessions:
            for item in sess:
                if item not in self.item_map:
                    idx = len(self.item_map)
                    self.item_map[item] = idx
                    self.reverse_item_map[idx] = item

        n_items = len(self.item_map)
        self._init_weights(n_items)
        rng = np.random.RandomState(self.seed)

        for _ in range(self.n_epochs):
            for sess in sessions:
                if len(sess) < 2:
                    continue

                seq_indices = [self.item_map[it] for it in sess]
                h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)

                for t in range(len(seq_indices) - 1):
                    x_t = self.embeddings[seq_indices[t]]
                    target_pos = seq_indices[t + 1]
                    target_neg = rng.randint(0, n_items)
                    while target_neg == target_pos and n_items > 1:
                        target_neg = rng.randint(0, n_items)

                    z = 1.0 / (
                        1.0
                        + np.exp(
                            -np.clip(
                                np.dot(x_t, self.W_z) + np.dot(h, self.U_z) + self.b_z,
                                -30.0,
                                30.0,
                            )
                        )
                    )
                    r = 1.0 / (
                        1.0
                        + np.exp(
                            -np.clip(
                                np.dot(x_t, self.W_r) + np.dot(h, self.U_r) + self.b_r,
                                -30.0,
                                30.0,
                            )
                        )
                    )
                    h_tilde = np.tanh(
                        np.dot(x_t, self.W_h) + np.dot(r * h, self.U_h) + self.b_h
                    )
                    h = (1.0 - z) * h + z * h_tilde

                    pos_e = self.embeddings[target_pos]
                    neg_e = self.embeddings[target_neg]
                    diff = float(np.dot(h, pos_e) - np.dot(h, neg_e))
                    coeff = float(1.0 / (1.0 + np.exp(np.clip(diff, -30.0, 30.0))))

                    self.embeddings[target_pos] += self.lr * coeff * h
                    self.embeddings[target_neg] -= self.lr * coeff * h
                    dh = coeff * (pos_e - neg_e)

                    self.embeddings[seq_indices[t]] += (
                        self.lr * 0.1 * np.dot(dh, self.W_h.T)
                    )

        self.is_fitted = True
        return self

    def predict_next(
        self,
        session: List[Union[int, str]],
        top_k: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Predict top-K next item recommendations for session."""
        if not self.is_fitted or self.embeddings is None:
            raise RuntimeError("Model must be fitted before predict_next.")

        seq = [self.item_map[it] for it in session if it in self.item_map]
        if not seq:
            return []

        h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        for idx in seq:
            x_t = self.embeddings[idx]
            z = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        np.dot(x_t, self.W_z) + np.dot(h, self.U_z) + self.b_z,
                        -30.0,
                        30.0,
                    )
                )
            )
            r = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        np.dot(x_t, self.W_r) + np.dot(h, self.U_r) + self.b_r,
                        -30.0,
                        30.0,
                    )
                )
            )
            h_tilde = np.tanh(
                np.dot(x_t, self.W_h) + np.dot(r * h, self.U_h) + self.b_h
            )
            h = (1.0 - z) * h + z * h_tilde

        scores = np.dot(self.embeddings, h)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.reverse_item_map[i], float(scores[i])) for i in top_indices]
