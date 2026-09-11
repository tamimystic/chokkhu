from __future__ import annotations

from typing import List, Optional, Sequence, Union, Tuple
import numpy as np


class AttentionRollout:
    """Attention Rollout for Mechanistic Transformer Interpretability in pure NumPy.

    Tracks the flow of information through all multi-head attention layers
    by recursive matrix multiplication incorporating identity residual skip connections:
        A_hat_l = 0.5 * A_l + 0.5 * I
        Rollout = A_hat_L @ A_hat_{L-1} @ ... @ A_hat_1
    """

    def __init__(
        self, head_reduction: str = "mean", discard_ratio: float = 0.0
    ) -> None:
        self.head_reduction = head_reduction
        self.discard_ratio = float(discard_ratio)

    def compute(self, all_layer_attentions: Sequence[np.ndarray]) -> np.ndarray:
        """Computes cumulative attention rollout matrix across transformer layers.

        Args:
            all_layer_attentions: List of attention matrices per layer.
                Each array has shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
                or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            np.ndarray: Cumulative rollout attention matrix of shape (seq_len, seq_len)
                or (batch_size, seq_len, seq_len).
        """
        if not all_layer_attentions:
            raise ValueError("all_layer_attentions cannot be empty.")

        processed_layers: List[np.ndarray] = []
        for att in all_layer_attentions:
            arr = np.asarray(att, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                if self.head_reduction == "mean":
                    arr = np.mean(arr, axis=0, keepdims=True)
                elif self.head_reduction == "max":
                    arr = np.max(arr, axis=0, keepdims=True)
                elif self.head_reduction == "min":
                    arr = np.min(arr, axis=0, keepdims=True)
                else:
                    arr = np.mean(arr, axis=0, keepdims=True)
            elif arr.ndim == 4:
                if self.head_reduction == "mean":
                    arr = np.mean(arr, axis=1)
                elif self.head_reduction == "max":
                    arr = np.max(arr, axis=1)
                elif self.head_reduction == "min":
                    arr = np.min(arr, axis=1)
                else:
                    arr = np.mean(arr, axis=1)
            else:
                raise ValueError(f"Invalid attention matrix dimension: {arr.ndim}")

            if self.discard_ratio > 0.0:
                for b in range(arr.shape[0]):
                    flat = arr[b].flatten()
                    k = int(len(flat) * self.discard_ratio)
                    if k > 0:
                        threshold = np.partition(flat, k)[k]
                        arr[b][arr[b] < threshold] = 0.0
                        row_sums = np.sum(arr[b], axis=-1, keepdims=True) + 1e-12
                        arr[b] = arr[b] / row_sums

            processed_layers.append(arr)

        batch_size, seq_len, _ = processed_layers[0].shape
        eye: np.ndarray = np.eye(seq_len, dtype=np.float64)

        rollout = 0.5 * processed_layers[0] + 0.5 * eye

        for layer_att in processed_layers[1:]:
            a_hat = 0.5 * layer_att + 0.5 * eye
            rollout = np.matmul(a_hat, rollout)

        row_sums = np.sum(rollout, axis=-1, keepdims=True) + 1e-12
        rollout = rollout / row_sums

        if rollout.shape[0] == 1:
            return rollout[0]
        return rollout


class DirectLogitAttribution:
    """Direct Logit Attribution (DLA) for mechanistic residual stream decomposition.

    Projects intermediate hidden layer representations directly to the vocabulary
    space using the unembedding matrix W_U:
        logit_attribution_l = h_l @ W_U
    """

    def __init__(self, unembedding_matrix: np.ndarray) -> None:
        self.W_U = np.asarray(
            unembedding_matrix, dtype=np.float64
        )  # (hidden_dim, vocab_size)

    def attribute(
        self,
        hidden_states: Union[np.ndarray, Sequence[np.ndarray]],
        target_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """Attributes output logits to specific hidden layers or components.

        Args:
            hidden_states: Array of shape (num_layers, hidden_dim) or (hidden_dim,)
            target_token_id: Optional specific vocabulary token index.

        Returns:
            np.ndarray: Logit attribution scores per layer / component.
        """
        h = np.asarray(hidden_states, dtype=np.float64)
        if h.ndim == 1:
            logits = np.dot(h, self.W_U)
            if target_token_id is not None:
                return float(logits[target_token_id])  # type: ignore[return-value]
            return logits

        logits = np.matmul(h, self.W_U)
        if target_token_id is not None:
            return logits[:, target_token_id]
        return logits


class TopKSAE:
    """Top-K Sparse Autoencoder for Mechanistic Feature Dictionary Extraction.

    Extracts overcomplete (d_sae >> d_in), monosemantic, sparse feature representations
    from dense neural activations using exact Top-K activation sparsity:
        f(x) = TopK(ReLU((x - b_dec) @ W_enc + b_enc), k)
        x_hat = f(x) @ W_dec + b_dec
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int = 16,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize TopKSAE.

        Args:
            d_in: Input representation dimension (e.g. hidden size).
            d_sae: Dictionary expansion dimension (d_sae >> d_in).
            k: Top-k non-zero activations retained per sample.
            seed: Random seed for parameter initialization.
        """
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.k = min(int(k), self.d_sae)

        rng = np.random.default_rng(seed)
        # Initialize encoder weights: standard normal scaled
        self.W_enc = rng.normal(
            0.0, 1.0 / np.sqrt(self.d_in), size=(self.d_in, self.d_sae)
        ).astype(np.float64)
        self.b_enc: np.ndarray = np.zeros(self.d_sae, dtype=np.float64)

        # Initialize decoder weights: unit-norm columns/rows (d_sae, d_in)
        self.W_dec = rng.normal(
            0.0, 1.0 / np.sqrt(self.d_sae), size=(self.d_sae, self.d_in)
        ).astype(np.float64)
        self._normalize_decoder()

        # Decoder bias initialized to geometric center
        self.b_dec: np.ndarray = np.zeros(self.d_in, dtype=np.float64)

    def _normalize_decoder(self) -> None:
        """Ensure decoder dictionary vectors have unit Euclidean norm."""
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True) + 1e-12
        self.W_dec = self.W_dec / norms

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Compute sparse Top-K feature dictionary activations.

        Args:
            x: Input activations (batch_size, d_in) or (d_in,).

        Returns:
            f: Sparse feature activations of shape (batch_size, d_sae).
        """
        arr = np.asarray(x, dtype=np.float64)
        single_input = arr.ndim == 1
        if single_input:
            arr = arr[np.newaxis, :]

        # Centering and encoding
        centered = arr - self.b_dec
        pre_acts = centered @ self.W_enc + self.b_enc
        acts = np.maximum(0.0, pre_acts)  # ReLU

        # Exact Top-K selection per sample
        B, D = acts.shape
        f = np.zeros_like(acts)

        if self.k >= D:
            f = acts
        else:
            # Find threshold for top k elements per row
            topk_indices = np.argpartition(acts, -self.k, axis=1)[:, -self.k :]
            for b in range(B):
                row_idx = topk_indices[b]
                f[b, row_idx] = acts[b, row_idx]

        if single_input:
            return f[0]
        return f

    def decode(self, f: np.ndarray) -> np.ndarray:
        """Reconstruct dense representations from sparse feature activations.

        Args:
            f: Feature activations (batch_size, d_sae) or (d_sae,).

        Returns:
            x_hat: Reconstructed dense activations (batch_size, d_in).
        """
        arr = np.asarray(f, dtype=np.float64)
        single_input = arr.ndim == 1
        if single_input:
            arr = arr[np.newaxis, :]

        x_hat = arr @ self.W_dec + self.b_dec

        if single_input:
            return x_hat[0]
        return x_hat

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through SAE.

        Returns:
            (x_hat, f) where x_hat is reconstruction and f is sparse feature vector.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> "TopKSAE":
        """Train TopKSAE on a collection of activation vectors using Adam optimizer."""
        X = np.asarray(X, dtype=np.float64)
        N = len(X)
        self.b_dec = np.mean(X, axis=0)

        # Adam optimizer state
        m_enc, v_enc = np.zeros_like(self.W_enc), np.zeros_like(self.W_enc)
        m_benc, v_benc = np.zeros_like(self.b_enc), np.zeros_like(self.b_enc)
        m_dec, v_dec = np.zeros_like(self.W_dec), np.zeros_like(self.W_dec)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            indices = rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b = X[batch_idx]
                B = len(x_b)

                # Forward
                centered = x_b - self.b_dec
                pre_acts = centered @ self.W_enc + self.b_enc
                acts = np.maximum(0.0, pre_acts)

                # Top-K
                f = np.zeros_like(acts)
                topk_indices = np.argpartition(acts, -self.k, axis=1)[:, -self.k :]
                active_mask = np.zeros_like(acts, dtype=bool)
                for b in range(B):
                    row_idx = topk_indices[b]
                    f[b, row_idx] = acts[b, row_idx]
                    active_mask[b, row_idx] = True

                x_hat = f @ self.W_dec + self.b_dec

                # MSE Loss gradient wrt x_hat: 2 * (x_hat - x_b) / B
                grad_x_hat = 2.0 * (x_hat - x_b) / B

                # Gradients wrt W_dec: f^T @ grad_x_hat
                grad_W_dec = f.T @ grad_x_hat

                # Gradients wrt f: grad_x_hat @ W_dec^T
                grad_f = grad_x_hat @ self.W_dec.T
                # Backprop through TopK and ReLU
                grad_pre_acts = np.where((acts > 0) & active_mask, grad_f, 0.0)

                # Gradients wrt W_enc and b_enc
                grad_W_enc = centered.T @ grad_pre_acts
                grad_b_enc = np.sum(grad_pre_acts, axis=0)

                # Adam update
                t += 1
                for param, grad, m, v in [
                    (self.W_enc, grad_W_enc, m_enc, v_enc),
                    (self.b_enc, grad_b_enc, m_benc, v_benc),
                    (self.W_dec, grad_W_dec, m_dec, v_dec),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad**2)
                    m_hat = m / (1.0 - beta1**t)
                    v_hat = v / (1.0 - beta2**t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

                self._normalize_decoder()

        return self


class JumpReLU:
    """JumpReLU Sparse Autoencoder with Discontinuous Activation Thresholding.

    Formula:
        f(x) = ReLU(z) * I(z > threshold)
        where z = (x - b_dec) @ W_enc + b_enc
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        threshold: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.threshold = float(threshold)

        rng = np.random.default_rng(seed)
        self.W_enc = rng.normal(
            0.0, 1.0 / np.sqrt(self.d_in), size=(self.d_in, self.d_sae)
        ).astype(np.float64)
        self.b_enc: np.ndarray = np.zeros(self.d_sae, dtype=np.float64)
        self.W_dec = rng.normal(
            0.0, 1.0 / np.sqrt(self.d_sae), size=(self.d_sae, self.d_in)
        ).astype(np.float64)
        self.b_dec: np.ndarray = np.zeros(self.d_in, dtype=np.float64)
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True) + 1e-12
        self.W_dec = self.W_dec / norms

    def encode(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        single_input = arr.ndim == 1
        if single_input:
            arr = arr[np.newaxis, :]

        centered = arr - self.b_dec
        z = centered @ self.W_enc + self.b_enc
        # JumpReLU: only activate if z > threshold
        f = np.where(z > self.threshold, np.maximum(0.0, z), 0.0)

        if single_input:
            return f[0]
        return f

    def decode(self, f: np.ndarray) -> np.ndarray:
        arr = np.asarray(f, dtype=np.float64)
        single_input = arr.ndim == 1
        if single_input:
            arr = arr[np.newaxis, :]

        x_hat = arr @ self.W_dec + self.b_dec

        if single_input:
            return x_hat[0]
        return x_hat

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class ActivationPatching:
    """Activation Patching / Interchange Interventions for Mechanistic Circuit Discovery.

    Measures causal mediation effects by substituting clean activations into corrupted model runs.
    """

    @staticmethod
    def compute_indirect_effect(
        clean_score: float,
        corrupted_score: float,
        patched_score: float,
    ) -> float:
        """Compute the Indirect Effect (IE) / Normalized Causal Restoration Ratio.

        Formula:
            IE = (patched - corrupted) / (clean - corrupted + eps)
        """
        denom = clean_score - corrupted_score
        if abs(denom) < 1e-12:
            return 0.0
        return float((patched_score - corrupted_score) / denom)

    @staticmethod
    def patch_activation(
        corrupted_act: np.ndarray,
        clean_act: np.ndarray,
        patch_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Patch clean activation values into corrupted activation tensor.

        Args:
            corrupted_act: Corrupted baseline activations.
            clean_act: Clean target activations.
            patch_mask: Boolean mask indicating which elements/positions to patch.
                        If None, completely replaces corrupted with clean.
        """
        if patch_mask is None:
            return np.copy(clean_act)
        return np.where(patch_mask, clean_act, corrupted_act)
