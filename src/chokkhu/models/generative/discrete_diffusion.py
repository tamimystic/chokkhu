"""Discrete Text Diffusion Models for Categorical Sequence Generation.

Formulated from first principles using discrete Markov transitions, absorbing
mask states, and bidirectional transformer denoising networks in pure NumPy.
"""

import numpy as np
from typing import Optional, Dict


class DiscreteTextDiffusion:
    """Discrete Absorbing-State Categorical Diffusion Model for Text and Sequences.

    Parameters
    ----------
    vocab_size : int, default=1000
        Size of categorical vocabulary (excluding or including mask token).
    max_seq_len : int, default=64
        Maximum sequence length.
    num_timesteps : int, default=50
        Total diffusion steps T.
    mask_token_id : Optional[int], default=None
        Token ID designated as the absorbing [MASK] state. Defaults to vocab_size.
    d_model : int, default=64
        Transformer embedding and hidden dimension.
    num_heads : int, default=4
        Number of self-attention heads.
    num_layers : int, default=2
        Number of transformer encoder layers.
    schedule : str, default="linear"
        Beta noise schedule: "linear" or "cosine".
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 64,
        num_timesteps: int = 50,
        mask_token_id: Optional[int] = None,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        schedule: str = "linear",
        seed: int = 42,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.num_timesteps = int(num_timesteps)
        self.mask_token_id = (
            int(vocab_size) if mask_token_id is None else int(mask_token_id)
        )
        self.total_vocab = max(self.vocab_size, self.mask_token_id + 1)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.schedule = schedule
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # 1. Diffusion Noise Schedule (alphas and alphas_cumprod)
        if schedule == "cosine":
            steps = np.linspace(0, num_timesteps, num_timesteps + 1)
            f_t = np.cos(((steps / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            self.alphas_cumprod = alphas_cumprod[1:]
        else:  # linear
            betas = np.linspace(1e-3, 0.2, num_timesteps)
            alphas = 1.0 - betas
            self.alphas_cumprod = np.cumprod(alphas)

        # 2. Neural Denoiser Network Parameters
        self.params: Dict[str, np.ndarray] = {}
        self._init_network()

    def _init_network(self) -> None:
        scale = 1.0 / np.sqrt(self.d_model)
        # Embeddings
        self.params["token_embed"] = (
            self.rng.randn(self.total_vocab, self.d_model) * scale
        )
        self.params["pos_embed"] = (
            self.rng.randn(self.max_seq_len, self.d_model) * scale
        )
        self.params["time_embed_w1"] = (
            self.rng.randn(self.d_model, self.d_model) * scale
        )
        self.params["time_embed_b1"] = np.zeros(self.d_model)
        self.params["time_embed_w2"] = (
            self.rng.randn(self.d_model, self.d_model) * scale
        )
        self.params["time_embed_b2"] = np.zeros(self.d_model)

        # Transformer Layers
        for l_idx in range(self.num_layers):
            self.params[f"l{l_idx}_q"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_k"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_v"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_out"] = (
                self.rng.randn(self.d_model, self.d_model) * scale
            )
            self.params[f"l{l_idx}_ffn1"] = (
                self.rng.randn(self.d_model, self.d_model * 2) * scale
            )
            self.params[f"l{l_idx}_ffn2"] = (
                self.rng.randn(self.d_model * 2, self.d_model) * scale
            )

        # Final Classification Head
        self.params["head_w"] = self.rng.randn(self.d_model, self.vocab_size) * scale
        self.params["head_b"] = np.zeros(self.vocab_size)

    def _get_timestep_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """Sinusoidal timestep embedding."""
        half_dim = self.d_model // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(0, half_dim) / half_dim)
        args = timesteps[:, None] * freqs[None, :]
        sin_emb = np.sin(args)
        cos_emb = np.cos(args)
        emb = np.concatenate([sin_emb, cos_emb], axis=-1)
        if self.d_model % 2 == 1:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        # MLP projection
        h = np.maximum(
            0.0,
            np.dot(emb, self.params["time_embed_w1"]) + self.params["time_embed_b1"],
        )
        return np.dot(h, self.params["time_embed_w2"]) + self.params["time_embed_b2"]

    def q_sample(
        self, x_0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward categorical corruption: mask tokens with probability (1 - alpha_t).

        Parameters
        ----------
        x_0 : np.ndarray of shape (B, S)
            Original clean token IDs.
        t : np.ndarray of shape (B,)
            Timesteps in [0, num_timesteps - 1].
        noise : Optional[np.ndarray] of shape (B, S)
            Uniform random values in [0, 1].

        Returns
        -------
        x_t : np.ndarray of shape (B, S)
            Corrupted token IDs with absorbing mask tokens.
        """
        B, S = x_0.shape
        if noise is None:
            noise = self.rng.uniform(0.0, 1.0, size=(B, S))

        # alpha_cumprod for each sample in batch
        alpha_t = self.alphas_cumprod[t][:, None]  # (B, 1)
        # With probability 1 - alpha_t, corrupt to mask token
        mask_condition = noise > alpha_t
        x_t = np.where(mask_condition, self.mask_token_id, x_0)
        return x_t

    def forward(self, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict clean token logits from corrupted tokens x_t at timestep t.

        Parameters
        ----------
        x_t : np.ndarray of shape (B, S)
            Noisy / masked token sequences.
        t : np.ndarray of shape (B,)
            Timesteps.

        Returns
        -------
        logits : np.ndarray of shape (B, S, vocab_size)
            Unnormalized predicted clean token logits.
        """
        B, S = x_t.shape
        x_t_clipped = np.clip(x_t, 0, self.total_vocab - 1)

        # 1. Embeddings
        tok_emb = self.params["token_embed"][x_t_clipped]  # (B, S, d_model)
        pos_indices: np.ndarray = np.arange(S) % self.max_seq_len
        pos_emb = self.params["pos_embed"][pos_indices]  # (S, d_model)
        time_emb = self._get_timestep_embedding(t)[:, None, :]  # (B, 1, d_model)

        h = tok_emb + pos_emb + time_emb  # (B, S, d_model)

        # 2. Transformer Encoder Layers
        head_dim = self.d_model // self.num_heads
        for l_idx in range(self.num_layers):
            # Self-Attention
            q = (
                np.dot(h, self.params[f"l{l_idx}_q"])
                .reshape(B, S, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )
            k = (
                np.dot(h, self.params[f"l{l_idx}_k"])
                .reshape(B, S, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )
            v = (
                np.dot(h, self.params[f"l{l_idx}_v"])
                .reshape(B, S, self.num_heads, head_dim)
                .swapaxes(1, 2)
            )

            scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(head_dim)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            attn_weights = np.exp(scores - scores_max)
            attn_weights = attn_weights / (
                np.sum(attn_weights, axis=-1, keepdims=True) + 1e-12
            )

            attn_out = (
                np.matmul(attn_weights, v).swapaxes(1, 2).reshape(B, S, self.d_model)
            )
            h = h + np.dot(attn_out, self.params[f"l{l_idx}_out"])
            # Layer norm
            h = (h - np.mean(h, axis=-1, keepdims=True)) / (
                np.std(h, axis=-1, keepdims=True) + 1e-6
            )

            # Feed-Forward Network
            ffn = np.maximum(0.0, np.dot(h, self.params[f"l{l_idx}_ffn1"]))
            h = h + np.dot(ffn, self.params[f"l{l_idx}_ffn2"])
            h = (h - np.mean(h, axis=-1, keepdims=True)) / (
                np.std(h, axis=-1, keepdims=True) + 1e-6
            )

        # 3. Output Logits
        logits = (
            np.dot(h, self.params["head_w"]) + self.params["head_b"]
        )  # (B, S, vocab_size)
        return logits

    def compute_loss(self, x_0: np.ndarray, t: Optional[np.ndarray] = None) -> float:
        """Compute cross-entropy ELBO variational reconstruction loss on masked tokens.

        Parameters
        ----------
        x_0 : np.ndarray of shape (B, S)
            Ground-truth token sequence.
        t : Optional[np.ndarray] of shape (B,)
            Random timesteps if None.

        Returns
        -------
        loss : float
            Average negative log-likelihood cross-entropy loss.
        """
        B, S = x_0.shape
        if t is None:
            t = self.rng.randint(0, self.num_timesteps, size=B)

        x_t = self.q_sample(x_0, t)
        logits = self.forward(x_t, t)  # (B, S, vocab_size)

        # Numerically stable cross-entropy
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(
            np.sum(np.exp(log_probs), axis=-1, keepdims=True) + 1e-12
        )

        # Target token log-probabilities
        target_log_probs = np.take_along_axis(
            log_probs, x_0[:, :, None], axis=-1
        ).squeeze(-1)

        # Focus loss on corrupted/masked tokens
        is_masked = x_t == self.mask_token_id
        if np.sum(is_masked) > 0:
            loss = -np.mean(target_log_probs[is_masked])
        else:
            loss = -np.mean(target_log_probs)
        return float(loss)

    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 16,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """Iterative reverse diffusion ancestral sampling from all [MASK] to clean tokens.

        Parameters
        ----------
        batch_size : int, default=1
            Number of sequences to generate.
        seq_len : int, default=16
            Sequence length to generate.
        num_steps : Optional[int], default=None
            Number of unmasking steps (defaults to self.num_timesteps).
        temperature : float, default=1.0
            Softmax sampling temperature.
        top_k : Optional[int], default=None
            Top-K token truncation.

        Returns
        -------
        tokens : np.ndarray of shape (batch_size, seq_len)
            Generated discrete token sequences.
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        # Initialize with all mask tokens
        x_t: np.ndarray = np.full((batch_size, seq_len), self.mask_token_id, dtype=int)
        timesteps = np.linspace(self.num_timesteps - 1, 0, num_steps, dtype=int)

        for s_idx, t_val in enumerate(timesteps):
            t_batch: np.ndarray = np.full((batch_size,), t_val, dtype=int)
            logits = self.forward(x_t, t_batch) / max(temperature, 1e-5)  # (B, S, V)

            # Top-K Filtering
            if top_k is not None and top_k > 0:
                top_k = min(top_k, self.vocab_size)
                thresh = np.partition(logits, -top_k, axis=-1)[:, :, -top_k][:, :, None]
                logits = np.where(logits < thresh, -1e9, logits)

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            # Sample predicted clean token x_0_hat for each position
            for b in range(batch_size):
                for s in range(seq_len):
                    if x_t[b, s] == self.mask_token_id:
                        p = probs[b, s]
                        p = p / np.sum(p)
                        pred_token = self.rng.choice(self.vocab_size, p=p)
                        # Decide whether to unmask at current step
                        unmask_prob = (s_idx + 1) / num_steps
                        if self.rng.uniform(0.0, 1.0) <= unmask_prob:
                            x_t[b, s] = pred_token

        # Ensure no remaining mask tokens at the end
        if np.any(x_t == self.mask_token_id):
            last_logits = self.forward(x_t, np.zeros((batch_size,), dtype=int))
            best_tokens = np.argmax(last_logits, axis=-1)
            x_t = np.where(x_t == self.mask_token_id, best_tokens, x_t)

        return x_t
