"""Statistical LLM Output Watermarking and Cryptographic Detection.

Formulated from first principles using deterministic pseudo-random hash partitioning
of vocabulary into green/red lists, logit bias injection, and z-score hypothesis testing.
"""

import numpy as np
from scipy import special
from typing import Dict, Any


class StatisticalTextWatermark:
    """Kirchenbauer et al. Statistical Token Watermarking & Detection Engine.

    Parameters
    ----------
    vocab_size : int, default=1000
        Size of the language model token vocabulary.
    gamma : float, default=0.5
        Fraction of vocabulary allocated to the green list (0 < gamma < 1).
    delta : float, default=2.0
        Logit bias magnitude added to green list tokens.
    hash_key : int, default=15485863
        Cryptographic prime seed key for deterministic pseudo-random hashing.
    window_size : int, default=1
        Prefix context window size h used for hashing preceding tokens.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        window_size: int = 1,
    ) -> None:
        self.vocab_size = int(vocab_size)
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.hash_key = int(hash_key)
        self.window_size = max(1, int(window_size))
        self.green_list_size = int(np.round(self.gamma * self.vocab_size))

    def _hash_prefix(self, prefix: np.ndarray) -> int:
        """Compute deterministic pseudo-random seed from context prefix."""
        prefix_arr = np.asarray(prefix, dtype=int).flatten()
        if len(prefix_arr) == 0:
            return self.hash_key

        ctx = prefix_arr[-self.window_size :]
        seed = self.hash_key
        for tok in ctx:
            # 64-bit integer mix hash
            seed = ((seed ^ int(tok)) * 1099511628211) & 0xFFFFFFFF
        return seed

    def get_greenlist_mask(self, prefix: np.ndarray) -> np.ndarray:
        """Get boolean mask of size (vocab_size,) where True indicates green list tokens.

        Parameters
        ----------
        prefix : np.ndarray
            1D array of preceding context tokens.

        Returns
        -------
        mask : np.ndarray of shape (vocab_size,), dtype=bool
            True for tokens in the green list.
        """
        seed = self._hash_prefix(prefix)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.vocab_size)
        green_tokens = perm[: self.green_list_size]

        mask: np.ndarray = np.zeros(self.vocab_size, dtype=bool)
        mask[green_tokens] = True
        return mask

    def apply_watermark_bias(
        self, logits: np.ndarray, prefix: np.ndarray
    ) -> np.ndarray:
        """Add watermark bias delta to green tokens for a single generation step.

        Parameters
        ----------
        logits : np.ndarray of shape (vocab_size,) or (1, vocab_size)
            Original unwatermarked next-token logits from language model.
        prefix : np.ndarray
            Preceding token context.

        Returns
        -------
        biased_logits : np.ndarray
            Logits with delta added to green list tokens.
        """
        mask = self.get_greenlist_mask(prefix)
        biased = logits.copy()
        if biased.ndim == 1:
            biased[mask] += self.delta
        else:
            biased[:, mask] += self.delta
        return biased

    def detect(self, tokens: np.ndarray, z_threshold: float = 3.0) -> Dict[str, Any]:
        """Detect watermark in token sequence via one-sided z-score hypothesis test.

        Parameters
        ----------
        tokens : np.ndarray of shape (T,)
            Evaluated token sequence.
        z_threshold : float, default=3.0
            Z-score detection decision boundary (typically 3.0 -> p < 0.00135).

        Returns
        -------
        result : Dict[str, Any]
            Dictionary with num_tokens, green_count, green_ratio, z_score, p_value, is_watermarked.
        """
        toks = np.asarray(tokens, dtype=int).flatten()
        T = len(toks) - self.window_size
        if T <= 0:
            return {
                "num_tokens": 0,
                "green_count": 0,
                "green_ratio": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "is_watermarked": False,
            }

        green_hits = 0
        for i in range(self.window_size, len(toks)):
            prefix = toks[:i]
            curr_tok = toks[i]
            if 0 <= curr_tok < self.vocab_size:
                mask = self.get_greenlist_mask(prefix)
                if mask[curr_tok]:
                    green_hits += 1

        expected_green = self.gamma * T
        std_green = np.sqrt(T * self.gamma * (1.0 - self.gamma))

        if std_green > 1e-12:
            z_score = (green_hits - expected_green) / std_green
        else:
            z_score = 0.0

        # One-sided asymptotic p-value using complementary error function
        p_val = float(0.5 * special.erfc(z_score / np.sqrt(2.0)))

        return {
            "num_tokens": int(T),
            "green_count": int(green_hits),
            "green_ratio": float(green_hits / T),
            "z_score": float(z_score),
            "p_value": p_val,
            "is_watermarked": bool(z_score >= z_threshold),
        }
