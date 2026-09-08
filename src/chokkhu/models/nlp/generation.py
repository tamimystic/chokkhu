"""Text Generation, Autoregressive Sampling, and Search Strategies (Greedy, Temperature, Top-K, Top-P)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from chokkhu.core.tensor import Tensor


@dataclass
class GenerationConfig:
    """Configuration dataclass for autoregressive text generation."""

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = 3
    pad_token_id: Optional[int] = 0
    do_sample: bool = True


def sample_next_token(
    logits: np.ndarray,
    generated_tokens: List[int],
    config: GenerationConfig,
) -> int:
    """Sample the next token ID from next-token logits with temperature, top-k, and nucleus top-p filtering."""
    logits = logits.copy()

    # Repetition penalty
    if config.repetition_penalty != 1.0 and generated_tokens:
        for tid in set(generated_tokens):
            if logits[tid] < 0:
                logits[tid] *= config.repetition_penalty
            else:
                logits[tid] /= config.repetition_penalty

    # Greedy search
    if not config.do_sample or config.temperature <= 0.0:
        return int(np.argmax(logits))

    # Temperature scaling
    logits = logits / max(1e-5, config.temperature)

    # Top-K Filtering
    if config.top_k > 0 and config.top_k < len(logits):
        indices_to_remove = logits < np.partition(logits, -config.top_k)[-config.top_k]
        logits[indices_to_remove] = -1e9

    # Softmax probabilities
    max_logit: float = float(np.max(logits))
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / (np.sum(exp_logits) + 1e-12)

    # Top-P (Nucleus) Filtering
    if 0.0 < config.top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > config.top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / (np.sum(probs) + 1e-12)

    # Sample from multinomial distribution
    next_token = int(np.random.choice(len(probs), p=probs))
    return next_token


class TextGenerator:
    """Autoregressive Text Generator Interface for Language Models."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text continuation given a prompt string."""
        if config is None:
            config = GenerationConfig(**kwargs)

        input_ids = self.tokenizer.encode(prompt, bos=True)
        generated = list(input_ids)

        for _ in range(config.max_new_tokens):
            curr_input = np.array([generated], dtype=np.int64)
            curr_tensor = Tensor(curr_input, requires_grad=False)

            # Model forward pass
            logits_tensor = self.model(curr_tensor)
            logits_data = logits_tensor.data  # (1, seq_len, vocab_size)
            last_logits = logits_data[0, -1, :]

            next_token = sample_next_token(last_logits, generated, config)
            generated.append(next_token)

            if config.eos_token_id is not None and next_token == config.eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)
