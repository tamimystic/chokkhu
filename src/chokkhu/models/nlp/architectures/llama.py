"""LLaMA (Large Language Model Meta AI) Architecture from Scratch (Touvron et al., 2023)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Linear, Module
from ..embeddings import RoPE, TokenEmbedding
from ..attention import GroupedQueryAttention
from ..transformer_blocks import RMSNorm, SwiGLU


class LLaMABlock(Module):
    """LLaMA Decoder Block with RMSNorm, RoPE, GQA, and SwiGLU."""

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        hidden_dim: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=eps)
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=eps)
        self.feed_forward = SwiGLU(
            in_features=embed_dim, hidden_features=hidden_dim, bias=False
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        norm_x = self.attn_norm(x)
        if use_cache:
            attn_res = self.attn(
                norm_x, mask=mask, is_causal=True, kv_cache=kv_cache, use_cache=True
            )
            assert isinstance(attn_res, tuple)
            attn_out, new_cache = attn_res
        else:
            attn_out_res = self.attn(
                norm_x, mask=mask, is_causal=True, kv_cache=kv_cache, use_cache=False
            )
            assert isinstance(attn_out_res, Tensor)
            attn_out = attn_out_res
            new_cache = None

        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        if use_cache and new_cache is not None:
            return out, new_cache
        return out


class LLaMA(Module, ChokkhuModel):
    """Modern LLaMA / Mistral Architecture from Scratch."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        hidden_dim: int = 1024,
        max_seq_len: int = 2048,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.rope = RoPE(dim=embed_dim // num_query_heads, max_seq_len=max_seq_len)

        self.blocks = [
            LLaMABlock(
                embed_dim=embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)

        self.norm = RMSNorm(embed_dim, eps=eps)
        self.output = Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[Tensor, Tensor]]]]:
        h = self.token_embed(input_ids)
        new_past_key_values: List[Tuple[Tensor, Tensor]] = []

        for i, block in enumerate(self.blocks):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                block_res = block(
                    h, mask=attention_mask, kv_cache=layer_cache, use_cache=True
                )
                assert isinstance(block_res, tuple)
                h, cache_out = block_res
                new_past_key_values.append(cache_out)
            else:
                block_out = block(
                    h, mask=attention_mask, kv_cache=layer_cache, use_cache=False
                )
                assert isinstance(block_out, Tensor)
                h = block_out

        normed = self.norm(h)
        N, S, D = normed.shape
        logits = self.output(normed.reshape(-1, D))
        logits_out = logits.reshape(N, S, -1)

        if use_cache:
            return logits_out, new_past_key_values
        return logits_out

    def generate(
        self,
        input_ids: Union[np.ndarray, List[int], Tensor],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = 3,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Autoregressive token generation with O(1) KV-caching and sampling decoders."""
        from ..generation import GenerationConfig, sample_next_token

        curr_tokens: np.ndarray
        if isinstance(input_ids, Tensor):
            curr_tokens = input_ids.data.astype(np.int64)
        else:
            curr_tokens = np.asarray(input_ids, dtype=np.int64)

        if curr_tokens.ndim == 1:
            curr_tokens = curr_tokens[np.newaxis, :]

        batch_size = curr_tokens.shape[0]
        all_generated: List[List[int]] = [
            list(curr_tokens[b]) for b in range(batch_size)
        ]
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            do_sample=(temperature > 0.0),
        )

        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None

        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                step_input = np.array(
                    [[tokens[-1]] for tokens in all_generated], dtype=np.int64
                )
            else:
                step_input = np.array(all_generated, dtype=np.int64)

            step_tensor = Tensor(step_input, requires_grad=False)
            if use_cache:
                fwd_res = self.forward(
                    step_tensor, past_key_values=past_key_values, use_cache=True
                )
                assert isinstance(fwd_res, tuple)
                logits, past_key_values = fwd_res
            else:
                logits_res = self.forward(step_tensor, use_cache=False)
                assert isinstance(logits_res, Tensor)
                logits = logits_res

            all_done = True
            for b in range(batch_size):
                last_logits = logits.data[b, -1, :]
                next_tok = sample_next_token(last_logits, all_generated[b], config)
                all_generated[b].append(next_tok)
                if eos_token_id is None or next_tok != eos_token_id:
                    all_done = False

            if all_done and eos_token_id is not None:
                break

        return np.array(all_generated, dtype=np.int64)


LlamaForCausalLM = LLaMA
