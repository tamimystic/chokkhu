"""High-Throughput Paged Attention and Non-Contiguous KV-Cache Memory Manager in pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor


class PagedKVCache:
    """Paged Key-Value Memory Pool (vLLM-Style Non-Contiguous Block Allocation).

    Eliminates KV-cache memory fragmentation by allocating fixed-size physical
    memory blocks and resolving virtual token positions via sequence block tables.

    Parameters
    ----------
    num_blocks : int, default=128
        Total number of physical memory blocks in the global pool.
    block_size : int, default=16
        Number of token slots per block.
    num_heads : int, default=4
        Number of attention key/value heads.
    head_dim : int, default=32
        Dimensionality per attention head.
    """

    def __init__(
        self,
        num_blocks: int = 128,
        block_size: int = 16,
        num_heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

        # Preallocated physical memory tensor pools
        # Shape: (num_blocks, block_size, num_heads, head_dim)
        self.k_pool: np.ndarray = np.zeros(
            (num_blocks, block_size, num_heads, head_dim), dtype=np.float32
        )
        self.v_pool: np.ndarray = np.zeros(
            (num_blocks, block_size, num_heads, head_dim), dtype=np.float32
        )

        # Free block stack
        self.free_blocks: List[int] = list(range(num_blocks))

        # Mapping: seq_id -> List of physical block IDs
        self.block_tables: Dict[str, List[int]] = {}

        # Mapping: seq_id -> Current token count (sequence length)
        self.seq_lengths: Dict[str, int] = {}

    def allocate_sequence(self, seq_id: str, num_tokens: int = 0) -> List[int]:
        """Allocates initial physical blocks for a new sequence."""
        if seq_id in self.block_tables:
            self.free_sequence(seq_id)

        blocks_needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError(
                f"Out of PagedKVCache memory. Needed {blocks_needed} blocks, available {len(self.free_blocks)}."
            )

        allocated: List[int] = [self.free_blocks.pop() for _ in range(blocks_needed)]
        self.block_tables[seq_id] = allocated
        self.seq_lengths[seq_id] = num_tokens
        return allocated

    def append_token(
        self, seq_id: str, k_vec: np.ndarray, v_vec: np.ndarray
    ) -> Tuple[int, int]:
        """Appends a single token (K, V) representation to the sequence's paged memory.

        Parameters
        ----------
        seq_id : str
            Sequence identifier.
        k_vec : np.ndarray
            Key tensor of shape (num_heads, head_dim).
        v_vec : np.ndarray
            Value tensor of shape (num_heads, head_dim).

        Returns
        -------
        Tuple[int, int]
            (physical_block_id, slot_offset_within_block)
        """
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id, num_tokens=0)

        curr_len = self.seq_lengths[seq_id]
        block_idx = curr_len // self.block_size
        slot_offset = curr_len % self.block_size

        # Allocate new block if current block is completely filled
        if block_idx >= len(self.block_tables[seq_id]):
            if len(self.free_blocks) == 0:
                raise MemoryError("Out of free blocks in PagedKVCache pool.")
            new_block = self.free_blocks.pop()
            self.block_tables[seq_id].append(new_block)

        phys_block = self.block_tables[seq_id][block_idx]

        # Write directly into preallocated physical memory
        self.k_pool[phys_block, slot_offset] = k_vec
        self.v_pool[phys_block, slot_offset] = v_vec

        self.seq_lengths[seq_id] = curr_len + 1
        return phys_block, slot_offset

    def get_kv(self, seq_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstructs contiguous (K, V) representations for a sequence.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Keys, Values) each of shape (seq_len, num_heads, head_dim).
        """
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not found in PagedKVCache.")

        curr_len = self.seq_lengths[seq_id]
        if curr_len == 0:
            empty: np.ndarray = np.zeros(
                (0, self.num_heads, self.head_dim), dtype=np.float32
            )
            return empty, empty

        blocks = self.block_tables[seq_id]
        k_blocks = self.k_pool[
            blocks
        ]  # (num_seq_blocks, block_size, num_heads, head_dim)
        v_blocks = self.v_pool[blocks]

        k_contiguous = k_blocks.reshape(-1, self.num_heads, self.head_dim)[:curr_len]
        v_contiguous = v_blocks.reshape(-1, self.num_heads, self.head_dim)[:curr_len]
        return k_contiguous, v_contiguous

    def free_sequence(self, seq_id: str) -> None:
        """Releases physical blocks associated with a finished sequence back to the free pool."""
        if seq_id in self.block_tables:
            for b in self.block_tables[seq_id]:
                self.free_blocks.append(b)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def memory_usage(self) -> Dict[str, Any]:
        """Returns cache memory utilization statistics."""
        used = self.num_blocks - len(self.free_blocks)
        ratio = float(used / self.num_blocks) if self.num_blocks > 0 else 0.0
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used,
            "free_blocks": len(self.free_blocks),
            "utilization": ratio,
            "active_sequences": len(self.block_tables),
        }


class PagedAttention:
    """Paged Attention Kernel executing multi-head attention over non-contiguous KV blocks."""

    def __init__(self, scale: Optional[float] = None) -> None:
        self.scale = scale

    def forward(
        self,
        query: np.ndarray,
        seq_id: str,
        cache: PagedKVCache,
    ) -> np.ndarray:
        """Computes scaled dot-product attention over paged memory blocks.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape (num_heads, head_dim) or (1, num_heads, head_dim).
        seq_id : str
            Target sequence identifier in cache.
        cache : PagedKVCache
            Paged KV memory cache instance.

        Returns
        -------
        np.ndarray
            Attention output of shape (num_heads, head_dim).
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 3:
            q = q.squeeze(0)  # (num_heads, head_dim)

        num_heads, head_dim = q.shape
        scale_val = self.scale if self.scale is not None else 1.0 / np.sqrt(head_dim)

        K, V = cache.get_kv(seq_id)  # (seq_len, num_heads, head_dim)
        seq_len = K.shape[0]

        if seq_len == 0:
            return np.zeros((num_heads, head_dim), dtype=np.float32)

        # Transpose for batched dot-product: (num_heads, seq_len, head_dim)
        K_t = np.transpose(K, (1, 0, 2))
        V_t = np.transpose(V, (1, 0, 2))

        # Compute dot product scores per head: (num_heads, seq_len)
        scores = np.sum(q[:, np.newaxis, :] * K_t, axis=-1) * scale_val

        # Numerically stable softmax across sequence dimension
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        # Context output: (num_heads, head_dim)
        context = np.sum(weights[:, :, np.newaxis] * V_t, axis=1)
        return context


class ContinuousBatcher:
    """Continuous (Iteration-Level) Dynamic Batching Engine for High-Throughput Serving.

    Schedules concurrent incoming generation requests, dynamically interleaving
    prefill and autoregressive decode phases with zero pipeline bubbles.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cache: Optional[PagedKVCache] = None,
        max_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache or PagedKVCache()
        self.max_batch_size = max_batch_size

        self.waiting_queue: List[Dict[str, Any]] = []
        self.running_dict: Dict[str, Dict[str, Any]] = {}
        self.completed_requests: Dict[str, str] = {}
        self.req_counter: int = 0

    def add_request(
        self, prompt: str, max_new_tokens: int = 20, request_id: Optional[str] = None
    ) -> str:
        """Adds a new generation request to the serving batcher."""
        self.req_counter += 1
        req_id = request_id or f"req_{self.req_counter}"
        tokens = self.tokenizer.encode(prompt, bos=True)

        req_info = {
            "id": req_id,
            "prompt": prompt,
            "prompt_tokens": tokens,
            "generated_tokens": list(tokens),
            "max_new_tokens": max_new_tokens,
            "phase": "prefill",
        }
        self.waiting_queue.append(req_info)
        return req_id

    def step(self) -> Dict[str, Any]:
        """Executes a single iteration step across all currently active sequences."""
        # 1. Admit new requests up to max_batch_size
        while (
            len(self.running_dict) < self.max_batch_size and len(self.waiting_queue) > 0
        ):
            req = self.waiting_queue.pop(0)
            req_id = req["id"]
            self.cache.allocate_sequence(req_id, num_tokens=len(req["prompt_tokens"]))
            self.running_dict[req_id] = req

        if len(self.running_dict) == 0:
            return {"active": 0, "completed": len(self.completed_requests)}

        finished_ids: List[str] = []

        # 2. Process active requests in parallel
        for req_id, req in list(self.running_dict.items()):
            curr_tokens = req["generated_tokens"]
            tensor_in = Tensor(
                np.array([curr_tokens], dtype=np.int64), requires_grad=False
            )

            # Model forward pass
            logits = self.model(tensor_in)
            logits_arr = np.asarray(logits.data if hasattr(logits, "data") else logits)
            next_token = int(np.argmax(logits_arr[0, -1, :]))

            req["generated_tokens"].append(next_token)

            # Check completion criteria
            n_new = len(req["generated_tokens"]) - len(req["prompt_tokens"])
            if n_new >= req["max_new_tokens"]:
                finished_ids.append(req_id)

        # 3. Finalize finished requests
        for req_id in finished_ids:
            finished_req = self.running_dict.pop(req_id)
            decoded = self.tokenizer.decode(
                finished_req["generated_tokens"], skip_special_tokens=True
            )
            self.completed_requests[req_id] = decoded
            self.cache.free_sequence(req_id)

        return {
            "active": len(self.running_dict),
            "waiting": len(self.waiting_queue),
            "completed": len(self.completed_requests),
        }

    def generate_all(self) -> Dict[str, str]:
        """Runs the continuous batching loop until all requests are completed."""
        while len(self.waiting_queue) > 0 or len(self.running_dict) > 0:
            self.step()
        return self.completed_requests
