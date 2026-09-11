"""Genomic DNA/RNA k-mer Tokenizer in pure NumPy."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence


class GenomicTokenizer:
    """Genomic k-mer Tokenizer for DNA/RNA sequences in pure NumPy.

    Splits continuous nucleotide sequences into overlapping or non-overlapping $k$-mers,
    supporting reverse complements and IUPAC ambiguity substitution.

    Parameters
    ----------
    k : int, default=6
        Length of each nucleotide $k$-mer.
    stride : int, default=1
        Stride step between consecutive $k$-mers (default 1 for overlapping).
    is_rna : bool, default=False
        Whether the sequences are RNA (U instead of T).
    """

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, k: int = 6, stride: int = 1, is_rna: bool = False) -> None:
        self.k = int(k)
        self.stride = int(stride)
        self.is_rna = is_rna

        bases = ["A", "C", "G", "U"] if is_rna else ["A", "C", "G", "T"]

        # Build vocabulary
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

        # Add special tokens
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

        # Generate all 4^k permutations
        all_kmers = ["".join(p) for p in itertools.product(bases, repeat=self.k)]
        start_idx = len(self.SPECIAL_TOKENS)
        for i, kmer in enumerate(all_kmers):
            token_id = start_idx + i
            self.vocab[kmer] = token_id
            self.inv_vocab[token_id] = kmer

        self.vocab_size = len(self.vocab)

    @staticmethod
    def reverse_complement(sequence: str, is_rna: bool = False) -> str:
        """Computes reverse complement of DNA or RNA sequence string."""
        complement_map = (
            {"A": "U", "U": "A", "C": "G", "G": "C", "N": "N"}
            if is_rna
            else {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        )
        seq_upper = sequence.upper()
        rev_comp = "".join(
            complement_map.get(base, "N") for base in reversed(seq_upper)
        )
        return rev_comp

    def tokenize(self, sequence: str) -> List[str]:
        """Splits raw nucleotide string into list of k-mer tokens."""
        seq = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")
        if self.is_rna:
            seq = seq.replace("T", "U")
        else:
            seq = seq.replace("U", "T")

        kmers: List[str] = []
        for i in range(0, len(seq) - self.k + 1, self.stride):
            kmer = seq[i : i + self.k]
            kmers.append(kmer)
        return kmers

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encodes nucleotide sequence into integer token IDs."""
        kmers = self.tokenize(sequence)
        tokens = [self.vocab.get(kmer, self.vocab["[UNK]"]) for kmer in kmers]

        if add_special_tokens:
            tokens = [self.vocab["[CLS]"]] + tokens + [self.vocab["[SEP]"]]

        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.vocab["[PAD]"]] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back to k-mer sequence."""
        tokens: List[str] = []
        for tid in token_ids:
            if tid in self.inv_vocab:
                token_str = self.inv_vocab[tid]
                if skip_special_tokens and token_str in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token_str)
        return " ".join(tokens)
