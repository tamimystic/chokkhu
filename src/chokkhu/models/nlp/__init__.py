"""Natural Language Processing, Classical IR, and Modern Sovereign Transformer Universe for Chokkhu."""

from __future__ import annotations
from .tokenizers import (
    BaseTokenizer,
    CharacterTokenizer,
    WordTokenizer,
    WhitespaceTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
)

from .classical import (
    PorterStemmer,
    TfidfVectorizer,
    BM25Retriever,
)

from .embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RoPE,
    RotaryPositionEmbedding,
    ALiBi,
)

from .recurrent import (
    RNN,
    LSTM,
    BiLSTM,
    GRU,
)

from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    MultiHeadLatentAttention,
    DeepSeekMLA,
    SlidingWindowAttention,
)

from .moe import (
    Expert,
    TopKRouter,
    MixtureOfExperts,
    MoE,
)

from .transformer_blocks import (
    RMSNorm,
    SwiGLU,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

from .architectures import (
    BERT,
    BertForMaskedLM,
    BertForSequenceClassification,
    BitLinear,
    BitNet158,
    DeBERTa,
    DebertaForSequenceClassification,
    DeepSeekMoE,
    DeepSeekV3,
    Gemma,
    GemmaForCausalLM,
    GPT,
    GPTForCausalLM,
    LLaMA,
    LlamaForCausalLM,
    MiniGPT,
    Mistral,
    MistralForCausalLM,
    Qwen2_5,
    RoBERTa,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    Seq2SeqTransformer,
    T5,
)

from .linear_attention import (
    RWKV6,
    RWKV6TimeMix,
    RetNet,
    RetNetRetention,
)

from .state_space import (
    Mamba,
    MambaSSM,
)

from .alignment import (
    DPOTrainer,
    KTOTrainer,
    ORPOTrainer,
)

from .generation import (
    GenerationConfig,
    TextGenerator,
    sample_next_token,
    SpeculativeDecoder,
    StructuredJSONDecoder,
)

__all__ = [
    # Tokenizers
    "BaseTokenizer",
    "CharacterTokenizer",
    "WordTokenizer",
    "WhitespaceTokenizer",
    "BPETokenizer",
    "WordPieceTokenizer",
    "SentencePieceTokenizer",
    # Classical & IR
    "PorterStemmer",
    "TfidfVectorizer",
    "BM25Retriever",
    # Embeddings & Encodings
    "TokenEmbedding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEmbedding",
    "RoPE",
    "RotaryPositionEmbedding",
    "ALiBi",
    # Recurrent
    "RNN",
    "LSTM",
    "BiLSTM",
    "GRU",
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "MultiHeadLatentAttention",
    "DeepSeekMLA",
    "SlidingWindowAttention",
    # Linear Attention & State Space
    "RWKV6TimeMix",
    "RWKV6",
    "RetNetRetention",
    "RetNet",
    "MambaSSM",
    "Mamba",
    # Alignment
    "DPOTrainer",
    "KTOTrainer",
    "ORPOTrainer",
    # MoE
    "Expert",
    "TopKRouter",
    "MixtureOfExperts",
    "MoE",
    # Blocks
    "RMSNorm",
    "SwiGLU",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    # Architectures
    "BERT",
    "BertForSequenceClassification",
    "BertForMaskedLM",
    "RoBERTa",
    "RobertaForSequenceClassification",
    "RobertaForMaskedLM",
    "DeBERTa",
    "DebertaForSequenceClassification",
    "GPT",
    "MiniGPT",
    "GPTForCausalLM",
    "LLaMA",
    "LlamaForCausalLM",
    "Mistral",
    "MistralForCausalLM",
    "Gemma",
    "GemmaForCausalLM",
    "Seq2SeqTransformer",
    "T5",
    "DeepSeekMoE",
    "DeepSeekV3",
    "Qwen2_5",
    "BitLinear",
    "BitNet158",
    # Generation & Accelerated Decoding
    "GenerationConfig",
    "sample_next_token",
    "TextGenerator",
    "SpeculativeDecoder",
    "StructuredJSONDecoder",
]
