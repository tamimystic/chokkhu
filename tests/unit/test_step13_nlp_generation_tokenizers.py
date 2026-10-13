import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.nlp.generation import (
    GenerationConfig,
    TextGenerator,
    sample_next_token,
)
from chokkhu.models.nlp.tokenizers.word_tokenizer import WordTokenizer


def test_tokenizer_batch_encode_with_attention_mask() -> None:
    tok = WordTokenizer()
    corpus = ["hello world from chokkhu", "deep learning artificial intelligence"]
    tok.train(corpus)

    texts = ["hello world", "hello world from chokkhu"]
    input_ids, mask = tok.batch_encode(texts, padding=True, return_attention_mask=True)

    assert input_ids.shape == (2, 4)
    assert mask.shape == (2, 4)

    # First text has 2 tokens and 2 pads
    np.testing.assert_array_equal(mask[0], np.array([1, 1, 0, 0]))
    # Second text has 4 tokens and 0 pads
    np.testing.assert_array_equal(mask[1], np.array([1, 1, 1, 1]))


def test_sample_next_token_no_repeat_ngram_and_min_tokens() -> None:
    logits = np.array([1.0, 2.0, 5.0, 10.0])  # token 3 has highest logit

    # Min new tokens bans EOS (token 3)
    cfg_min = GenerationConfig(min_new_tokens=5, eos_token_id=3, do_sample=False)
    tok_selected = sample_next_token(logits, generated_tokens=[1, 2], config=cfg_min)
    assert tok_selected != 3  # token 3 banned by min_new_tokens

    # No repeat 2-gram: if prefix is [1], token 2 was already paired with 1
    cfg_ngram = GenerationConfig(no_repeat_ngram_size=2, do_sample=False)
    # Generated history has [0, 1, 2, 1]. Next token cannot be 2!
    logits_ngram = np.array(
        [1.0, 0.0, 10.0, 0.0]
    )  # token 2 has highest logit but should be banned
    tok_ngram = sample_next_token(
        logits_ngram, generated_tokens=[0, 1, 2, 1], config=cfg_ngram
    )
    assert tok_ngram != 2


def test_text_generator_min_tokens() -> None:
    tok = WordTokenizer()
    tok.train(["cat sat on mat"])

    class MockModel:
        def __call__(self, x: Tensor) -> Tensor:
            # Output logits where EOS (3) has high score
            N, L = x.shape
            out = np.zeros((N, L, tok.vocab_size))
            out[:, :, 3] = 100.0  # prefer EOS
            out[:, :, tok.token2id["cat"]] = 50.0
            return Tensor(out)

    generator = TextGenerator(model=MockModel(), tokenizer=tok)
    cfg = GenerationConfig(
        max_new_tokens=4, min_new_tokens=2, eos_token_id=3, do_sample=False
    )
    res = generator.generate("cat", config=cfg)
    assert isinstance(res, str)
