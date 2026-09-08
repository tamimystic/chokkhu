"""Unit tests for Classical NLP and Information Retrieval (TF-IDF, BM25, Porter Stemmer)."""

from __future__ import annotations
import numpy as np

from chokkhu.models.nlp import BM25Retriever, PorterStemmer, TfidfVectorizer


def test_porter_stemmer() -> None:
    stemmer = PorterStemmer()
    assert stemmer.stem("connecting") == "connect"
    assert stemmer.stem("connections") == "connect"
    assert stemmer.stem("troubled") == "troubl"
    assert stemmer.stem("relational") == "relat"
    assert stemmer.stem("conditional") == "condit"
    assert stemmer.stem("rational") == "ration"
    assert stemmer.stem("valuable") == "valuabl" or stemmer.stem("valuable") == "valu"


def test_tfidf_vectorizer_basic() -> None:
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog again",
        "Machine learning and artificial intelligence are powerful",
    ]

    tfidf = TfidfVectorizer(stop_words="english", norm="l2")
    X = tfidf.fit_transform(corpus)

    assert X.shape[0] == 3
    assert X.shape[1] > 5
    # L2 norm of each row should be ~1.0
    row_norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(3), atol=1e-5)

    feat_names = tfidf.get_feature_names_out()
    assert "fox" in feat_names
    assert "lazy" in feat_names


def test_tfidf_ngrams_and_sublinear() -> None:
    corpus = ["deep learning models are scalable", "deep models learn fast"]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10)
    X = tfidf.fit_transform(corpus)
    assert X.shape[0] == 2
    assert X.shape[1] <= 10


def test_bm25_retrieval() -> None:
    corpus = [
        "Convolutional neural networks are effective for image classification",
        "Transformer models revolutionized natural language processing and vision",
        "Deep reinforcement learning agents master games like chess and go",
        "Object detection frameworks locate bounding boxes in photos",
    ]

    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.fit(corpus)

    # Query matching doc 1 (Transformer)
    results = bm25.query("transformer natural language processing", top_k=2)
    assert len(results) >= 1
    top_doc_idx, top_score = results[0]
    assert top_doc_idx == 1
    assert top_score > 0.0

    # Query matching doc 0 (CNN)
    results_cnn = bm25.query("convolutional networks image", top_k=2)
    assert len(results_cnn) >= 1
    assert results_cnn[0][0] == 0
