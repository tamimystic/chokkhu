"""Unit tests for Vector Retrieval and ANN Subsystem."""

import numpy as np
import pytest

from chokkhu.models.retrieval.dense_retriever import (
    DenseRetriever,
    reciprocal_rank_fusion,
)
from chokkhu.models.retrieval.hnsw import HNSWIndex
from chokkhu.models.retrieval.ivf_pq import IVFPQIndex
from chokkhu.models.retrieval.lsh import MinHashLSH, RandomHyperplaneLSH


def test_hnsw_basic():
    np.random.seed(42)
    dim = 16
    n_samples = 100
    X = np.random.randn(n_samples, dim).astype(np.float32)

    index = HNSWIndex(
        dim=dim, metric="euclidean", m=8, ef_construction=32, ef_search=32, seed=42
    )
    index.add(X)

    assert len(index) == n_samples
    assert "HNSWIndex" in repr(index)

    query = X[0]  # Query exact first point
    dists, ids = index.search(query, k=5)

    assert len(dists) == 5
    assert len(ids) == 5
    # The nearest item to X[0] should be index 0 with distance approx 0
    assert ids[0] == 0
    assert dists[0] == pytest.approx(0.0, abs=1e-4)


def test_hnsw_cosine_recall():
    np.random.seed(42)
    dim = 32
    n_samples = 200
    X = np.random.randn(n_samples, dim).astype(np.float32)

    index = HNSWIndex(
        dim=dim, metric="cosine", m=16, ef_construction=64, ef_search=64, seed=42
    )
    index.add(X)

    # Brute force search
    query = np.random.randn(dim).astype(np.float32)
    norm_q = np.linalg.norm(query)
    norm_X = np.linalg.norm(X, axis=1)
    cos_sims = np.dot(X, query) / (norm_X * norm_q)
    exact_top10 = set(np.argsort(1.0 - cos_sims)[:10])

    _, hnsw_top10_ids = index.search(query, k=10)
    hnsw_top10_set = set(hnsw_top10_ids)

    # High recall (>80% on random data)
    overlap = len(exact_top10.intersection(hnsw_top10_set))
    assert overlap >= 7


def test_hnsw_custom_ids_and_empty():
    index = HNSWIndex(dim=4, metric="euclidean")
    # Empty search
    dists, ids = index.search(np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(dists) == 0
    assert len(ids) == 0

    custom_ids = ["doc_a", "doc_b", "doc_c"]
    vecs = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    index.add(vecs, ids=custom_ids)

    d, found_ids = index.search([1.0, 0.1, 0.0, 0.0], k=1)
    assert found_ids[0] == "doc_a"


def test_ivf_pq_basic():
    np.random.seed(42)
    dim = 16
    n_samples = 120
    X = np.random.randn(n_samples, dim).astype(np.float32)

    # 16 dim with 4 sub-vectors = 4 dim each, 4 coarse lists
    index = IVFPQIndex(dim=dim, n_lists=4, n_subvectors=4, n_bits=4, seed=42)
    index.train(X)
    index.add(X)

    assert len(index) == n_samples
    assert index.is_trained
    assert "IVFPQIndex" in repr(index)

    # Query
    query = X[5]
    dists, ids = index.search(query, k=5, n_probe=4)
    assert len(dists) > 0
    assert len(ids) > 0
    assert 5 in ids[:3]  # Query point should be in top results


def test_lsh_random_hyperplanes():
    np.random.seed(42)
    dim = 16
    n_samples = 80
    X = np.random.randn(n_samples, dim).astype(np.float32)

    lsh = RandomHyperplaneLSH(dim=dim, n_tables=10, n_bits=8, seed=42)
    lsh.add(X)

    assert len(lsh) == n_samples
    assert "RandomHyperplaneLSH" in repr(lsh)

    dists, ids = lsh.search(X[0], k=3)
    assert len(ids) > 0
    assert ids[0] == 0


def test_lsh_minhash():
    lsh = MinHashLSH(n_permutations=128, threshold=0.5, seed=42)

    doc1 = {1, 2, 3, 4, 5, 6, 7, 8}
    doc2 = {1, 2, 3, 4, 5, 6, 7, 9}  # 7/9 = 78% Jaccard overlap with doc1
    doc3 = {20, 21, 22, 23, 24, 25}  # Disjoint

    lsh.add("doc1", doc1)
    lsh.add("doc2", doc2)
    lsh.add("doc3", doc3)

    assert len(lsh) == 3
    assert "MinHashLSH" in repr(lsh)

    sims, match_ids = lsh.search(doc1, k=2)
    assert len(match_ids) >= 1
    assert match_ids[0] == "doc1"
    assert "doc2" in match_ids


def test_dense_retriever_and_rrf():
    dim = 8
    retriever = DenseRetriever(dim=dim, backend="hnsw", metric="cosine")

    vecs = np.eye(dim, dtype=np.float32)
    docs = [f"This is document number {i}" for i in range(dim)]
    metas = [{"category": "tech" if i % 2 == 0 else "finance"} for i in range(dim)]

    retriever.add_documents(vecs, documents=docs, metadatas=metas)
    assert len(retriever) == dim
    assert "DenseRetriever" in repr(retriever)

    query = vecs[2]
    results = retriever.search(query, k=2)

    assert len(results) == 2
    assert results[0]["id"] == 2
    assert results[0]["document"] == "This is document number 2"
    assert results[0]["metadata"]["category"] == "tech"

    # Test Reciprocal Rank Fusion
    list1 = [1, 2, 3, 4]
    list2 = [2, 1, 5, 6]
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)

    assert len(fused) == 3
    # IDs 1 and 2 appear at the top in both lists
    top_ids = [item[0] for item in fused]
    assert set(top_ids[:2]) == {1, 2}
