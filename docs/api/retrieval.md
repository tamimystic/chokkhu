# Vector Retrieval & Approximate Nearest Neighbors API Reference

The `chokkhu.models.retrieval` module implements vector search indexes, quantization schemes, and ranking fusion methods in pure NumPy.

---

## 1. Vector Indexes

### Hierarchical Navigable Small World (`HNSWIndex`)
Constructs a multi-layer graph where lower layers contain fine-grained connections and upper layers provide long-range skip highways for $O(\log N)$ search time.

```python
import numpy as np
from chokkhu.models.retrieval import HNSWIndex

embeddings = np.random.randn(1000, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

index = HNSWIndex(dim=128, m=16, ef_construction=100, metric="cosine")
index.build(embeddings)

query = embeddings[0]
neighbor_ids, distances = index.search(query, k=5)
print(f"Nearest neighbor IDs: {neighbor_ids}")
```

### Inverted File Product Quantization (`IVFPQIndex`)
Clusters the vector space into $K$ Voronoi cells via $k$-means and quantizes sub-vector residuals into compact 8-bit byte codes for memory-efficient dense retrieval.

### Locality-Sensitive Hashing (`RandomHyperplaneLSH` / `MinHashLSH`)
Fast sublinear approximate nearest neighbor lookups for cosine similarity and Jaccard set distance.

---

## 2. Reciprocal Rank Fusion (`reciprocal_rank_fusion`)

Combines multiple ranked retrieval lists from different retrieval systems (e.g. dense + sparse):

$$\text{RRF\_Score}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k + r_m(d)}$$

```python
from chokkhu.models.retrieval import reciprocal_rank_fusion

dense_ranks = ["doc_A", "doc_B", "doc_C"]
sparse_ranks = ["doc_B", "doc_D", "doc_A"]

fused_ranks = reciprocal_rank_fusion([dense_ranks, sparse_ranks], k=60)
print(f"Fused Hybrid Ranking: {fused_ranks}")
```
