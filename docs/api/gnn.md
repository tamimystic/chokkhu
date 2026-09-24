# Graph Neural Networks (GNNs) API Reference

The `chokkhu.models.gnn` module implements message-passing spatial, spectral, relational, and geometric graph neural network architectures from first principles.

---

## 1. Spatial Message-Passing GNNs

### Graph Convolutional Network (`GCN`)
Spectral first-order approximation of Chebyshev graph convolutions:

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right), \quad \tilde{A} = A + I_N$$

```python
import numpy as np
from chokkhu.models.gnn import GCN

# 5 nodes, 16 features per node
node_feats = np.random.randn(5, 16)
adj_matrix = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

gcn = GCN(in_feats=16, hidden_dim=32, out_classes=2, num_layers=2)
logits = gcn(node_feats, adj_matrix)
print(f"Node Classification Logits: {logits.shape}")  # (5, 2)
```

### Graph Attention Network (`GAT`)
Learns anisotropic attention coefficients across neighborhood edges:

$$\alpha_{i,j} = \frac{\exp\left( \text{LeakyReLU}\left( a^T [W h_i \,\|\, W h_j] \right) \right)}{\sum_{k \in \mathcal{N}_i} \exp\left( \text{LeakyReLU}\left( a^T [W h_i \,\|\, W h_k] \right) \right)}$$

### GraphSAGE & Graph Isomorphism Network (`GIN`)
- **GraphSAGE**: Inductive neighborhood aggregation via mean/max pooling.
- **GIN**: Maximally expressive 1-Weisfeiler-Lehman graph representation: $h_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$.

---

## 2. Advanced Geometric & Temporal GNNs

- **Equivariant GNN (`EGNN`)**: Guarantees $E(n)$ rotation and translation equivariance over 3D coordinates.
- **Temporal Graph Network (`TemporalGraphNetwork`)**: Continuous-time dynamic graph learning with memory modules.
- **Hypergraph Neural Network (`HGNN`)**: Models high-order complex relational interactions connecting $>2$ nodes per hyperedge.
