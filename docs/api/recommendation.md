# Recommendation Systems API Reference

The `chokkhu.models.recommendation` module provides matrix factorization, deep collaborative filtering, click-through rate (CTR) prediction, and sequential recommendation models.

---

## 1. Classical Collaborative Filtering

### SVD++ (`SVDPlusPlus`)
Extends standard SVD by incorporating implicit user feedback:

$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T \left( p_u + |I_u|^{-\frac{1}{2}} \sum_{j \in I_u} y_j \right)$$

### Implicit Alternating Least Squares (`ImplicitALS`)
Weighted matrix factorization tailored for implicit click/view feedback matrices.

---

## 2. Deep CTR & Sequential Models

### Neural Collaborative Filtering (`NeuralCollaborativeFiltering`)
Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) to capture linear and non-linear user-item interactions.

```python
import numpy as np
from chokkhu.models.recommendation import NeuralCollaborativeFiltering

model = NeuralCollaborativeFiltering(
    num_users=1000,
    num_items=500,
    latent_dim=32,
    layers=[64, 32, 16]
)

user_ids = np.array([10, 25, 30])
item_ids = np.array([105, 200, 450])
scores = model.predict(user_ids, item_ids)
```

### DeepFM & DLRM
- **DeepFM**: Integrates Factorization Machines with deep neural networks for automatic low- and high-order feature interactions.
- **DLRM**: Facebook's Deep Learning Recommendation Model with dense bottom MLPs and dot-product interaction layers.

### Self-Attentive Sequential Recommender (`SASRec`)
Uses multi-head self-attention over user historical interaction sequences to predict the next item.

## Example

```python
import numpy as np
from chokkhu.models.recommendation.matrix_factorization import MatrixFactorization

# User-item interactions
interactions = np.array([[1, 0, 3], [0, 2, 1]])
mf = MatrixFactorization(n_users=2, n_items=3, n_factors=5)
mf.fit(interactions, epochs=10)
predictions = mf.predict_all()
print("Predicted interactions:", predictions.shape)
```
