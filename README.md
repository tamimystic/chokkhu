<div align="center">

<img src="https://raw.githubusercontent.com/tamimystic/chokkhu/main/profile.jpg" width="140" height="140" style="border-radius:50%;" alt="Author Profile">

# Chokkhu

**A Sovereign, Zero-Heavy-Dependency ML, Deep Learning, Computer Vision, NLP, Audio, Generative AI, GNN, Time Series, XAI, and AutoML Ecosystem Built from First Principles.**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/chokkhu/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/chokkhu/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/tamimystic/chokkhu/blob/main/LICENSE)

> "Minimalistic Code. Sovereign Execution. Ultra-Lightweight Wheel (< 1 MB). Zero Heavy Dependencies. First Principles."

</div>

---

## Table of Contents

1. [Key Highlights & Sovereign Philosophy](#key-highlights--sovereign-philosophy)
2. [Installation](#installation)
3. [Architecture & Feature Matrix](#architecture--feature-matrix)
4. [AutoML & End-to-End Pipeline Engine](#1-automl--end-to-end-pipeline-engine)
5. [Vector Retrieval & Approximate Nearest Neighbors (Milestone 1)](#2-vector-retrieval--approximate-nearest-neighbors-ann)
6. [Recommendation Systems Universe (Milestone 2)](#3-recommendation-systems-universe)
7. [Causal Inference & Uplift Modeling (Milestone 3)](#4-causal-inference--uplift-modeling)
8. [Survival Analysis & Reliability Engineering (Milestone 4)](#5-survival-analysis--reliability-engineering)
9. [Multi-Modal Vision-Language Universe (Milestone 5)](#6-multi-modal-vision-language-universe)
10. [Scientific Machine Learning & PINNs (Milestone 6)](#7-scientific-machine-learning--pinns)
11. [Algorithmic Fairness & Bias Mitigation (Milestone 7)](#8-algorithmic-fairness--bias-mitigation)
12. [Edge AI, Quantization & Model Compression (Milestone 8)](#9-edge-ai-quantization--model-compression)
13. [3D Computer Vision, Point Clouds & NeRF / 3DGS (Milestone 9)](#10-3d-computer-vision-point-clouds--nerf--3dgs)
14. [Autonomous AI Agents & Execution Engines (Milestone 10)](#11-autonomous-ai-agents--execution-engines)
15. [Reinforcement Learning Universe (Milestone 11)](#12-reinforcement-learning-universe)
16. [Advanced Graph Neural Networks Universe (Milestone 12)](#13-advanced-graph-neural-networks-universe)
17. [Frontier LLMs, Linear Attention & Alignment (Milestone 13)](#14-frontier-llms-linear-attention--alignment)
18. [Classical ML, 2D Vision, Audio & Generative DL](#15-classical-ml-2d-vision-audio--generative-dl)
19. [Sovereign Explainable AI (XAI)](#16-sovereign-explainable-ai-xai)
20. [Next-Gen Generative AI, LoRA, Conformal & Stacking (Milestone 14)](#17-next-gen-generative-ai-lora-conformal--stacking-milestone-14)
21. [Privacy, Quantum ML, Geo-Spatial, NAS & TDA (Milestone 15)](#18-privacy-quantum-ml-geo-spatial-nas--tda-milestone-15)
22. [License & Citation](#license--citation)

---

## Key Highlights & Sovereign Philosophy

- **100% Zero-Heavy-Dependency**: Written strictly from fundamental mathematical formulations using only standard `NumPy`, `SciPy`, `Pandas`, `Matplotlib`, `Seaborn`, and `OpenCV-headless`. Zero reliance on PyTorch, TensorFlow, Scikit-Learn, Transformers, HuggingFace, SHAP, or Librosa.
- **Ultra-Lightweight Distribution**: An entire universe of production-ready machine learning, deep learning, computer vision, and LLMs packed into a single wheel under **1 Megabyte**.
- **Universal Compatibility**: Cross-platform verification across Ubuntu Linux, Windows, and macOS on Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- **Battle-Tested Quality**: 100% test pass rate with strict type checking (`mypy`), linting (`flake8`), and formatting (`black`).

---

## Installation

Install Chokkhu effortlessly via pip:

```bash
pip install --upgrade chokkhu
```

Or install locally from source:

```bash
git clone https://github.com/tamimystic/chokkhu.git
cd chokkhu
pip install -e .
```

---

## Architecture & Feature Matrix

| Subsystem | Key Components & Architectures |
| :--- | :--- |
| **AutoML & Pipelines** | `AutoTrainer`, `auto_train()`, Bayesian Optimization (GP-EI/UCB), Hyperband Tuner, Leak-free `pipeline()` |
| **Vector Retrieval** | `HNSWIndex`, `IVFPQIndex`, `RandomHyperplaneLSH`, `MinHashLSH`, `DenseRetriever`, `reciprocal_rank_fusion` |
| **Recommendation** | `SVDPlusPlus`, `ImplicitALS`, `NeuralCollaborativeFiltering`, `DeepFM`, `DLRM`, `SASRec`, `GRU4Rec` |
| **Causal & Uplift** | `PropensityScoreMatching`, `InverseProbabilityWeighting`, `DoublyRobustLearner`, `XLearner`, `TwoModelUplift`, `qini_score` |
| **Survival Analysis** | `KaplanMeierFitter`, `NelsonAalenFitter`, `CoxPHRegression`, `DeepSurv`, `concordance_index`, `integrated_brier_score` |
| **Multi-Modal Vision** | `CLIP`, `SigLIP`, `LLaVAMLPProjector`, `PerceiverResampler` |
| **Scientific ML** | `PINN`, `BurgersPINN`, `HeatPINN`, `WavePINN`, `NeuralODE` (RK4), `SymbolicRegressor` (Genetic Programming) |
| **Fairness & Ethics** | `demographic_parity_difference`, `equalized_odds_difference`, `ReweighingTransformer`, `DisparateImpactRemover`, `ThresholdOptimizer` |
| **Edge AI & Compression** | `UniformQuantizer`, `PostTrainingQuantizer` (INT8/INT4 PTQ), `QuantizedLinear`, `GlobalMagnitudePruner`, `KnowledgeDistiller` |
| **3D Vision & NeRF** | `PointNetClassifier`, `PointNetSegmenter`, `PointNet2Classifier` (FPS + Ball Query), `NeRFMLP`, `volume_render`, `GaussianSplatting3D` |
| **Autonomous AI Agents** | `ReActAgent`, `ToolRegistry`, `ChainOfThought`, `TreeOfThoughts`, `GraphOfThoughts`, `ReflexionEngine`, `MultiAgentCoordinator` |
| **Reinforcement Learning**| `DQN`, `DoubleDQN`, `DuelingDQN`, `PrioritizedReplayBuffer`, `REINFORCE`, `ActorCritic`, `PPO` (GAE), `SAC`, `LinUCBBandit`, `DecisionTransformer` |
| **Advanced GNNs** | `GCN`, `GAT`, `GraphSAGE`, `GIN`, `LaplacianPositionalEncoding`, `Graphormer`, `RGCNClassifier`, `EGNN`, `TemporalGraphNetwork`, `HGNN` |
| **Frontier LLMs** | `RWKV6` (Linear Attention), `RetNet` (Multi-Scale Retention), `Mamba` (S6 SSM), `DeepSeekV3` (MLA + MoE + MTP), `Qwen2_5`, `BitNet158` (1.58b ternary), `DPOTrainer`, `KTOTrainer`, `ORPOTrainer` |
| **2D Vision & Audio** | ResNet, ConvNeXt, SwinTransformer, DeiT, ViT, YOLO, Faster R-CNN, RetinaNet; STFT, MelSpectrogram, MFCC, Conformer, AST, Wav2Vec2 |
| **Explainable AI (XAI)** | `IntegratedGradients`, `SmoothGrad`, `DeepLIFT`, `KernelSHAP`, `PermutationImportance`, `PartialDependence` |

---

## 1. AutoML & End-to-End Pipeline Engine

### 1.1 Automated Machine Learning in One Line (`auto_train`)

```python
import chokkhu as ck
import numpy as np

# Synthetic classification dataset
X_train = np.random.randn(200, 10)
y_train = np.random.randint(0, 2, size=200)

# Run full AutoML search: profiles data, tests candidate models, performs CV ranking
result = ck.auto_train(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    time_budget_secs=30,
    metric="accuracy",
    cv=5,
    random_state=42
)

print(result.summary())
y_pred = result.predict(X_train[:10])
```

#### Parameter Breakdown: `ck.auto_train`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `X_train` | `np.ndarray` / `pd.DataFrame` | *Required* | Feature matrix with shape `(n_samples, n_features)`. |
| `y_train` | `np.ndarray` / `pd.Series` | *Required* | Target label array with shape `(n_samples,)`. |
| `task` | `str` | `"auto"` | Learning task type: `"classification"`, `"regression"`, or `"auto"` (auto-detected). |
| `time_budget_secs`| `int` / `float` | `60` | Maximum time allocation in seconds for candidate search. |
| `metric` | `str` / `None` | `None` | Optimization objective: `"accuracy"`, `"f1"`, `"roc_auc"`, `"mse"`, `"r2"`. |
| `cv` | `int` | `5` | Number of cross-validation folds for leaderboard validation. |
| `random_state` | `int` | `42` | Seed for deterministic data splitting and candidate initialization. |

---

### 1.2 Bayesian Optimization (`BayesianOptimization`)

```python
from chokkhu.automl import BayesianOptimization

# Objective function to maximize
def objective(params):
    x, y = params[0], params[1]
    return -(x**2 + y**2) + 10.0

# Define parameter search bounds [(min_x, max_x), (min_y, max_y)]
bounds = [(-5.0, 5.0), (-5.0, 5.0)]

opt = BayesianOptimization(
    func=objective,
    bounds=bounds,
    n_init=5,
    n_iter=20,
    acq="ei",
    xi=0.01,
    random_state=42
)
best_params, best_value = opt.optimize()
print(f"Optimal parameters: {best_params}, Max objective: {best_value}")
```

#### Parameter Breakdown: `BayesianOptimization`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `func` | `Callable` | *Required* | Black-box target function mapping a 1D parameter array to a scalar score. |
| `bounds` | `List[Tuple[float, float]]` | *Required* | List of `(min, max)` tuples specifying box constraints for each dimension. |
| `n_init` | `int` | `5` | Number of initial random sample points prior to Gaussian Process fitting. |
| `n_iter` | `int` | `20` | Number of sequential Bayesian optimization iterations. |
| `acq` | `str` | `"ei"` | Acquisition function choice: `"ei"` (Expected Improvement) or `"ucb"` (Upper Confidence Bound). |
| `xi` | `float` | `0.01` | Exploration-exploitation balance trade-off parameter for Expected Improvement. |
| `kappa` | `float` | `2.576` | Exploration multiplier for UCB acquisition (`mean + kappa * std`). |
| `random_state` | `int` | `42` | Pseudorandom seed for reproducible initial sampling. |

---

## 2. Vector Retrieval & Approximate Nearest Neighbors (ANN)

### 2.1 Hierarchical Navigable Small World (`HNSWIndex`)

```python
import numpy as np
from chokkhu.models.retrieval import HNSWIndex

# 128-dimensional dense vector embeddings
embeddings = np.random.randn(1000, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build sovereign HNSW index
index = HNSWIndex(
    dim=128,
    max_elements=5000,
    M=16,
    ef_construction=100,
    ef_search=50,
    metric="cosine",
    random_state=42
)
index.add(embeddings)

# Query top-5 nearest neighbors
query = embeddings[0:1]
indices, distances = index.search(query, k=5)
print(f"Nearest indices: {indices}, Distances: {distances}")
```

#### Parameter Breakdown: `HNSWIndex`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dim` | `int` | *Required* | Dimensionality $D$ of the input feature vectors. |
| `max_elements` | `int` | `10000` | Maximum vector storage capacity allocated for the index. |
| `M` | `int` | `16` | Maximum number of bi-directional connection links per node in graph layers. |
| `ef_construction` | `int` | `100` | Size of dynamic candidate list during graph construction (controls index quality). |
| `ef_search` | `int` | `50` | Size of priority search queue during query time (controls recall vs. latency). |
| `metric` | `str` | `"cosine"` | Distance metric: `"cosine"` or `"l2"`. |
| `random_state` | `int` | `42` | Random generator seed for probabilistic layer assignment. |

---

### 2.2 Inverted File with Product Quantization (`IVFPQIndex`)

```python
import numpy as np
from chokkhu.models.retrieval import IVFPQIndex

vectors = np.random.randn(2000, 64).astype(np.float32)

# Build compressed IVF-PQ index
ivfpq = IVFPQIndex(
    dim=64,
    n_clusters=16,
    n_subvectors=8,
    n_bits=8,
    n_probe=4,
    metric="l2",
    random_state=42
)
ivfpq.fit(vectors)
ivfpq.add(vectors)

query = vectors[0:1]
indices, distances = ivfpq.search(query, k=5)
print(f"IVF-PQ Top-5 matches: {indices}")
```

#### Parameter Breakdown: `IVFPQIndex`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dim` | `int` | *Required* | Total dimension of input vector (must be divisible by `n_subvectors`). |
| `n_clusters` | `int` | `16` | Number of coarse Voronoi centroids (coarse quantizer clusters). |
| `n_subvectors` | `int` | `8` | Number of low-dimensional sub-spaces $M$ for product quantization. |
| `n_bits` | `int` | `8` | Codebook bits per sub-vector ($2^{	ext{n\_bits}}$ centroids per sub-space). |
| `n_probe` | `int` | `4` | Number of nearest coarse clusters inspected during query search. |
| `metric` | `str` | `"l2"` | Distance metric for residual scoring (`"l2"` or `"cosine"`). |
| `random_state` | `int` | `42` | Deterministic seed for k-means centroid clustering. |

---

## 3. Recommendation Systems Universe

### 3.1 DeepFM (Deep Factorization Machine)

```python
import numpy as np
from chokkhu.models.recommendation import DeepFM

# Cardinalities for 3 categorical features: [100 users, 50 items, 10 categories]
cat_dims = [100, 50, 10]
model = DeepFM(
    cat_dims=cat_dims,
    num_dim=4,
    embedding_dim=16,
    dnn_hidden_dims=[64, 32],
    lr=0.001,
    dropout=0.1,
    random_state=42
)

# Training batch: categorical indices and continuous dense features
cat_inputs = np.array([[5, 12, 1], [42, 8, 3]])
num_inputs = np.random.randn(2, 4).astype(np.float32)
targets = np.array([1.0, 0.0])

# Forward pass & training step
preds = model.forward(cat_inputs, num_inputs)
loss = model.train_step(cat_inputs, num_inputs, targets)
print(f"DeepFM Predictions: {preds.squeeze()}, Loss: {loss:.4f}")
```

#### Parameter Breakdown: `DeepFM`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `cat_dims` | `List[int]` | *Required* | Vocabulary / cardinality size for each categorical sparse field. |
| `num_dim` | `int` | `0` | Number of continuous numerical dense input features. |
| `embedding_dim` | `int` | `16` | Latent embedding vector size per categorical feature field. |
| `dnn_hidden_dims`| `List[int]` | `[64, 32]` | Hidden layer dimensions for the deep neural network component. |
| `lr` | `float` | `0.001` | Learning rate for stochastic gradient optimization. |
| `dropout` | `float` | `0.1` | Dropout regularization probability applied between deep dense layers. |
| `random_state` | `int` | `42` | Random seed for Xavier weight initialization. |

---

### 3.2 SASRec (Self-Attentive Sequential Recommendation)

```python
import numpy as np
from chokkhu.models.recommendation import SASRec

# Sequence of interacted item IDs: shape (batch_size, max_seq_len)
item_sequences = np.array([
    [1, 5, 23, 45, 12],
    [3, 8, 19, 0, 0]    # 0 is padding
])

sasrec = SASRec(
    item_count=100,
    max_len=5,
    hidden_dim=32,
    num_heads=2,
    num_blocks=2,
    dropout_rate=0.1,
    lr=0.001,
    random_state=42
)

# Sequence embeddings & scores for candidate items
seq_embeddings = sasrec.forward(item_sequences)
print(f"Sequence representations shape: {seq_embeddings.shape}")
```

#### Parameter Breakdown: `SASRec`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `item_count` | `int` | *Required* | Total number of unique catalog items (including 0 padding token). |
| `max_len` | `int` | `50` | Maximum history sequence length considered per user session. |
| `hidden_dim` | `int` | `64` | Dimensionality of item embeddings and multi-head attention states. |
| `num_heads` | `int` | `2` | Number of parallel self-attention heads per transformer block. |
| `num_blocks` | `int` | `2` | Number of stacked causal self-attention transformer layers. |
| `dropout_rate` | `float` | `0.1` | Dropout probability for feedforward and attention layers. |
| `lr` | `float` | `0.001` | Learning rate for parameter updates. |
| `random_state` | `int` | `42` | Seed for model parameter initialization. |

---

## 4. Causal Inference & Uplift Modeling

### 4.1 Doubly Robust Learner (`DoublyRobustLearner`)

```python
import numpy as np
from chokkhu.models.causal import DoublyRobustLearner

# Covariates (X), binary treatment (T), and observed outcome (y)
X = np.random.randn(500, 5)
T = np.random.binomial(1, 0.5, size=500)
# Real treatment effect = +2.5
y = 1.2 * X[:, 0] + 2.5 * T + np.random.randn(500) * 0.1

dr = DoublyRobustLearner(clip_min=0.02, clip_max=0.98, random_state=42)
dr.fit(X, T, y)

# Estimate Individual Treatment Effects (ITE) and Average Treatment Effect (ATE)
ite_preds = dr.predict_ite(X[:5])
ate_estimate = dr.estimate_ate(X, T, y)
print(f"Estimated ATE: {ate_estimate:.3f} (True: 2.500)")
```

#### Parameter Breakdown: `DoublyRobustLearner`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `clip_min` | `float` | `0.01` | Lower threshold for propensity score clipping to prevent division by zero. |
| `clip_max` | `float` | `0.99` | Upper threshold for propensity score clipping. |
| `random_state` | `int` | `42` | Random seed for outcome regression and propensity estimators. |

---

### 4.2 Qini Score & Cumulative Uplift (`qini_score`)

```python
import numpy as np
from chokkhu.models.causal import qini_score, uplift_at_k

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
treatment = np.array([1, 1, 1, 0, 0, 1, 0, 0])
uplift_preds = np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2])

score = qini_score(y_true=y_true, treatment=treatment, uplift_preds=uplift_preds)
top_k_uplift = uplift_at_k(y_true=y_true, treatment=treatment, uplift_preds=uplift_preds, k=0.5)
print(f"Qini Score: {score:.4f}, Uplift@50%: {top_k_uplift:.4f}")
```

#### Parameter Breakdown: `qini_score`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `y_true` | `np.ndarray` | *Required* | Binary ground truth conversion/outcome array `(n_samples,)`. |
| `treatment` | `np.ndarray` | *Required* | Binary treatment assignment indicators `(1=treated, 0=control)`. |
| `uplift_preds` | `np.ndarray` | *Required* | Predicted uplift / treatment effect score for ordering individuals. |

---

## 5. Survival Analysis & Reliability Engineering

### 5.1 Cox Proportional Hazards Regression (`CoxPHRegression`)

```python
import numpy as np
from chokkhu.models.survival import CoxPHRegression, concordance_index

# Covariates, event durations, and binary censoring events (1=died, 0=censored)
X = np.random.randn(200, 4)
times = np.random.exponential(scale=10.0, size=200)
events = np.random.binomial(1, 0.7, size=200)

cox = CoxPHRegression(lr=0.01, max_iter=150, l2_reg=1e-4)
cox.fit(X, times, events)

# Predict partial hazard log-ratios and evaluate Harrell's C-index
risk_scores = cox.predict_risk(X)
c_idx = concordance_index(times, -risk_scores, events)
print(f"Concordance Index: {c_idx:.4f}")
```

#### Parameter Breakdown: `CoxPHRegression`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.01` | Learning rate for Newton-Raphson / gradient ascent optimization. |
| `max_iter` | `int` | `100` | Maximum optimization iterations for partial likelihood maximization. |
| `l2_reg` | `float` | `1e-4` | L2 weight regularization coefficient to prevent coefficient explosion. |
| `tol` | `float` | `1e-6` | Convergence tolerance threshold on gradient norm. |

---

### 5.2 DeepSurv (Deep Cox Proportional Hazards)

```python
import numpy as np
from chokkhu.models.survival import DeepSurv

X = np.random.randn(300, 6)
times = np.random.exponential(scale=12.0, size=300)
events = np.random.binomial(1, 0.8, size=300)

deep_surv = DeepSurv(
    input_dim=6,
    hidden_dims=[32, 16],
    lr=0.001,
    l2_reg=1e-4,
    epochs=50,
    batch_size=32,
    random_state=42
)
deep_surv.fit(X, times, events)
risks = deep_surv.predict_risk(X[:5])
print(f"DeepSurv Predicted Risk Scores: {risks.flatten()}")
```

#### Parameter Breakdown: `DeepSurv`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Number of input baseline clinical/covariate features. |
| `hidden_dims` | `List[int]` | `[32, 16]` | Hidden neural network layer architecture. |
| `lr` | `float` | `0.001` | Gradient learning rate. |
| `l2_reg` | `float` | `1e-4` | Weight decay penalty coefficient. |
| `epochs` | `int` | `100` | Number of full dataset training epochs. |
| `batch_size` | `int` | `32` | Minibatch size for stochastic updates. |
| `random_state` | `int` | `42` | Seed for weight initialization. |

---

## 6. Multi-Modal Vision-Language Universe

### 6.1 Contrastive Language-Image Pretraining (`CLIP`)

```python
import numpy as np
from chokkhu.models.multimodal import CLIP

# Create sovereign multi-modal CLIP model
clip = CLIP(
    vision_dim=128,
    text_dim=128,
    embed_dim=64,
    temperature=0.07,
    random_state=42
)

# Vision and text feature batches
img_features = np.random.randn(8, 128).astype(np.float32)
txt_features = np.random.randn(8, 128).astype(np.float32)

# Compute normalized embeddings & similarity matrix
img_emb, txt_emb, logits = clip.forward(img_features, txt_features)
loss = clip.compute_loss(img_emb, txt_emb)
print(f"CLIP Similarity Matrix Shape: {logits.shape}, InfoNCE Loss: {loss:.4f}")
```

#### Parameter Breakdown: `CLIP`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `vision_dim` | `int` | `256` | Feature dimension output from the vision backbone. |
| `text_dim` | `int` | `256` | Feature dimension output from the text encoder backbone. |
| `embed_dim` | `int` | `128` | Shared joint latent multi-modal projection dimension. |
| `temperature` | `float` | `0.07` | Temperature scaling factor $	au$ for cosine similarity logits. |
| `random_state` | `int` | `42` | Seed for linear projector weight initialization. |

---

### 6.2 Perceiver Resampler (`PerceiverResampler`)

```python
import numpy as np
from chokkhu.models.multimodal import PerceiverResampler

# Arbitrary-length visual tokens: (batch_size=2, num_tokens=196, dim=256)
visual_tokens = np.random.randn(2, 196, 256).astype(np.float32)

resampler = PerceiverResampler(
    dim=256,
    num_latents=32,
    latent_dim=512,
    num_heads=8,
    num_layers=3,
    random_state=42
)

# Resample 196 variable tokens into exactly 32 fixed LLM prefix tokens
fixed_prefix_tokens = resampler.forward(visual_tokens)
print(f"Resampled Visual Tokens Shape: {fixed_prefix_tokens.shape}") # (2, 32, 512)
```

#### Parameter Breakdown: `PerceiverResampler`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dim` | `int` | `256` | Input token embedding dimension from vision encoder. |
| `num_latents` | `int` | `64` | Fixed number of output latent query tokens (Flamingo style). |
| `latent_dim` | `int` | `512` | Dimensionality of the learned latent queries and output tokens. |
| `num_heads` | `int` | `8` | Number of multi-head cross-attention heads. |
| `num_layers` | `int` | `3` | Number of stacked Perceiver cross-attention + FFN blocks. |
| `random_state` | `int` | `42` | Random seed for learnable latent queries. |

---

## 7. Scientific Machine Learning & PINNs

### 7.1 Burgers' Equation Physics-Informed Neural Network (`BurgersPINN`)

```python
import numpy as np
from chokkhu.models.sciml import BurgersPINN

# Sovereign PINN for Burgers' PDE: u_t + u * u_x - nu * u_xx = 0
pinn = BurgersPINN(
    layers=[2, 32, 32, 1],
    nu=0.01 / np.pi,
    lr=0.001,
    random_state=42
)

# Spatio-temporal collocation points (x, t)
collocation_points = np.random.uniform(-1.0, 1.0, size=(100, 2))
bc_points = np.array([[-1.0, 0.2], [1.0, 0.2]]) # boundary conditions
bc_u = np.array([[0.0], [0.0]])

# Physics-informed training step (data loss + PDE residual loss)
total_loss, pde_residual = pinn.train_step(collocation_points, bc_points, bc_u)
print(f"PINN Total Loss: {total_loss:.5f}, PDE Residual: {pde_residual:.5f}")
```

#### Parameter Breakdown: `BurgersPINN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `layers` | `List[int]` | `[2, 32, 32, 1]`| Neural network architecture `[input_dim(x,t), hidden..., output_dim(u)]`. |
| `nu` | `float` | `0.01 / np.pi`| Fluid kinematic viscosity parameter $
u$. |
| `lr` | `float` | `0.001` | Adam/SGD optimizer learning rate. |
| `random_state` | `int` | `42` | Random seed for neural network layer initialization. |

---

### 7.2 Neural Ordinary Differential Equations (`NeuralODE`)

```python
import numpy as np
from chokkhu.models.sciml import NeuralODE

# Model continuous dynamical state trajectory dx/dt = f_theta(x, t)
node = NeuralODE(
    in_features=2,
    hidden_dim=32,
    method="rk4",
    random_state=42
)

# Initial state at t0
x0 = np.array([[1.0, 0.0], [-0.5, 0.8]])
t_span = np.linspace(0.0, 2.0, num=10)

# Integrate trajectory forward in time via 4th-order Runge-Kutta
trajectory = node.forward(x0, t_span)
print(f"Trajectory shape: {trajectory.shape}") # (10 timestamps, 2 samples, 2 dims)
```

#### Parameter Breakdown: `NeuralODE`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | `2` | State space dimensionality. |
| `hidden_dim` | `int` | `32` | Hidden layer width of derivative MLP $f_	heta$. |
| `method` | `str` | `"rk4"` | ODE solver: `"rk4"` (Runge-Kutta 4) or `"euler"`. |
| `rtol` | `float` | `1e-3` | Relative error tolerance for adaptive step sizing. |
| `atol` | `float` | `1e-5` | Absolute error tolerance for adaptive step sizing. |
| `random_state` | `int` | `42` | Weight initialization seed. |

---

## 8. Algorithmic Fairness & Bias Mitigation

### 8.1 Fairness Metrics & Disparate Impact Remediation

```python
import numpy as np
from chokkhu.fairness import (
    demographic_parity_difference,
    equalized_odds_difference,
    DisparateImpactRemover,
    ReweighingTransformer
)

# Binary predictions, true labels, and protected attribute (e.g., 0 vs 1)
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0])
y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
sensitive_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])

dp_diff = demographic_parity_difference(y_pred, sensitive_attr)
eq_odds = equalized_odds_difference(y_true, y_pred, sensitive_attr)
print(f"Demographic Parity Diff: {dp_diff:.4f}, Equalized Odds Diff: {eq_odds:.4f}")

# Pre-processing mitigation: Repair feature distributions
features = np.random.randn(8, 4)
remover = DisparateImpactRemover(repair_level=0.8)
repaired_features = remover.fit_transform(features, sensitive_attr)
```

#### Parameter Breakdown: `DisparateImpactRemover`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `repair_level` | `float` | `1.0` | Geometric repair interpolation factor $\lambda \in [0, 1]$ ($1.0 = 	ext{full repair}$). |
| `sensitive_feature_idx`| `int` / `None`| `None` | Optional column index of sensitive feature if included in $X$. |

---

## 9. Edge AI, Quantization & Model Compression

### 9.1 Uniform Post-Training Quantization (`PostTrainingQuantizer`)

```python
import numpy as np
from chokkhu.compression import PostTrainingQuantizer

# Dense FP32 weight matrix
weights_fp32 = np.random.randn(256, 128).astype(np.float32)

ptq = PostTrainingQuantizer(
    target_bits=8,
    method="uniform_symmetric",
    per_channel=True
)

# Quantize to INT8 integers and recover dequantized approximation
q_weights, scales, zero_points = ptq.quantize(weights_fp32)
recovered_fp32 = ptq.dequantize(q_weights, scales, zero_points)

quant_error = np.mean(np.abs(weights_fp32 - recovered_fp32))
print(f"Quantized Dtype: {q_weights.dtype}, Mean Absolute Quant Error: {quant_error:.6f}")
```

#### Parameter Breakdown: `PostTrainingQuantizer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `target_bits` | `int` | `8` | Target bit precision (`8` for INT8, `4` for INT4). |
| `method` | `str` | `"uniform_symmetric"`| Quantization scheme: `"uniform_symmetric"` or `"uniform_asymmetric"`. |
| `per_channel` | `bool` | `False` | Whether to compute independent scale per output channel/row. |

---

## 10. 3D Computer Vision, Point Clouds & NeRF / 3DGS

### 10.1 PointNet++ Hierarchical Point Cloud Classifier (`PointNet2Classifier`)

```python
import numpy as np
from chokkhu.models.vision_3d import PointNet2Classifier

# 3D Point cloud batch: (batch_size=2, num_points=512, xyz_coords=3)
point_clouds = np.random.randn(2, 512, 3).astype(np.float32)

pnet2 = PointNet2Classifier(
    num_classes=10,
    in_channels=3,
    random_state=42
)

# Forward pass using Set Abstraction (Farthest Point Sampling + Ball Query grouping)
logits = pnet2.forward(point_clouds)
print(f"PointNet++ Logits Shape: {logits.shape}") # (2, 10)
```

#### Parameter Breakdown: `PointNet2Classifier`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `num_classes` | `int` | *Required* | Number of output 3D object classification categories. |
| `in_channels` | `int` | `3` | Input channel dimension (3 for $(x, y, z)$, 6 for $(x, y, z, r, g, b)$). |
| `random_state` | `int` | `42` | Seed for weight initialization. |

---

### 10.2 Neural Radiance Fields (`NeRFMLP` & `volume_render`)

```python
import numpy as np
from chokkhu.models.vision_3d import NeRFMLP, volume_render, generate_camera_rays

# Generate camera rays from camera pose matrix
c2w = np.eye(4, dtype=np.float32)
rays_o, rays_d = generate_camera_rays(H=64, W=64, focal=50.0, c2w=c2w)

nerf = NeRFMLP(pos_dim=63, dir_dim=27, hidden_dim=128, num_layers=6)

# Query positional densities and directional radiance colors
sample_pts = np.random.randn(100, 63).astype(np.float32)
sample_dirs = np.random.randn(100, 27).astype(np.float32)
density, rgb = nerf.forward(sample_pts, sample_dirs)

# Differentiable volume rendering along ray z-steps
z_vals = np.linspace(2.0, 6.0, 100, dtype=np.float32)
rendered_rgb, depth_map, acc_map = volume_render(density, rgb, z_vals, rays_d[:100])
print(f"Rendered RGB shape: {rendered_rgb.shape}")
```

#### Parameter Breakdown: `NeRFMLP`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `pos_dim` | `int` | `63` | Sinusoidal frequency encoding dimension for 3D coordinate $(x,y,z)$. |
| `dir_dim` | `int` | `27` | Sinusoidal frequency encoding dimension for viewing direction $(	heta, \phi)$. |
| `hidden_dim` | `int` | `256` | Hidden layer width of the radiance MLP backbone. |
| `num_layers` | `int` | `8` | Total number of fully-connected layers. |
| `skip_layer` | `int` | `4` | Layer index to concatenate residual input coordinate embeddings. |
| `random_state` | `int` | `42` | Random initialization seed. |

---

## 11. Autonomous AI Agents & Execution Engines

### 11.1 ReAct Reasoning & Acting Agent (`ReActAgent`)

```python
from chokkhu.models.nlp.agents import Tool, ToolRegistry, ReActAgent

# 1. Register custom Python execution tools
registry = ToolRegistry()

def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return float(a) + float(b)

registry.register(Tool(
    name="calculator",
    func=add_numbers,
    description="Adds two numbers a and b together."
))

# 2. Initialize sovereign ReAct agent
agent = ReActAgent(
    tools=registry,
    max_steps=5,
    verbose=True
)

# Run deterministic agent loop
result = agent.run("Calculate the sum of 15.5 and 24.5")
print(f"Agent Final Output: {result.output}")
```

#### Parameter Breakdown: `ReActAgent`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `tools` | `ToolRegistry` / `List[Tool]` | *Required* | Registry or list of callable tools with validation schemas. |
| `max_steps` | `int` | `10` | Maximum reasoning-action-observation cycles before termination. |
| `model_callable`| `Callable` / `None` | `None` | Optional LLM completion backend (defaults to rule-based parser). |
| `verbose` | `bool` | `False` | If True, logs step-by-step Thought, Action, Action Input, and Observation. |

---

### 11.2 Tree of Thoughts Reasoning (`TreeOfThoughts`)

```python
from chokkhu.models.nlp.agents import TreeOfThoughts

# Custom thought evaluator heuristic
def custom_evaluator(thought_state: str) -> float:
    return 1.0 if "valid" in thought_state else 0.2

tot = TreeOfThoughts(
    max_depth=3,
    num_branches=3,
    search_method="bfs",
    evaluator=custom_evaluator
)

best_path = tot.solve(initial_problem="Solve constraint satisfaction riddle")
print(f"Best reasoning trajectory: {best_path}")
```

#### Parameter Breakdown: `TreeOfThoughts`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `max_depth` | `int` | `3` | Maximum tree search depth / reasoning steps. |
| `num_branches` | `int` | `3` | Number of alternative candidate thoughts generated per node. |
| `search_method`| `str` | `"dfs"` | Tree exploration strategy: `"dfs"` (Depth-First) or `"bfs"` (Breadth-First). |
| `evaluator` | `Callable` / `None` | `None` | Value function scoring candidate thought nodes. |
| `generator` | `Callable` / `None` | `None` | Generator function expanding candidate subsequent thoughts. |

---

## 12. Reinforcement Learning Universe

### 12.1 Proximal Policy Optimization with GAE (`PPO`)

```python
import numpy as np
from chokkhu.models.rl import PPO

# State dim = 4 (e.g. CartPole), Action dim = 2
ppo = PPO(
    state_dim=4,
    action_dim=2,
    actor_hidden=[64, 64],
    critic_hidden=[64, 64],
    lr_actor=0.0003,
    lr_critic=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=10,
    batch_size=32,
    entropy_coeff=0.01,
    random_state=42
)

# Collect trajectory transitions
state = np.array([0.1, -0.2, 0.05, 0.3])
action, log_prob, value = ppo.select_action(state)

# Update policy with Generalized Advantage Estimation (GAE)
states = np.random.randn(64, 4)
actions = np.random.randint(0, 2, size=64)
old_log_probs = np.random.randn(64)
rewards = np.random.randn(64)
dones = np.zeros(64, dtype=bool)
values = np.random.randn(64)

loss_dict = ppo.update(states, actions, old_log_probs, rewards, dones, values)
print(f"PPO Loss Info: {loss_dict}")
```

#### Parameter Breakdown: `PPO`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `state_dim` | `int` | *Required* | Dimensionality of the environment observation state vector. |
| `action_dim` | `int` | *Required* | Number of discrete actions available to the agent. |
| `actor_hidden` | `List[int]` | `[64, 64]` | Hidden layer sizes for the actor policy network $\pi_	heta(a\|s)$. |
| `critic_hidden`| `List[int]` | `[64, 64]` | Hidden layer sizes for the critic baseline network $V_\phi(s)$. |
| `lr_actor` | `float` | `0.0003` | Learning rate for actor parameter updates. |
| `lr_critic` | `float` | `0.001` | Learning rate for value function critic updates. |
| `gamma` | `float` | `0.99` | Discount factor $\gamma$ for future rewards. |
| `gae_lambda` | `float` | `0.95` | Factor $\lambda$ for Generalized Advantage Estimation trade-off. |
| `clip_epsilon` | `float` | `0.2` | PPO surrogate objective clipping range $(1 \pm \epsilon)$. |
| `ppo_epochs` | `int` | `10` | Optimization epochs per rollout batch. |
| `batch_size` | `int` | `64` | Minibatch size during policy updates. |
| `entropy_coeff`| `float` | `0.01` | Exploration entropy bonus coefficient. |
| `random_state` | `int` | `42` | Seed for weight initialization. |

---

### 12.2 Soft Actor-Critic (`SAC`) & Multi-Armed Bandits (`LinUCBBandit`)

```python
import numpy as np
from chokkhu.models.rl import SAC, LinUCBBandit

# 1. Soft Actor-Critic for continuous control
sac = SAC(
    state_dim=8,
    action_dim=2,
    actor_hidden=[64, 64],
    critic_hidden=[64, 64],
    lr=0.0003,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    random_state=42
)
action = sac.select_action(np.random.randn(8))

# 2. Contextual Multi-Armed Bandits with LinUCB
bandit = LinUCBBandit(n_arms=5, context_dim=10, alpha=1.0, random_state=42)
context = np.random.randn(10)
chosen_arm = bandit.select_arm(context)
bandit.update(arm=chosen_arm, context=context, reward=1.0)
print(f"SAC Action: {action}, LinUCB Chosen Arm: {chosen_arm}")
```

#### Parameter Breakdown: `SAC`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `state_dim` | `int` | *Required* | Continuous state observation space dimensionality. |
| `action_dim` | `int` | *Required* | Continuous action space dimensionality. |
| `actor_hidden` | `List[int]` | `[64, 64]` | Layer dimensions for Gaussian policy network. |
| `critic_hidden`| `List[int]` | `[64, 64]` | Layer dimensions for twin Q-critic networks. |
| `lr` | `float` | `0.0003` | Learning rate for actor, critic, and entropy optimizer. |
| `gamma` | `float` | `0.99` | Reward discount factor. |
| `tau` | `float` | `0.005` | Soft target update polyak averaging coefficient $	au$. |
| `alpha` | `float` | `0.2` | Entropy temperature balancing exploration and maximum reward. |
| `random_state` | `int` | `42` | Random seed. |

---

## 13. Advanced Graph Neural Networks Universe

### 13.1 Graphormer with Laplacian Positional Encoding (`Graphormer`)

```python
import numpy as np
from chokkhu.models.gnn import Graphormer, LaplacianPositionalEncoding

# Node features: (num_nodes=6, in_features=16)
x = np.random.randn(6, 16).astype(np.float32)
# Adjacency matrix
adj = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
], dtype=np.float32)

# Compute graph Laplacian positional eigenvectors
pe_encoder = LaplacianPositionalEncoding(k=4, normalization="sym")
lap_pe = pe_encoder.compute(adj)

graphormer = Graphormer(
    in_features=16,
    hidden_dim=32,
    out_features=8,
    num_heads=4,
    num_layers=2,
    pe_dim=4,
    dropout=0.1,
    random_state=42
)

node_embeddings = graphormer.forward(x, adj, pe=lap_pe)
print(f"Graphormer Output Node Embeddings Shape: {node_embeddings.shape}") # (6, 8)
```

#### Parameter Breakdown: `Graphormer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | *Required* | Dimension of input node feature representations. |
| `hidden_dim` | `int` | `64` | Transformer hidden dimension. |
| `out_features` | `int` | `16` | Final node embedding output dimensionality. |
| `num_heads` | `int` | `4` | Number of multi-head graph attention heads. |
| `num_layers` | `int` | `2` | Number of stacked Graphormer transformer layers. |
| `pe_dim` | `int` | `8` | Dimension of input Laplacian positional encoding vectors. |
| `dropout` | `float` | `0.1` | Attention and feedforward dropout probability. |
| `random_state` | `int` | `42` | Random initialization seed. |

---

### 13.2 Equivariant GNN (`EGNN`) & Relational GCN (`RGCNClassifier`)

```python
import numpy as np
from chokkhu.models.gnn import EGNN, RGCNClassifier

# 1. E(n) Equivariant Graph Neural Network for 3D Molecular Coordinates
h = np.random.randn(5, 16).astype(np.float32)      # Invariant scalar features
x_coords = np.random.randn(5, 3).astype(np.float32) # 3D Cartesian coordinates
edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

egnn = EGNN(
    in_node_features=16,
    hidden_dim=32,
    out_node_features=16,
    num_layers=2,
    coord_dim=3,
    random_state=42
)
h_out, x_coords_out = egnn.forward(h, x_coords, edge_index)
print(f"EGNN Equivariant 3D Coordinates Output Shape: {x_coords_out.shape}")

# 2. Relational Graph Convolutional Network (Knowledge Graphs)
edge_types = np.array([0, 1, 0, 2])
rgcn = RGCNClassifier(
    in_features=16,
    hidden_dim=32,
    out_features=4,
    num_relations=3,
    num_layers=2,
    random_state=42
)
node_logits = rgcn.forward(h, edge_index, edge_types)
print(f"R-GCN Logits Shape: {node_logits.shape}")
```

#### Parameter Breakdown: `EGNN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_node_features`| `int` | *Required* | Dimension of scalar node invariant features. |
| `hidden_dim` | `int` | `32` | Hidden feature channel width. |
| `out_node_features`| `int`| `16` | Dimension of transformed invariant node features. |
| `num_layers` | `int` | `2` | Number of equivariant message-passing layers. |
| `coord_dim` | `int` | `3` | Geometric coordinate dimension (e.g., 3 for 3D point clouds). |
| `random_state` | `int` | `42` | Weight initialization seed. |

---

## 14. Frontier LLMs, Linear Attention & Alignment

### 14.1 DeepSeek-V3 MoE with Multi-Head Latent Attention (`DeepSeekV3`)

```python
import numpy as np
from chokkhu.models.nlp import DeepSeekV3

# Token sequence: (batch_size=2, seq_len=8)
input_tokens = np.array([
    [12, 45, 89, 230, 412, 18, 92, 5],
    [7, 102, 304, 511, 22, 67, 88, 14]
])

model = DeepSeekV3(
    vocab_size=1000,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    q_lora_rank=16,
    kv_lora_rank=16,
    n_routed_experts=4,
    n_shared_experts=1,
    top_k=2,
    mtp_depth=1,
    random_state=42
)

# Forward pass computing next-token logits + auxiliary MoE load balancing loss
logits, aux_loss = model.forward(input_tokens)
print(f"DeepSeek-V3 Logits Shape: {logits.shape}, MoE Auxiliary Loss: {aux_loss:.4f}")
```

#### Parameter Breakdown: `DeepSeekV3`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `1000` | Size of token vocabulary dictionary. |
| `hidden_dim` | `int` | `128` | Model latent embedding dimension. |
| `num_layers` | `int` | `2` | Number of stacked DeepSeek transformer layers. |
| `num_heads` | `int` | `4` | Number of Multi-Head Latent Attention (MLA) heads. |
| `q_lora_rank` | `int` | `32` | Low-rank query compression dimension $d_c$. |
| `kv_lora_rank` | `int` | `32` | Low-rank key-value cache compression dimension $d_c'$. |
| `n_routed_experts`| `int` | `4` | Total number of routed Mixture-of-Experts. |
| `n_shared_experts`| `int` | `1` | Number of dedicated always-active shared experts. |
| `top_k` | `int` | `2` | Number of routed experts selected per token. |
| `mtp_depth` | `int` | `1` | Multi-Token Prediction (MTP) depth head count. |
| `random_state` | `int` | `42` | Seed for weight initialization. |

---

### 14.2 Mamba Selective State Space Model (`Mamba`) & RWKV-v6 (`RWKV6`)

```python
import numpy as np
from chokkhu.models.nlp import Mamba, RWKV6

input_ids = np.array([[15, 34, 128, 492, 10]])

# 1. Sovereign Mamba (S6 Selective State Space Model)
mamba = Mamba(
    vocab_size=1000,
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
    num_layers=2,
    random_state=42
)
mamba_logits = mamba.forward(input_ids)
print(f"Mamba S6 Logits Shape: {mamba_logits.shape}")

# 2. Sovereign RWKV-v6 (WKV linear attention with dynamic time-mixing)
rwkv = RWKV6(
    vocab_size=1000,
    hidden_dim=64,
    num_layers=2,
    head_size=16,
    random_state=42
)
rwkv_logits = rwkv.forward(input_ids)
print(f"RWKV-v6 Logits Shape: {rwkv_logits.shape}")
```

#### Parameter Breakdown: `Mamba`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `1000` | Size of vocabulary. |
| `d_model` | `int` | `128` | Model state dimension. |
| `d_state` | `int` | `16` | Latent SSM state expansion dimension $N$. |
| `d_conv` | `int` | `4` | 1D temporal convolution kernel width. |
| `expand` | `int` | `2` | Block expansion factor $E$ ($E \cdot d_{	ext{model}}$ inner width). |
| `num_layers` | `int` | `2` | Number of stacked Mamba SSM layers. |
| `random_state` | `int` | `42` | Random seed. |

---

### 14.3 Direct Preference Optimization (`DPOTrainer`)

```python
import numpy as np
from chokkhu.models.nlp import DPOTrainer, MiniGPT

# Create policy model and frozen reference model
policy = MiniGPT(vocab_size=500, hidden_dim=64, num_layers=2, num_heads=2)
ref_model = MiniGPT(vocab_size=500, hidden_dim=64, num_layers=2, num_heads=2)

dpo = DPOTrainer(
    model=policy,
    ref_model=ref_model,
    beta=0.1,
    lr=1e-4
)

# Chosen vs. Rejected completion token batches
prompt_chosen = np.array([[10, 25, 42, 100], [12, 19, 88, 204]])
prompt_rejected = np.array([[10, 25, 42, 311], [12, 19, 88, 499]])

loss, implicit_reward_chosen, implicit_reward_rejected = dpo.train_step(
    prompt_chosen=prompt_chosen,
    prompt_rejected=prompt_rejected
)
print(f"DPO Loss: {loss:.4f}, Margin: {(implicit_reward_chosen - implicit_reward_rejected).mean():.4f}")
```

#### Parameter Breakdown: `DPOTrainer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `model` | `Module` | *Required* | Trainable actor LLM policy being aligned. |
| `ref_model` | `Module` | *Required* | Frozen reference baseline model $\pi_{	ext{ref}}$. |
| `beta` | `float` | `0.1` | Temperature regularization parameter $eta$ controlling KL divergence penalty. |
| `lr` | `float` | `1e-4` | Learning rate for policy optimizer. |

---

## 15. Classical ML, 2D Vision, Audio & Generative DL

### 15.1 Generative AI: Variational Autoencoder (`VAE`) & Diffusion (`DDPM`)

```python
import numpy as np
from chokkhu.models import VAE, DDPM

# 1. Variational Autoencoder (VAE)
images_flat = np.random.randn(100, 784).astype(np.float32)
vae = VAE(input_dim=784, latent_dim=32, hidden_dim=128)
recon, mu, logvar = vae.forward(images_flat)
vae_loss = vae.compute_loss(images_flat, recon, mu, logvar)
print(f"VAE Loss: {vae_loss:.4f}")

# 2. Denoising Diffusion Probabilistic Model (DDPM)
ddpm = DDPM(input_dim=784, timesteps=1000, beta_start=1e-4, beta_end=0.02)
sampled_images = ddpm.sample(num_samples=4)
print(f"DDPM Generated Samples Shape: {sampled_images.shape}")
```

#### Parameter Breakdown: `DDPM`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Flattened data dimension. |
| `timesteps` | `int` | `1000` | Number of forward noise diffusion steps $T$. |
| `beta_start` | `float` | `1e-4` | Initial variance schedule value $eta_1$. |
| `beta_end` | `float` | `0.02` | Final variance schedule value $eta_T$. |

---

### 15.2 Speech & Audio Signal Processing (`STFT`, `MFCC`, `Conformer`)

```python
import numpy as np
from chokkhu.models import stft, melspectrogram, mfcc, Conformer

# 1-second 16kHz raw audio waveform
waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

# Extract Mel-Spectrogram and MFCCs
mel_spec = melspectrogram(waveform, sample_rate=16000, n_fft=512, hop_length=160, n_mels=80)
mfcc_features = mfcc(waveform, sample_rate=16000, n_mfcc=13)
print(f"Mel-Spectrogram Shape: {mel_spec.shape}, MFCC Shape: {mfcc_features.shape}")

# Speech Conformer (Macaron-style FFN + MHA + Depthwise Conv)
conformer = Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=128,
    num_blocks=2,
    conv_kernel_size=15
)
spec_batch = np.random.randn(2, 100, 80).astype(np.float32)
conformer_out = conformer.forward(spec_batch)
print(f"Conformer Output Shape: {conformer_out.shape}")
```

#### Parameter Breakdown: `Conformer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Feature dimension per audio time frame (e.g. 80 Mel bins). |
| `num_heads` | `int` | `4` | Number of multi-head attention heads. |
| `ffn_dim` | `int` | `128` | Feedforward intermediate dimension (Macaron structure). |
| `num_blocks` | `int` | `2` | Number of stacked Conformer blocks. |
| `conv_kernel_size`| `int` | `15` | Depthwise 1D convolution kernel receptive field. |

---

## 16. Sovereign Explainable AI (XAI)

### 16.1 Integrated Gradients & DeepLIFT (`IntegratedGradients`, `DeepLIFT`)

```python
import numpy as np
import chokkhu as ck
from chokkhu.explainability import IntegratedGradients, DeepLIFT

# Create a trained neural model
model = ck.NeuralNetwork(layers=[10, 32, 2], activation="relu")
sample = np.random.randn(1, 10)

# 1. Axiomatic feature attribution via Integrated Gradients
ig = IntegratedGradients(model=model, steps=50, baseline=None)
attributions = ig.attribute(sample, target_class=0)
print(f"Integrated Gradients Attributions Shape: {attributions.shape}")

# 2. DeepLIFT difference-from-reference conservation
dlift = DeepLIFT(model=model, baseline=np.zeros((1, 10)))
dlift_attributions = dlift.attribute(sample, target_class=0)
print(f"DeepLIFT Attributions: {dlift_attributions.flatten()}")
```

#### Parameter Breakdown: `IntegratedGradients`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `model` | `Any` | *Required* | Differentiable neural network model instance. |
| `steps` | `int` | `50` | Number of Riemann summation interpolation steps along path. |
| `baseline` | `np.ndarray` / `None` | `None` | Reference neutral baseline vector (defaults to zero array). |

---


---

## 17. Next-Gen Generative AI, LoRA, Conformal & Stacking (Milestone 14)

### 17.1 Continuous ODE Flow Matching & Rectified Flows (`FlowMatching`, `RectifiedFlow`)

```python
import numpy as np
from chokkhu.models.generative import FlowMatching, RectifiedFlow

# Generate 2D training data
X_train = np.random.randn(200, 2) * 0.5 + 2.0

# 1. Continuous Velocity Field Flow Matching with Optimal Transport
flow = FlowMatching(input_dim=2, hidden_dim=64, num_layers=3, lr=1e-3, sigma_min=1e-4)
flow.fit(X_train, epochs=20, batch_size=32)

# Sample new points using 4th-order Runge-Kutta (RK4) integration
synthetic_samples = flow.sample(num_samples=100, steps=50, method="rk4")
print(f"Generated Flow Samples Shape: {synthetic_samples.shape}")

# 2. Rectified Flow interpolation with straight-line trajectories
rect_flow = RectifiedFlow(input_dim=2, hidden_dim=32, lr=1e-3)
rect_flow.fit(X_train, epochs=10)
rect_samples = rect_flow.sample(num_samples=50, steps=20, method="midpoint")
print(f"Rectified Flow Samples Shape: {rect_samples.shape}")
```

#### Parameter Breakdown: `FlowMatching`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Dimensionality of the continuous input space $D$. |
| `hidden_dim` | `int` | `64` | Hidden layer width for the time-dependent velocity MLP network $v_\theta(x, t)$. |
| `num_layers` | `int` | `3` | Number of dense hidden layers in the velocity field approximator. |
| `lr` | `float` | `1e-3` | Gradient descent Adam/SGD learning rate during flow matching loss minimization. |
| `sigma_min` | `float` | `1e-4` | Minimum perturbation noise standard deviation for numerical stability. |

---

### 17.2 Invertible Normalizing Flows (`RealNVP`, `AffineCouplingLayer`)

```python
import numpy as np
from chokkhu.models.generative import RealNVP

# Continuous tabular / latent features
X = np.random.randn(100, 4)

# Multi-layer RealNVP with analytical Jacobian log-determinant
nvp = RealNVP(input_dim=4, num_layers=4, hidden_dim=32, lr=1e-3)
nvp.fit(X, epochs=15, batch_size=16)

# Exact log-likelihood computation
log_probs = nvp.log_prob(X)
print(f"Mean Log-Likelihood: {np.mean(log_probs):.4f}")

# Invertible sampling from latent base standard Gaussian
generated_data = nvp.sample(num_samples=50)
print(f"RealNVP Sample Shape: {generated_data.shape}")
```

#### Parameter Breakdown: `RealNVP`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Feature dimensionality $D$ (must be $\ge 2$ for split-coupling). |
| `num_layers` | `int` | `4` | Number of alternating affine coupling layers in the flow cascade. |
| `hidden_dim` | `int` | `32` | Hidden layer dimension for scale $s(x)$ and translation $t(x)$ sub-networks. |
| `lr` | `float` | `1e-3` | Learning rate for maximum likelihood estimation via backpropagation. |

---

### 17.3 Parameter-Efficient Fine-Tuning (`LoRALinear`, `LoRAAdapter`)

```python
import numpy as np
from chokkhu.models.generative import LoRALinear, LoRAAdapter

# 1. Direct Low-Rank Linear Adaptation: W_adapted = W0 + (alpha / r) * (B @ A)
linear = LoRALinear(in_features=128, out_features=64, rank=4, alpha=8.0)

x = np.random.randn(8, 128)
y = linear.forward(x)
print(f"Adapted Linear Output Shape: {y.shape}")

# Merge adapter weights directly into frozen base weight matrix for zero-latency inference
linear.merge()
y_merged = linear.forward(x)
np.testing.assert_allclose(y, y_merged, atol=1e-5)

# 2. Attach PEFT LoRA adapters to arbitrary deep architectures
model_weights = {
    "encoder.attn.q_proj": np.random.randn(64, 64),
    "encoder.attn.v_proj": np.random.randn(64, 64),
    "classifier.dense": np.random.randn(64, 10),
}
adapter = LoRAAdapter(model_weights, target_modules=["attn.q_proj", "attn.v_proj"], rank=4, alpha=8.0)
print(f"Total Base Params: {adapter.count_parameters()['base_params']}, Trainable LoRA Params: {adapter.count_parameters()['trainable_lora_params']}")
```

#### Parameter Breakdown: `LoRALinear`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | *Required* | Input feature dimension $d_{\text{in}}$. |
| `out_features` | `int` | *Required* | Output feature dimension $d_{\text{out}}$. |
| `rank` | `int` | `4` | Low-rank bottleneck dimension $r \ll \min(d_{\text{in}}, d_{\text{out}})$. |
| `alpha` | `float` | `1.0` | Scaling hyperparameter $\alpha$ where $\Delta W = \frac{\alpha}{r} B A$. |

---

### 17.4 Distribution-Free Time Series Conformal Prediction (`ConformalPredictor`)

```python
import numpy as np
from chokkhu.models.timeseries import ConformalPredictor, conformal_interval
from chokkhu.models import LinearRegression

# Generate synthetic sequential / regression dataset
X = np.linspace(0, 10, 150).reshape(-1, 1)
y = 2.5 * X.flatten() + np.sin(X.flatten()) + np.random.randn(150) * 0.5

# Base forecaster
base_model = LinearRegression()
base_model.fit(X[:80], y[:80])

# Fit conformal prediction interval with exact 95% coverage guarantee
conformal = ConformalPredictor(base_model=base_model, alpha=0.05, method="split")
conformal.calibrate(X[80:110], y[80:110])

# Predict lower and upper confidence bounds
y_pred, lower, upper = conformal.predict_interval(X[110:])
coverage = np.mean((y[110:] >= lower) & (y[110:] <= upper))
print(f"Guaranteed Coverage: >= 95%, Empirical Empirical Coverage: {coverage * 100:.1f}%")
```

#### Parameter Breakdown: `ConformalPredictor`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `base_model` | `Any` | *Required* | Any fitted regression or forecasting model providing `.predict(X)`. |
| `alpha` | `float` | `0.1` | Miscoverage significance level $\alpha \in (0, 1)$, ensuring $1-\alpha$ coverage. |
| `method` | `str` | `"split"` | Conformal method: `"split"` or `"cross_val"`. |

---

### 17.5 Matrix Profile Time Series Motif & Discord Mining (`MatrixProfile`)

```python
import numpy as np
from chokkhu.models.timeseries import MatrixProfile, find_motifs, find_discords

# Generate continuous time series with an embedded anomaly
np.random.seed(42)
t = np.linspace(0, 50, 500)
ts = np.sin(t) + np.random.randn(500) * 0.05
ts[250:270] += 4.0  # Anomaly burst

# Fast STOMP / STAMP sliding-window distance profile computation
mp = MatrixProfile(window_size=30)
profile, profile_idx = mp.fit_transform(ts)

# Identify recurring patterns (motifs) and anomalous subsequences (discords)
motifs = find_motifs(profile, profile_idx, top_k=2)
discords = find_discords(profile, top_k=1)
print(f"Top Discord (Anomaly) Subsequence Index: {discords[0]}")
```

#### Parameter Breakdown: `MatrixProfile`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `window_size` | `int` | *Required* | Subsequence window length $m$ for z-normalized Euclidean distance comparison. |
| `algorithm` | `str` | `"stomp"` | Matrix profile algorithm: `"stomp"` (fast $O(n^2)$) or `"stamp"`. |

---

### 17.6 Multi-Fidelity Bayesian Optimization & HyperBand (`BOHB`)

```python
import numpy as np
from chokkhu.automl import BOHB

# Objective function simulating hyperparameter tuning across variable budget (e.g. epochs)
def objective_fn(config, budget):
    lr = config["lr"]
    hidden = config["hidden"]
    # Simulated loss decreasing with optimal lr & budget
    loss = (np.log10(lr) + 3.0)**2 + (hidden - 64)**2 * 1e-4 + (1.0 / budget)
    return float(loss)

# Hyperparameter search space
param_distributions = {
    "lr": ("log_uniform", 1e-4, 1e-1),
    "hidden": ("int_uniform", 16, 128),
}

# Run BOHB successive halving optimization
bohb = BOHB(
    objective_fn=objective_fn,
    param_distributions=param_distributions,
    min_budget=1,
    max_budget=9,
    eta=3,
    num_iterations=3
)
best_config, best_loss = bohb.optimize()
print(f"Optimal Hyperparameters: {best_config}, Minimum Loss: {best_loss:.5f}")
```

#### Parameter Breakdown: `BOHB`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `objective_fn` | `Callable` | *Required* | Objective function `f(config, budget) -> float` to minimize. |
| `param_distributions` | `dict` | *Required* | Dictionary defining hyperparameter search bounds and distributions. |
| `min_budget` | `float` | `1.0` | Minimum resource budget allocated per candidate configuration. |
| `max_budget` | `float` | `9.0` | Maximum resource budget allocated to top-performing configurations. |
| `eta` | `int` | `3` | Halving reduction factor (prunes $1 - 1/\eta$ candidates each round). |
| `num_iterations` | `int` | `3` | Number of HyperBand outer brackets executed. |

---

### 17.7 SuperLearner Out-of-Fold Cross-Validated Stacking (`SuperLearner`)

```python
import numpy as np
from chokkhu.automl import SuperLearner
from chokkhu.models import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier

# Multi-class classification dataset
X = np.random.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Define diverse ensemble of base learners
base_learners = [
    ("logreg", LogisticRegression()),
    ("dt", DecisionTreeClassifier(max_depth=4)),
    ("rf", RandomForestClassifier(n_estimators=10, max_depth=4)),
]

# Construct SuperLearner with out-of-fold CV meta-learning
super_learner = SuperLearner(
    base_models=base_learners,
    meta_model="ridge",
    cv=5,
    task="classification",
    use_probabilities=True
)
super_learner.fit(X, y)

# Predict class probabilities and labels
y_pred_proba = super_learner.predict_proba(X[:5])
y_pred = super_learner.predict(X[:5])
print(f"Ensemble Predictions: {y_pred}")
print(f"Optimal Base Learner Weights: {super_learner.get_weights()}")
```

#### Parameter Breakdown: `SuperLearner`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `base_models` | `list[tuple[str, Any]]` | *Required* | List of named base estimators `[(name, model), ...]`. |
| `meta_model` | `str` / `Any` | `"ridge"` | Meta-learner strategy (`"ridge"`, `"nnls"`, `"logistic"`, or custom estimator). |
| `cv` | `int` | `5` | Number of out-of-fold cross-validation folds for meta-feature generation. |
| `task` | `str` | `"classification"` | Task type: `"classification"` or `"regression"`. |
| `use_probabilities` | `bool` | `True` | Whether classification meta-features use class probabilities or hard labels. |


---

## 18. Privacy, Quantum ML, Geo-Spatial, NAS & TDA (Milestone 15)

### 18.1 Differential Privacy & DP-SGD (`LaplaceMechanism`, `GaussianMechanism`, `DP_SGD`)

```python
import numpy as np
from chokkhu.privacy import LaplaceMechanism, GaussianMechanism, DP_SGD

# 1. Laplace Mechanism for query perturbation
laplace = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)
noisy_count = laplace.perturb(100.0)
print(f"Differentially Private Perturbed Count: {noisy_count:.2f}")

# 2. Gaussian Mechanism for (epsilon, delta)-DP
gaussian = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
noisy_stat = gaussian.perturb(42.0)
print(f"Gaussian Perturbed Statistic: {noisy_stat:.2f}")

# 3. Differentially Private SGD Optimizer with Per-Sample Clipping
dp_optimizer = DP_SGD(lr=0.01, l2_norm_clip=1.0, noise_multiplier=0.8)
params = {"W": np.random.randn(10, 2), "b": np.zeros(2)}
per_sample_grads = [{"W": np.random.randn(10, 2), "b": np.random.randn(2)} for _ in range(8)]

updated_params = dp_optimizer.step(params, per_sample_grads)
privacy_spent = dp_optimizer.compute_privacy_spent(total_samples=1000, batch_size=8, target_delta=1e-5)
print(f"Privacy Spent: epsilon={privacy_spent['epsilon']:.4f}, delta={privacy_spent['delta']}")
```

#### Parameter Breakdown: `DP_SGD`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `lr` | `float` | `0.01` | Optimization step learning rate. |
| `l2_norm_clip` | `float` | `1.0` | Maximum L2 norm bound $C$ for per-sample gradient clipping. |
| `noise_multiplier` | `float` | `1.0` | Calibrated noise ratio $\sigma = \text{noise\_multiplier} \times C$. |
| `random_state` | `int` / `None` | `None` | Seed for reproducible pseudo-random noise generation. |

---

### 18.2 Decentralized Federated Learning (`FederatedServer`, `FederatedClient`, `FedAvg`, `FedProx`)

```python
import numpy as np
from chokkhu.privacy import FederatedServer, FederatedClient

# Simulate decentralized non-IID client partitions
client1_X, client1_y = np.random.randn(50, 4), np.random.randn(50, 1)
client2_X, client2_y = np.random.randn(60, 4), np.random.randn(60, 1)

# Register clients with FedProx proximal regularization (mu > 0)
c1 = FederatedClient(client_id="hospital_A", X=client1_X, y=client1_y, mu=0.01, lr=0.01)
c2 = FederatedClient(client_id="hospital_B", X=client2_X, y=client2_y, mu=0.01, lr=0.01)

# Orchestrate federated aggregation rounds
server = FederatedServer(
    initial_weights={"W": np.zeros((4, 1)), "b": np.zeros((1,))},
    clients=[c1, c2],
    strategy="fedprox"
)

for r in range(5):
    round_summary = server.train_round(fraction_fit=1.0, local_epochs=3, batch_size=16)
    print(f"Round {r+1} Complete - Mean Client Loss: {round_summary['mean_client_loss']:.4f}")

global_predictions = server.predict(np.random.randn(5, 4))
print(f"Global Model Predictions Shape: {global_predictions.shape}")
```

#### Parameter Breakdown: `FederatedServer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `initial_weights` | `dict` | *Required* | Initial global model parameters `{'W': ..., 'b': ...}`. |
| `clients` | `list` | `None` | Registered `FederatedClient` instances. |
| `strategy` | `str` | `"fedavg"` | Aggregation strategy (`"fedavg"` or `"fedprox"`). |

---

### 18.3 Quantum Machine Learning Simulator & Variational Quantum Classifier (`QuantumCircuit`, `VQC`, `QuantumKernel`)

```python
import numpy as np
from chokkhu.quantum import QuantumCircuit, VariationalQuantumClassifier, QuantumKernel

# 1. State Vector Quantum Circuit Simulation & Bell State Generation
qc = QuantumCircuit(n_qubits=2)
qc.h(0).cnot(0, 1)  # Create maximally entangled Bell state (|00> + |11>) / sqrt(2)
print(f"Computational Basis Probabilities: {qc.probabilities()}")
print(f"Pauli-Z Expectation <Z_0>: {qc.expectation_z(0):.4f}")

# 2. Variational Quantum Classifier (QNN) with Exact Parameter Shift Rule Gradients
X_quantum = np.random.uniform(-np.pi, np.pi, size=(30, 2))
y_quantum = (X_quantum[:, 0] + X_quantum[:, 1] > 0).astype(int)

vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=2, lr=0.1)
vqc.fit(X_quantum, y_quantum, epochs=10, batch_size=8)
q_preds = vqc.predict(X_quantum[:5])
print(f"VQC Predictions: {q_preds}")

# 3. Quantum Kernel Matrix Estimation
qk = QuantumKernel(n_qubits=2)
gram_matrix = qk.compute_matrix(X_quantum[:10])
print(f"Quantum Gram Matrix (Fidelity) Shape: {gram_matrix.shape}")
```

#### Parameter Breakdown: `VariationalQuantumClassifier`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_qubits` | `int` | `2` | Number of simulated qubits in quantum register. |
| `n_layers` | `int` | `2` | Number of parameterized variational rotation and entangling ansatz layers. |
| `lr` | `float` | `0.05` | Learning rate for parameter shift rule gradient descent. |

---

### 18.4 Geo-Spatial Statistics, Spatial Autoregression & GWR (`SpatialWeights`, `morans_i`, `SAR`, `GWR`)

```python
import numpy as np
from chokkhu.geospatial import SpatialWeights, morans_i, SpatialAutoregression, GeographicallyWeightedRegression

# 2D geographic coordinates [lat, lon] and spatial attributes
coords = np.random.uniform(20.0, 26.0, size=(50, 2))
values = coords[:, 0] * 3.5 + np.random.randn(50) * 0.2

# 1. Spatial Weight Matrix and Global Moran's I Autocorrelation
sw = SpatialWeights(coords, method="knn", k=5, row_standardize=True)
moran = morans_i(values, sw, permutations=99)
print(f"Global Moran's I: {moran['I']:.4f}, p-value: {moran['p_value']:.4f}")

# 2. Spatial Autoregression (Spatial Lag Model: y = rho * W * y + X * beta)
X_geo = np.random.randn(50, 3)
sar = SpatialAutoregression(model_type="lag")
sar.fit(X_geo, values, sw)
print(f"Estimated Spatial Autoregressive Rho: {sar.rho:.4f}")

# 3. Geographically Weighted Regression (Localized Spatial Beta Estimation)
gwr = GeographicallyWeightedRegression(bandwidth=2.0, kernel="gaussian")
gwr.fit(coords, X_geo, values)
gwr_preds = gwr.predict(coords[:5], X_geo[:5])
print(f"GWR Predictions: {gwr_preds}")
```

#### Parameter Breakdown: `SpatialWeights`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `coords` | `np.ndarray` | *Required* | Coordinate matrix $(N \times 2)$ containing spatial coordinates. |
| `method` | `str` | `"knn"` | Kernel structure: `"knn"`, `"inverse_distance"`, `"gaussian"`, or `"threshold"`. |
| `k` | `int` | `5` | Number of nearest neighbors for `"knn"` method. |
| `bandwidth` | `float` | `1.0` | Distance decay parameter for `"gaussian"` or distance threshold. |
| `row_standardize`| `bool` | `True` | Row-standardization ($\sum_j W_{ij} = 1.0$). |

---

### 18.5 Geostatistical Variogram Modeling & Spatial Kriging (`OrdinaryKriging`)

```python
import numpy as np
from chokkhu.geospatial import OrdinaryKriging

# Spatial point observations
obs_coords = np.random.uniform(0.0, 10.0, size=(30, 2))
z_values = np.sin(obs_coords[:, 0]) + np.cos(obs_coords[:, 1])

# Ordinary Kriging with Spherical Variogram Model
kriging = OrdinaryKriging(variogram_model="spherical", nugget=0.02, sill=1.0, range_val=4.0)
kriging.fit(obs_coords, z_values)

# Interpolate over target spatial grid
target_coords = np.array([[2.5, 3.5], [7.0, 8.0]])
z_pred, error_variance = kriging.predict(target_coords)
print(f"Kriging Interpolated Values: {z_pred}, Estimation Variances: {error_variance}")
```

#### Parameter Breakdown: `OrdinaryKriging`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `variogram_model` | `str` | `"spherical"` | Theoretical semivariogram model: `"spherical"`, `"exponential"`, or `"gaussian"`. |
| `nugget` | `float` | `0.0` | Micro-scale variance / measurement error $c_0$. |
| `sill` | `float` | `1.0` | Total asymptotic semivariance $c_0 + c$. |
| `range_val` | `float` | `1.0` | Spatial correlation range threshold $a$. |

---

### 18.6 Differentiable Neural Architecture Search (`DARTS`)

```python
import numpy as np
from chokkhu.automl import DARTS

# Tabular classification dataset
X_nas = np.random.randn(100, 8)
y_nas = (X_nas[:, 0] + X_nas[:, 1] > 0).astype(int)

# Continuous relaxation of candidate operations (Identity, Linear, ReLU, GELU, Residual)
darts = DARTS(
    input_dim=8,
    num_classes=2,
    num_intermediate_nodes=3,
    hidden_dim=32,
    lr_weights=0.01,
    lr_arch=0.005
)
darts.fit(X_nas, y_nas, epochs=10, batch_size=16)

# Extract discovered discrete optimal architecture
genotype = darts.genotype()
print(f"Discovered Optimal Neural Architecture Genotype: {genotype}")
```

#### Parameter Breakdown: `DARTS`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | *Required* | Feature input dimension $D$. |
| `num_classes` | `int` | `2` | Number of output target classes. |
| `num_intermediate_nodes` | `int` | `3` | Number of mixed-op candidate layers in supernet search space. |
| `hidden_dim` | `int` | `32` | Layer width for intermediate candidate operations. |
| `lr_weights` | `float` | `0.01` | Learning rate for candidate operation parameters $w$. |
| `lr_arch` | `float` | `0.005` | Learning rate for architecture parameters $\alpha$. |

---

### 18.7 Topological Data Analysis & Persistent Homology (`VietorisRipsComplex`, `PersistenceLandscape`)

```python
import numpy as np
from chokkhu.tda import VietorisRipsComplex, PersistenceDiagram, PersistenceLandscape, bottleneck_distance

# Point cloud with topological hole (circle)
angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
point_cloud = np.column_stack([np.cos(angles), np.sin(angles)]) + np.random.randn(24, 2) * 0.05

# 1. Build Vietoris-Rips Simplicial Filtration
rips = VietorisRipsComplex(max_edge_length=2.5, max_dimension=1)
diagrams = rips.fit_transform(point_cloud)

# 2. Extract H0 (connected components) and H1 (loops) Persistence
h0_diag = PersistenceDiagram(diagrams[0], dimension=0)
h1_diag = PersistenceDiagram(diagrams[1], dimension=1)
print(f"H0 Total Persistence: {h0_diag.total_persistence():.4f}, Persistent Entropy: {h0_diag.persistent_entropy():.4f}")

# 3. Vectorize into Continuous Persistence Landscapes for ML Classifiers
landscape = PersistenceLandscape(num_landscapes=3, resolution=40)
topo_features = landscape.transform(diagrams[0])
print(f"Extracted Topological Feature Vector Shape: {topo_features.shape}")
```

#### Parameter Breakdown: `VietorisRipsComplex`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `max_edge_length` | `float` | `inf` | Maximum filtration scale $\epsilon$ for simplex construction. |
| `max_dimension` | `int` | `1` | Maximum homology dimension ($0$ for $H_0$, $1$ for $H_0$ and $H_1$). |

## License & Citation

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

```bibtex
@software{chokkhu2026,
  author = {tamimystic and Inception BD Team},
  title = {Chokkhu: Sovereign Zero-Dependency Machine Learning and LLM Ecosystem},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/tamimystic/chokkhu}
}
```
