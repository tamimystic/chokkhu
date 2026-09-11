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
22. [Spiking Neural Networks, Equilibrium Models & GFlowNets (Milestone 16)](#19-advanced-frontier-architectures-milestone-16)
23. [Production Generative AI, Audio Codecs & Genetic AutoML (Milestone 17)](#20-sovereign-production-generative-ai-audio-codecs-speculative-llms--genetic-automl)
24. [Edge Serving, State-Space Models & Optimal Transport (Milestone 18)](#21-sovereign-edge-to-cloud-serving-hybrid-state-space-continual-learning--optimal-transport-milestone-18)
25. [Robotics, World Models, Multi-Agent RL, Genomic AI & HDC (Milestone 19)](#22-robotics-world-models-multi-agent-rl-genomic-ai--hyperdimensional-computing-milestone-19)
26. [The Grand Frontier: Discrete Diffusion, Physics, Safety, Causal DAGs & Hyperbolic (Milestone 20)](#23-the-grand-frontier-discrete-diffusion-differentiable-physics-llm-safety-causal-dags--hyperbolic-geometry-milestone-20)
27. [License & Citation](#license--citation)

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
| **Multi-Agent Economy & XAI Circuits** | Replicator-Mutator Dynamics, VCG Combinatorial Auctions, Causal Activation Patching, CFG Guided Diffusion Inpainting, Raw Byte Transformers, NeuralSort |

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
    X_train,
    y_train,
    task="classification",
    time_budget_secs=30,
    metric="accuracy",
    cv=5,
    random_state=42,
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
def objective(config):
    x, y = config["x"], config["y"]
    return -(x**2 + y**2) + 10.0

# Define parameter search bounds
bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
bo = BayesianOptimization(
    objective_fn=objective,
    param_bounds=bounds,
    n_init=5,
    n_iter=10,
    acquisition="ei",
)
best_params = bo.optimize()
print(f"Optimal Parameters: {best_params}, Maximum Objective: {bo.best_score:.4f}")
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
embeddings = np.random.randn(100, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build HNSW index with hierarchical multi-layer graph
index = HNSWIndex(
    dim=128,
    metric="euclidean",
    m=16,
    ef_construction=64,
    ef_search=32,
)
index.add(embeddings)

# Query top-5 nearest neighbors
query = embeddings[0]
labels, distances = index.search(query, k=5)
print(f"HNSW Top-5 Neighbor Indices: {labels}")
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

vectors = np.random.randn(200, 64).astype(np.float32)

# Build compressed IVF-PQ index
ivfpq = IVFPQIndex(
    dim=64,
    n_lists=16,
    n_subvectors=8,
    n_bits=8,
    metric="euclidean",
)
ivfpq.train(vectors)
ivfpq.add(vectors)

query_vec = vectors[0]
dists, indices = ivfpq.search(query_vec, k=5)
print(f"IVF-PQ Approximate Nearest Neighbors: {indices}")
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
    field_cardinalities=cat_dims,
    embedding_dim=16,
    mlp_layers=[32, 16],
    lr=0.01,
    n_epochs=5,
)
# Categorical sample index vectors: (batch_size=4, num_fields=3)
X_cat = np.array([[12, 5, 2], [34, 18, 7], [90, 42, 1], [0, 11, 9]])
y = np.array([1, 0, 1, 0])

model.fit(X_cat, y)
predictions = model.predict_proba(X_cat)
print(f"DeepFM Click Probability Predictions: {predictions}")
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
from chokkhu.models.recommendation import SASRec

# Sequence of interacted item IDs: list of user interaction sequences
item_sequences = [
    [1, 5, 23, 45, 12],
    [3, 8, 19, 2, 10],
    [10, 11, 12, 13, 14],
]

sasrec = SASRec(
    max_len=5,
    hidden_dim=32,
    n_heads=2,
    n_layers=2,
    lr=0.01,
    n_epochs=3,
)
sasrec.fit(item_sequences)
scores = sasrec.predict_next(item_sequences[0], top_k=5)
print(f"Next-Item Ranking Predictions: {scores}")
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
X = np.random.randn(100, 5)
T = np.random.binomial(1, 0.5, size=100)
y = 1.5 * T + np.dot(X, [0.5, -0.2, 0.8, -0.1, 0.3]) + np.random.randn(100) * 0.1

dr = DoublyRobustLearner(
    clip_range=(0.05, 0.95),
)
dr.fit(X, T, y)
ite_estimates = dr.predict_cate(X[:5])
ate_estimate = dr.estimate_ate()
print(f"Average Treatment Effect (ATE): {ate_estimate:.4f}")
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
X = np.random.randn(100, 4)
times = np.random.exponential(scale=10.0, size=100)
events = np.random.binomial(1, 0.7, size=100)

cox = CoxPHRegression(
    alpha=0.01,
    tie_method="efron",
    max_iter=50,
)
cox.fit(X, times, events)
risk_scores = cox.predict_risk(X[:5])
c_index = concordance_index(times[:5], risk_scores, events[:5])
print(f"Cox Proportional Hazards C-Index: {c_index:.4f}")
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

X = np.random.randn(100, 6)
times = np.random.exponential(scale=12.0, size=100)
events = np.random.binomial(1, 0.8, size=100)

deep_surv = DeepSurv(
    hidden_layers=[32, 16],
    lr=0.01,
    weight_decay=1e-4,
    n_epochs=20,
)
deep_surv.fit(X, times, events)
risk_scores = deep_surv.predict_risk(X[:5])
print(f"DeepSurv Log Hazard Ratios: {risk_scores.flatten()}")
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
    embed_dim=64,
    image_size=32,
    patch_size=8,
    vocab_size=500,
    init_temperature=0.07,
    seed=42,
)

# Vision images (B, C, H, W) and text token sequences (B, S)
images = np.random.randn(4, 3, 32, 32).astype(np.float32)
texts = np.random.randint(0, 500, size=(4, 8))

# Compute normalized embeddings & cross-entropy contrastive loss
img_emb, txt_emb, loss = clip.forward(images, texts)
print(f"CLIP Image Embeddings Shape: {img_emb.shape}, Loss: {loss:.4f}")
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

# Arbitrary-length visual tokens: (batch_size=2, num_tokens=16, dim=256)
visual_tokens = np.random.randn(2, 16, 256).astype(np.float32)

resampler = PerceiverResampler(
    vision_dim=256,
    text_dim=256,
    num_latents=16,
    num_heads=4,
)
compact_visual_prompts = resampler.forward(visual_tokens)
print(f"Fixed-Size Visual Prompt Embeddings Shape: {compact_visual_prompts.shape}")
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
    nu=0.01 / np.pi,
    hidden_layers=[32, 32],
    lr=0.005,
)

# Collocation spatio-temporal coordinates (x, t) in [-1, 1] x [0, 1]
tx_colloc = np.random.uniform(low=[-1.0, 0.0], high=[1.0, 1.0], size=(50, 2))
tx_init = np.column_stack([np.random.uniform(-1.0, 1.0, size=30), np.zeros(30)])
u_init = -np.sin(np.pi * tx_init[:, 0:1])
tx_bnd = np.column_stack([np.random.choice([-1.0, 1.0], size=30), np.random.uniform(0.0, 1.0, size=30)])
u_bnd = np.zeros((30, 1))

pinn.fit(tx_colloc, tx_init, u_init, tx_bnd, u_bnd, n_epochs=10)

tx_test = np.column_stack([np.linspace(-1, 1, 20), np.full(20, 0.5)])
u_pred = pinn.predict(tx_test)
print(f"Burgers Equation Solution u(x, t=0.5) Shape: {u_pred.shape}")
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
    dim=2,
    hidden_dim=32,
    solver="rk4",
)
x0 = np.random.randn(4, 2).astype(np.float32)
t_span = np.linspace(0, 1, 10).astype(np.float32)

trajectories = node.forward(x0, t_span)
print(f"Integrated ODE Trajectories Shape: {trajectories.shape}")
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
    method="minmax",
    bits=8,
)
quantizer = ptq.calibrate(weights_fp32)
q_weights = quantizer.quantize(weights_fp32)
weights_reconstructed = quantizer.dequantize(q_weights)
quant_error = np.mean(np.abs(weights_fp32 - weights_reconstructed))
print(f"INT8 Quantization Error (MAE): {quant_error:.6f}")
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

# 3D Point cloud batch: (batch_size=2, num_points=64, xyz_coords=3)
point_clouds = np.random.randn(2, 64, 3).astype(np.float32)

pointnet2 = PointNet2Classifier(
    num_classes=10,
)
logits = pointnet2.forward(point_clouds)
print(f"PointNet++ Classification Logits Shape: {logits.shape}")
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
rays_o, rays_d = generate_camera_rays(
    height=64,
    width=64,
    focal_length=100.0,
    camera_pose=c2w,
)
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
    """Add two floating point numbers."""
    return float(a + b)

registry.register(
    Tool(
        name="calculator",
        func=add_numbers,
        description="Adds two numbers a and b.",
    )
)

agent = ReActAgent(
    tools=registry,
    max_iterations=3,
)
result = agent.run("Calculate the sum of 45.5 and 54.5")
print(f"ReAct Agent Thought & Execution Result: {result}")
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

def thought_gen(state: str, n: int):
    return [f"{state} -> candidate_{i}" for i in range(n)]

def custom_evaluator(thought_state: str) -> float:
    return 1.0 if "candidate_0" in thought_state else 0.2

tot = TreeOfThoughts(
    thought_generator=thought_gen,
    state_evaluator=custom_evaluator,
    max_depth=2,
    branching_factor=2,
)
solution = tot.solve(initial_state="Given input data X: find optimal pipeline.")
print(f"Tree-of-Thoughts Optimal Node State: {solution}")
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
    lr_actor=1e-3,
    lr_critic=2e-3,
    gamma=0.99,
)
state = np.random.randn(4)
action, log_prob, value = ppo.select_action(state)
print(f"PPO Selected Action: {action}, Log Prob: {log_prob:.4f}, State Value: {value:.4f}")
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
    hidden_dim=64,
    lr=1e-3,
    gamma=0.99,
)
continuous_action, _ = sac.sample_action(np.random.randn(8))
print(f"SAC Continuous Action: {continuous_action}")

# 2. Linear Upper Confidence Bound (LinUCB) Contextual Bandit
bandit = LinUCBBandit(n_arms=5, n_features=10, alpha=1.0)
context = np.random.randn(10)
chosen_arm = bandit.select_arm(context)
bandit.update(arm=chosen_arm, context=context, reward=1.0)
print(f"LinUCB Selected Arm: {chosen_arm}")
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
from chokkhu.models.gnn import Graphormer

# Node features: (num_nodes=6, in_features=16)
x = np.random.randn(6, 16).astype(np.float32)
adj = np.eye(6, dtype=np.float32)

# Graphormer with Spatial & Degree Centrality Encodings
graphormer = Graphormer(
    in_dim=16,
    hidden_dim=32,
    out_dim=2,
    num_layers=2,
    n_heads=2,
    pe_dim=4,
)
out = graphormer.forward(x, adj)
print(f"Graphormer Output Graph Embeddings Shape: {out.shape}")
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
# Coordinates: (num_nodes=5, 3D xyz)
x_coords = np.random.randn(5, 3).astype(np.float32)
edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

egnn = EGNN(
    in_dim=16,
    hidden_dim=32,
    out_dim=16,
    num_layers=2,
)
h_out, x_out = egnn.forward(h, x_coords, edge_index)
print(f"EGNN Scalar Feature Shape: {h_out.shape}, Equivariant 3D Coordinates Shape: {x_out.shape}")
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

# Token sequence: (batch_size=2, seq_len=4)
input_tokens = np.array([
    [12, 45, 89, 230],
    [7, 102, 304, 511],
])

# Sovereign DeepSeek-V3 with MLA & DeepSeekMoE
deepseek = DeepSeekV3(
    vocab_size=1000,
    hidden_dim=64,
    num_layers=2,
    n_routed_experts=4,
    top_k=2,
)
logits = deepseek.forward(input_tokens)
print(f"DeepSeek-V3 Logits Shape: {logits.shape}")
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
    num_layers=2,
    d_state=16,
)
mamba_logits = mamba.forward(input_ids)
print(f"Mamba S6 Output Shape: {mamba_logits.shape}")
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
from chokkhu.models.nlp import DPOTrainer

dpo = DPOTrainer(
    beta=0.1,
)
chosen_logps = np.array([-1.2, -0.8])
rejected_logps = np.array([-2.5, -2.1])
ref_chosen_logps = np.array([-1.5, -1.0])
ref_rejected_logps = np.array([-2.0, -1.8])

loss, chosen_rewards, rejected_rewards = dpo.compute_loss(
    chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
)
print(f"DPO Alignment Loss: {loss:.4f}")
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
images_flat = np.random.randn(20, 784).astype(np.float32)
vae = VAE(in_features=784, hidden_dim=64, latent_dim=16)
recon, mu, logvar = vae.forward(images_flat)
print(f"VAE Reconstructed Shape: {recon.shape}")

# 2. Denoising Diffusion Probabilistic Model (DDPM)
ddpm = DDPM(data_dim=32, timesteps=20, beta_start=1e-4, beta_end=0.02)
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
mel_spec = melspectrogram(waveform, sr=16000, n_fft=512, hop_length=160, n_mels=80)
mfcc_features = mfcc(waveform, sr=16000, n_mfcc=13)
print(f"Mel-Spectrogram Shape: {mel_spec.shape}, MFCC Shape: {mfcc_features.shape}")

# Speech Conformer (Macaron-style FFN + MHA + Depthwise Conv)
conformer = Conformer(
    in_features=80,
    num_classes=10,
    embed_dim=64,
    num_blocks=2,
    num_heads=2,
    ffn_dim=128,
)
spec_batch = np.random.randn(2, 20, 80).astype(np.float32)
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
X_train = np.random.randn(50, 10)
y_train = np.random.randint(0, 2, size=50)
model = ck.NeuralNetwork(layers=[32], activation="relu", epochs=5)
model.fit(X_train, y_train)

sample = np.random.randn(10)

# 1. Axiomatic feature attribution via Integrated Gradients
ig = IntegratedGradients(model=model, steps=20, baseline=None)
attributions = ig.attribute(sample)
print(f"Integrated Gradients Attributions Shape: {attributions.shape}")

# 2. DeepLIFT difference-from-reference conservation
dlift = DeepLIFT(model=model, baseline=np.zeros(10))
dlift_attributions = dlift.attribute(sample)
print(f"DeepLIFT Attributions: {dlift_attributions.flatten()[:5]}")
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
flow = FlowMatching(input_dim=2, hidden_dims=[64, 64], lr=1e-3, sigma_min=1e-4)
for _ in range(5):
    loss = flow.train_step(X_train)

# Sample new points using 4th-order Runge-Kutta (RK4) integration
synthetic_samples = flow.sample(num_samples=100, steps=50, method="rk4")
print(f"Generated Flow Samples Shape: {synthetic_samples.shape}")

# 2. Rectified Flow interpolation with straight-line trajectories
rect_flow = RectifiedFlow(input_dim=2, hidden_dims=[32, 32], lr=1e-3)
for _ in range(5):
    loss = rect_flow.train_step(X_train)
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
nvp = RealNVP(dim=4, num_layers=4, hidden_dim=32, lr=1e-3)
for _ in range(5):
    loss = nvp.compute_loss(X)

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
linear = LoRALinear(in_features=128, out_features=64, r=4, lora_alpha=8.0)

x = np.random.randn(8, 128)
y = linear.forward(x)
print(f"Adapted Linear Output Shape: {y.shape}")

# Merge adapter weights directly into frozen base weight matrix for zero-latency inference
linear.merge()
y_merged = linear.forward(x)
np.testing.assert_allclose(y, y_merged, atol=1e-5)

# 2. Attach PEFT LoRA adapter tracking
param_summary = LoRAAdapter.count_parameters([linear])
print(f"LoRA Parameter Summary: {param_summary}")
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
conformal = ConformalPredictor(base_estimator=base_model, alpha=0.05)
conformal.calibrate(X[80:110], y[80:110])

# Predict lower and upper confidence bounds
bounds = conformal.predict_interval(X[110:])
lower, upper = bounds[0], bounds[1]
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
profile, profile_idx = mp.compute(ts)

# Identify recurring patterns (motifs) and anomalous subsequences (discords)
motifs = find_motifs(ts, window_size=30, top_k=2)
discords = find_discords(ts, window_size=30, top_k=1)
print(f"Top Discord (Anomaly) Subsequence Info: {discords[0]}")
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
def eval_func(config, budget):
    lr = config["lr"]
    hidden = config["hidden"]
    loss = (np.log10(lr) + 3.0)**2 + (hidden - 64)**2 * 1e-4 + (1.0 / budget)
    return float(-loss)  # maximize objective

param_bounds = {
    "lr": (1e-4, 1e-1),
    "hidden": (16.0, 128.0),
}

bohb = BOHB(
    eval_func=eval_func,
    param_bounds=param_bounds,
    min_budget=1.0,
    max_budget=9.0,
    eta=3,
)
best_config, best_score = bohb.optimize()
print(f"Optimal Hyperparameters: {best_config}, Best Score: {best_score:.5f}")
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
from chokkhu.models import LogisticRegression, DecisionTree, RandomForest

# Multi-class classification dataset
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Define diverse ensemble of base learners
base_learners = [
    LogisticRegression(),
    DecisionTree(max_depth=4),
    RandomForest(n_estimators=5, max_depth=4),
]

# Construct SuperLearner with out-of-fold CV meta-learning
super_learner = SuperLearner(
    estimators=base_learners,
    cv=3,
    task="classification",
    use_probabilities=True
)
super_learner.fit(X, y)

# Predict class probabilities and labels
y_pred_proba = super_learner.predict_proba(X[:5])
y_pred = super_learner.predict(X[:5])
print(f"Ensemble Predictions: {y_pred}")
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

---

## 19. Advanced Frontier Architectures (Milestone 16)

### 19.1 Hybrid Lexical-Dense Retrieval & Reciprocal Rank Fusion (`BM25Plus`, `OkapiBM25`, `HybridReranker`)

```python
import numpy as np
from chokkhu.models.retrieval import BM25Plus, OkapiBM25, HybridReranker

corpus = [
    "quantum computing and machine learning algorithms",
    "deep learning neural networks for computer vision",
    "natural language processing with transformers and attention",
    "quantum physics and quantum entanglement theory",
]

# 1. Okapi BM25 & BM25+ Sparse Retrieval
bm25 = BM25Plus(k1=1.5, b=0.75, delta=1.0)
bm25.fit(corpus)
sparse_scores = bm25.get_scores("quantum computing")
top_scores, top_indices = bm25.search("quantum computing", top_k=2)
print(f"BM25+ Top Results: Indices={top_indices}, Scores={top_scores}")

# 2. Hybrid Reranking (RRF & Convex Score Blending)
dense_scores = np.array([0.92, 0.45, 0.68, 0.12])
reranker = HybridReranker(rrf_k=60, alpha=0.6)
blended_scores = reranker.blend_scores(dense_scores, sparse_scores, alpha=0.6)
fused_rankings = reranker.reciprocal_rank_fusion([[0, 2, 1, 3], top_indices], top_k=2)
print(f"Fused Hybrid Rankings (RRF): {fused_rankings}")
```

#### Parameter Breakdown: `BM25Plus` & `OkapiBM25`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `k1` | `float` | `1.5` | Term frequency saturation non-linear scaling parameter. |
| `b` | `float` | `0.75` | Document length normalization penalty parameter ($0 \le b \le 1$). |
| `delta` | `float` | `1.0` | Lower-bound tuning parameter in BM25+ to prevent excessive length penalties ($0.0$ for standard Okapi BM25). |

#### Parameter Breakdown: `HybridReranker`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `rrf_k` | `int` | `60` | Constant dampening factor in Reciprocal Rank Fusion ($1 / (k + \text{rank})$). |
| `alpha` | `float` | `0.5` | Convex combination weight for dense scores vs sparse lexical scores ($\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$). |

---

### 19.2 Mechanistic Interpretability & Actionable Counterfactuals (`AttentionRollout`, `DirectLogitAttribution`, `WachterCounterfactualExplainer`, `TCAV`)

```python
import numpy as np
from chokkhu.explainability import (
    AttentionRollout,
    DirectLogitAttribution,
    WachterCounterfactualExplainer,
    TCAV,
)

# 1. Mechanistic Attention Rollout across transformer layers
rollout = AttentionRollout(discard_ratio=0.1, head_reduction="mean")
layers_attn = [np.random.rand(4, 6, 6) for _ in range(3)]
layers_attn = [a / a.sum(axis=-1, keepdims=True) for a in layers_attn]
rollout_matrix = rollout.compute(layers_attn)
print(f"Attention Rollout Matrix Shape: {rollout_matrix.shape}")

# 2. Direct Logit Attribution (DLA) on residual streams
unembedding = np.random.randn(16, 50)
dla = DirectLogitAttribution(unembedding_matrix=unembedding)
residual_hidden_states = np.random.randn(4, 16)
layer_attributions = dla.attribute(residual_hidden_states, target_token_id=5)
print(f"Layer-wise Logit Attributions for Token #5: {layer_attributions}")

# 3. Wachter Optimization-Based Counterfactual Explainer
def predict_fn(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.dot(x, np.array([1.5, 1.2]))))

explainer = WachterCounterfactualExplainer(predict_fn=predict_fn, lr=0.1, max_iter=100)
cf_summary = explainer.explain(np.array([-1.0, -0.5]), target_prediction=0.8)
print(f"Discovered Counterfactual Input: {cf_summary['counterfactual']}, Distance={cf_summary['l2_distance']:.4f}")

# 4. Testing with Concept Activation Vectors (TCAV)
tcav = TCAV(seed=42)
cav = tcav.compute_cav(np.random.randn(20, 16) + 2.0, np.random.randn(20, 16) - 2.0)
tcav_result = tcav.compute_tcav_score(lambda a: np.dot(a, cav)[:, None], np.random.randn(10, 16))
print(f"Quantitative Concept Sensitivity (TCAV Score): {tcav_result['tcav_score']:.4f}")
```

#### Parameter Breakdown: `AttentionRollout` & `DirectLogitAttribution`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `discard_ratio` | `float` | `0.0` | Fraction of lowest attention weights to zero out per layer ($0.0 \le r < 1.0$). |
| `head_reduction` | `str` | `"mean"` | Aggregation strategy across attention heads (`"mean"`, `"max"`, or `"min"`). |
| `unembedding_matrix` | `np.ndarray` | *Required* | Final output projection matrix $W_U$ mapping hidden state $D \to V$. |

#### Parameter Breakdown: `WachterCounterfactualExplainer` & `TCAV`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `predict_fn` | `Callable` | *Optional* | Black-box predictor or scoring function $f(x)$. |
| `lambda_1` | `float` | `0.01` | $L_1$ sparsity regularizer penalty weight. |
| `lambda_2` | `float` | `0.05` | $L_2$ proximity regularizer penalty weight. |
| `lr` | `float` | `0.05` | Learning rate for Adam perturbation optimizer. |
| `max_iter` | `int` | `300` | Maximum inner gradient descent iterations. |
| `tolerance` | `float` | `0.05` | Target outcome convergence tolerance threshold. |
| `seed` | `int` | `42` | Random seed for concept activation separation. |

---

### 19.3 Dense & Promptable Computer Vision (`DeepLabV3Plus`, `SegmentAnythingModel`)

```python
import numpy as np
from chokkhu.models.vision.architectures import DeepLabV3Plus, SegmentAnythingModel

# 1. DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP)
deeplab = DeepLabV3Plus(
    in_channels=3,
    num_classes=3,
    backbone_channels=8,
    aspp_channels=8,
    seed=42,
)
img_batch = np.random.randn(2, 3, 16, 16).astype(np.float32)
logits = deeplab.forward(img_batch)
masks = deeplab.predict(img_batch)
print(f"DeepLabV3+ Segmentation Logits Shape: {logits.shape}, Predicted Mask Map: {masks.shape}")

# 2. Segment Anything Model (SAM) Promptable Mask Decoder
sam = SegmentAnythingModel(embed_dim=16, num_heads=2, num_mask_tokens=3, seed=42)
img_embeddings = np.random.randn(1, 16, 8, 8).astype(np.float32)
point_coords = np.array([[[2.0, 3.0], [5.0, 6.0]]], dtype=np.float32)
point_labels = np.array([[1, 0]], dtype=np.int32)
pred_masks, iou_scores = sam.forward(img_embeddings, points=point_coords, labels=point_labels)
print(f"SAM Decoded Masks Shape: {pred_masks.shape}, Predicted IoU Quality: {iou_scores}")
```

#### Parameter Breakdown: `DeepLabV3Plus`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | `3` | Number of image input color channels (e.g. RGB=3). |
| `num_classes` | `int` | `2` | Number of target semantic segmentation classes. |
| `backbone_channels` | `int` | `32` | Number of feature filters in convolutional stem. |
| `aspp_channels` | `int` | `32` | Channel width for dilated ASPP pyramid branches. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

#### Parameter Breakdown: `SegmentAnythingModel`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `embed_dim` | `int` | `64` | Transformer embedding dimensionality $D$. |
| `num_heads` | `int` | `4` | Number of multi-head two-way attention heads. |
| `num_mask_tokens` | `int` | `3` | Number of multi-scale promptable mask hypotheses generated. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

### 19.4 Contrastive Language-Audio Pretraining (`CLAP`)

```python
import numpy as np
from chokkhu.models.multimodal import CLAP

clap = CLAP(embed_dim=16, n_mels=16, vocab_size=50, init_temperature=0.07, seed=42)

# Paired audio spectrograms (B, n_mels, time_steps) and text token sequences (B, seq_len)
audio_spectrograms = np.random.randn(2, 16, 32).astype(np.float32)
text_tokens = np.random.randint(0, 50, size=(2, 8))

# Joint multimodal forward pass & contrastive loss
audio_emb, text_emb, loss = clap.forward(audio_spectrograms, text_tokens)
similarity_matrix = clap.predict_similarity(audio_spectrograms, text_tokens)
print(f"CLAP Multimodal Loss: {loss:.4f}, Zero-Shot Audio-Text Cosine Similarity:\n{similarity_matrix}")
```

#### Parameter Breakdown: `CLAP`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `embed_dim` | `int` | `64` | Joint multimodal shared latent space dimension. |
| `n_mels` | `int` | `64` | Number of mel-spectrogram frequency filterbank bins. |
| `vocab_size` | `int` | `500` | Text tokenizer vocabulary size. |
| `init_temperature` | `float` | `0.07` | Learnable InfoNCE contrastive temperature scale $\tau$. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

### 19.5 Speech Recognition & Autoregressive Decoding (`Whisper`)

```python
import numpy as np
from chokkhu.models.audio.architectures import Whisper

whisper = Whisper(
    n_mels=16,
    vocab_size=25,
    d_model=16,
    n_heads=2,
    n_encoder_layers=1,
    n_decoder_layers=1,
    seed=42,
)

# Audio mel-spectrogram and target token sequences
mel_input = np.random.randn(2, 16, 32).astype(np.float32)
target_tokens = np.array([[1, 5, 2], [1, 7, 2]], dtype=np.int32)

# 1. Forward teacher-forced cross-entropy logits
logits = whisper.forward(mel_input, target_tokens)
print(f"Whisper Logits Shape: {logits.shape}")

# 2. Autoregressive Speech Transcription Generation
transcribed_tokens = whisper.generate(mel_input, max_len=4, prompt_tokens=[1])
print(f"Generated Transcription Sequence: {transcribed_tokens}")
```

#### Parameter Breakdown: `Whisper`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_mels` | `int` | `80` | Number of mel-frequency channels in input audio spectrogram. |
| `vocab_size` | `int` | `500` | Vocabulary size for speech tokens and special symbols. |
| `d_model` | `int` | `64` | Transformer hidden dimension $D$. |
| `n_heads` | `int` | `4` | Number of parallel attention heads. |
| `n_encoder_layers` | `int` | `2` | Number of audio transformer encoder blocks. |
| `n_decoder_layers` | `int` | `2` | Number of causal cross-attention decoder blocks. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

### 19.6 Latent Diffusion & Cross-Attention Conditioning (`LatentDiffusionModel`)

```python
import numpy as np
from chokkhu.models.generative import LatentDiffusionModel

ldm = LatentDiffusionModel(
    latent_dim=8,
    context_dim=8,
    num_timesteps=50,
    beta_start=0.0001,
    beta_end=0.02,
    num_heads=2,
    seed=42,
)

# Clean latent state z_0 and conditioning context vector
z_0 = np.random.randn(2, 8).astype(np.float32)
timesteps = np.array([5, 20], dtype=np.int32)

# 1. Forward diffusion: add calibrated Gaussian noise at timestep t
z_t, injected_noise = ldm.q_sample(z_0, timesteps)

# 2. Reverse noise prediction conditioned on cross-attention context
predicted_noise = ldm.predict_noise(z_t, timesteps, context=z_0)

# 3. Deterministic DDIM fast generation sampling from pure Gaussian noise
generated_latents = ldm.sample(shape=(2, 8), context=z_0, n_steps=5)
print(f"Generated Latent Representations Shape: {generated_latents.shape}")
```

#### Parameter Breakdown: `LatentDiffusionModel`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `latent_dim` | `int` | `32` | Dimension of compressed VAE latent state $z$. |
| `context_dim` | `int` | `64` | Dimension of conditioning prompt/text representations. |
| `num_timesteps` | `int` | `1000` | Total number of discrete forward diffusion noise steps $T$. |
| `beta_start` | `float` | `1e-4` | Initial variance schedule value $\beta_1$. |
| `beta_end` | `float` | `0.02` | Terminal variance schedule value $\beta_T$. |
| `num_heads` | `int` | `4` | Number of heads in cross-attention conditioning block. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

## 20. Sovereign Production Generative AI, Audio Codecs, Speculative LLMs & Genetic AutoML

### 20.1 Spatial Conditional Control with Zero-Convolution (`ControlNet`)

```python
import numpy as np
from chokkhu import ControlNet

# Initialize ControlNet adapter with zero-initialized convolution layers
cnet = ControlNet(
    in_channels=4,
    cond_channels=3,
    base_channels=8,
    num_stages=2,
    seed=42,
)

# Latent feature representation and spatial conditioning guide (e.g., edge map, depth)
latents = np.random.randn(1, 8, 16, 16).astype(np.float32)
condition = np.random.randn(1, 3, 16, 16).astype(np.float32)

# Forward pass injects condition while preserving base model representations
out_latents, stage_residuals = cnet.forward(latents, condition)
print(f"Modulated Latents Shape: {out_latents.shape}, Intermediate Stages: {len(stage_residuals)}")
```

#### Parameter Breakdown: `ControlNet`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | `4` | Number of channels in base input latent representation. |
| `cond_channels` | `int` | `3` | Number of channels in spatial condition image/map. |
| `base_channels` | `int` | `32` | Internal channel capacity of feature backbone and ZeroConv layers. |
| `num_stages` | `int` | `3` | Number of cascaded Zero-Convolution conditioning stages. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

### 20.2 Unpaired Image-to-Image Domain Translation (`CycleGAN`)

```python
import numpy as np
from chokkhu import CycleGAN

# Initialize bidirectional domain translation network (A <-> B)
cyclegan = CycleGAN(
    channels_a=3,
    channels_b=3,
    base_channels=8,
    lambda_cycle=10.0,
    lambda_identity=5.0,
    seed=42,
)

# Domain A and Domain B unaligned sample batches
real_a = np.random.randn(1, 3, 16, 16).astype(np.float32)
real_b = np.random.randn(1, 3, 16, 16).astype(np.float32)

# Bidirectional translations
fake_b = cyclegan.translate_a2b(real_a)
fake_a = cyclegan.translate_b2a(real_b)

# Full training pass computing cycle consistency and adversarial losses
metrics = cyclegan.forward(real_a, real_b)
print(f"Cycle Loss: {metrics['loss_cycle']:.4f}, Total Gen Loss: {metrics['total_loss']:.4f}")
```

#### Parameter Breakdown: `CycleGAN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `channels_a` | `int` | `3` | Number of image channels in domain A. |
| `channels_b` | `int` | `3` | Number of image channels in domain B. |
| `base_channels` | `int` | `16` | Base filter depth in ResNet generators and PatchGAN discriminators. |
| `lambda_cycle` | `float` | `10.0` | Weight scalar for L1 cycle consistency reconstruction loss. |
| `lambda_identity` | `float` | `5.0` | Weight scalar for identity preservation loss. |
| `seed` | `int` | `42` | Random seed for reproducibility. |

---

### 20.3 Neural Audio Compression & Discrete Codecs (`ResidualVectorQuantizer`)

```python
import numpy as np
from chokkhu import ResidualVectorQuantizer

# 4-stage residual vector quantizer (SoundStream / EnCodec style)
rvq = ResidualVectorQuantizer(
    num_quantizers=4,
    codebook_size=32,
    embed_dim=8,
    commitment_weight=0.25,
    seed=42,
)

# Continuous audio frame representations (Batch, Time, Dim)
audio_embeddings = np.random.randn(2, 10, 8).astype(np.float32)

# 1. Multi-stage hierarchical quantization
quantized, codes, commitment_loss = rvq.forward(audio_embeddings)
print(f"Discrete Code Tokens Shape: {codes.shape}, Loss: {commitment_loss:.4f}")

# 2. Reconstruct continuous latents from discrete indices
reconstructed = rvq.decode_codes(codes)
print(f"Reconstructed Latents Match Original Shape: {reconstructed.shape == audio_embeddings.shape}")
```

#### Parameter Breakdown: `ResidualVectorQuantizer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `num_quantizers` | `int` | `4` | Number of cascaded residual quantization stages ($N_q$). |
| `codebook_size` | `int` | `1024` | Number of discrete code vectors per stage codebook ($K$). |
| `embed_dim` | `int` | `64` | Embedding dimensionality of continuous latent vectors ($D$). |
| `commitment_weight` | `float` | `0.25` | Scalar multiplier for codebook commitment loss. |
| `seed` | `int` | `42` | Random seed for codebook initialization. |

---

### 20.4 Multi-Cue Voice Activity Detection (`VoiceActivityDetector`)

```python
import numpy as np
from chokkhu import VoiceActivityDetector

# Multi-cue VAD with Short-Time Energy, ZCR, and Spectral Flux
vad = VoiceActivityDetector(
    sample_rate=16000,
    frame_length_ms=25.0,
    hop_length_ms=10.0,
    energy_threshold=0.02,
)

# Generate synthetic audio with speech tone and silence
t = np.linspace(0, 0.5, 8000)
tone = np.sin(2 * np.pi * 440 * t) * 0.5
silence = np.zeros(4000)
audio = np.concatenate([silence, tone, silence])

# Run VAD speech segmentation
vad_result = vad.detect(audio)
print(f"Speech Ratio: {vad_result['speech_ratio']:.2%}")
print(f"Detected Speech Intervals (sec): {vad_result['segments']}")
```

#### Parameter Breakdown: `VoiceActivityDetector`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `sample_rate` | `int` | `16000` | Sampling frequency of the input audio signal in Hz. |
| `frame_length_ms` | `float` | `25.0` | Duration of each analysis frame in milliseconds. |
| `hop_length_ms` | `float` | `10.0` | Hop step between consecutive frames in milliseconds. |
| `energy_threshold` | `float` | `0.02` | Sensitivity threshold for speech classification. |
| `zcr_threshold` | `float` | `0.15` | Zero-Crossing Rate threshold for unvoiced speech discrimination. |
| `min_speech_duration_ms` | `float` | `100.0` | Minimum duration required to trigger active speech state. |
| `min_silence_duration_ms` | `float` | `200.0` | Minimum silence duration required to terminate active speech state. |

---

### 20.5 High-Speed Speculative & Grammar-Constrained LLM Decoding (`SpeculativeDecoder`, `StructuredJSONDecoder`)

```python
import numpy as np
from chokkhu import SpeculativeDecoder, StructuredJSONDecoder
from chokkhu.core.tensor import Tensor

# Define mock target and draft model logit functions
def mock_target_model(tensor_in):
    logits = np.zeros((1, 1, 30), dtype=np.float32)
    logits[0, 0, 5] = 5.0
    return Tensor(logits, requires_grad=False)

def mock_draft_model(tensor_in):
    return Tensor(np.ones((1, 1, 30), dtype=np.float32), requires_grad=False)

class SimpleTokenizer:
    def encode(self, text, bos=True): return [1, 2, 3]
    def decode(self, tokens, skip_special_tokens=True): return '{"status": "success", "result": "generated text"}'

tok = SimpleTokenizer()

# 1. Speculative Decoding for 2x-3x speedup via draft-verification sampling
spec_dec = SpeculativeDecoder(
    target_model=mock_target_model,
    draft_model=mock_draft_model,
    tokenizer=tok,
    gamma=3,
    temperature=1.0,
)
spec_result = spec_dec.generate("Analyze this transaction:", max_new_tokens=4, eos_token_id=None)
print(f"Speculative Output: {spec_result['text']}, Acceptance Rate: {spec_result['acceptance_rate']:.2%}")

# 2. Grammar-Constrained Structured JSON Generation
json_dec = StructuredJSONDecoder(model=mock_target_model, tokenizer=tok)
json_result = json_dec.generate(
    "Extract user data",
    schema={"properties": {"name": {"type": "string"}, "age": {"type": "number"}}},
    max_new_tokens=5,
)
print(f"Parsed JSON Valid: {json_result['parsed_json']}")
```

#### Parameter Breakdown: `SpeculativeDecoder`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `target_model` | `Any` | *Required* | Main target high-capacity language model. |
| `draft_model` | `Any` | *Required* | Lightweight draft model for fast token generation. |
| `tokenizer` | `Any` | *Required* | Tokenizer instance providing `encode` and `decode`. |
| `gamma` | `int` | `4` | Number of speculative draft tokens generated per iteration. |
| `temperature` | `float` | `1.0` | Sampling temperature parameter. |

---

### 20.6 Non-Linear Manifold Learning & Dimension Reduction (`UMAP`)

```python
import numpy as np
from chokkhu import UMAP

# Create high-dimensional clustered dataset
np.random.seed(42)
c1 = np.random.randn(25, 10) + 3.0
c2 = np.random.randn(25, 10) - 3.0
X = np.vstack([c1, c2])

# Initialize pure NumPy UMAP reducer
umap = UMAP(
    n_components=2,
    n_neighbors=5,
    min_dist=0.1,
    n_epochs=50,
    random_state=42,
)

# Project to 2D manifold preserving both local and global topology
embedding = umap.fit_transform(X)
print(f"UMAP 2D Projection Shape: {embedding.shape}")
```

#### Parameter Breakdown: `UMAP`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_components` | `int` | `2` | Dimensionality of embedded target manifold. |
| `n_neighbors` | `int` | `15` | Size of local fuzzy simplicial metric neighborhood. |
| `min_dist` | `float` | `0.1` | Minimum distance between embedded points in low-dimensional space. |
| `n_epochs` | `int` | `200` | Number of SGD cross-entropy layout optimization iterations. |
| `learning_rate` | `float` | `1.0` | Initial learning rate for manifold layout SGD. |
| `random_state` | `int | None` | `None` | Seed for reproducibility. |

---

### 20.7 Evolutionary Pipeline Search & Dynamic Classifier Selection (`GeneticPipelineSearch`, `DynamicEnsembleSelection`)

```python
import numpy as np
from chokkhu import GeneticPipelineSearch, DynamicEnsembleSelection

# Synthetic classification dataset
np.random.seed(42)
X = np.random.randn(40, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 1. Genetic TPOT-Style Pipeline Evolution
search = GeneticPipelineSearch(
    population_size=6,
    generations=3,
    mutation_rate=0.2,
    crossover_rate=0.5,
    cv_folds=2,
    seed=42,
)
search.fit(X, y)
print(f"Best Evolved Pipeline: {search.best_individual.scaler_type} -> PCA({search.best_individual.pca_components}) -> {search.best_individual.model_type}")

# 2. Dynamic Ensemble Selection (KNORA-Eliminate)
pool = [
    lambda data: (data[:, 0] > 0).astype(int),
    lambda data: (data[:, 1] > 0).astype(int),
]
des = DynamicEnsembleSelection(pool_classifiers=pool, k_neighbors=3, method="knora_e")
des.fit(X, y)
des_predictions = des.predict(X)
print(f"Dynamic Selection Predictions Shape: {des_predictions.shape}")
```

#### Parameter Breakdown: `GeneticPipelineSearch`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `population_size` | `int` | `12` | Number of candidate pipeline chromosomes per generation. |
| `generations` | `int` | `4` | Number of evolutionary iterations. |
| `mutation_rate` | `float` | `0.3` | Probability of mutating individual pipeline hyperparameters. |
| `crossover_rate` | `float` | `0.5` | Probability of chromosome crossover reproduction. |
| `tournament_size` | `int` | `3` | Number of individuals competing in Pareto tournament selection. |
| `cv_folds` | `int` | `3` | Cross-validation folds used for individual fitness scoring. |
| `seed` | `int` | `42` | Random seed for evolution reproducibility. |

---

### 20.8 Advanced Density & Robust Outlier Detection (`LocalOutlierFactor`, `EllipticEnvelope`)

```python
import numpy as np
from chokkhu import LocalOutlierFactor, EllipticEnvelope

# Synthetic data with anomalies
np.random.seed(42)
inliers = np.random.randn(40, 2)
outliers = np.random.uniform(low=-8, high=8, size=(4, 2))
X = np.vstack([inliers, outliers])

# 1. Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1, novelty=True)
lof_labels = lof.fit_predict(X)
print(f"LOF Detected Outliers Count: {np.sum(lof_labels == -1)}")

# 2. Elliptic Envelope (FastMCD Robust Covariance)
ee = EllipticEnvelope(contamination=0.1, random_state=42)
ee.fit(X)
ee_labels = ee.predict(X)
print(f"EllipticEnvelope Detected Outliers Count: {np.sum(ee_labels == -1)}")
```

#### Parameter Breakdown: `LocalOutlierFactor`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_neighbors` | `int` | `20` | Number of nearest neighbors used for local reachability density. |
| `contamination` | `float` | `0.1` | Expected proportion of outliers in the dataset $(0, 0.5]$. |
| `metric` | `str` | `'euclidean'` | Distance metric (`'euclidean'`, `'manhattan'`, `'cosine'`). |
| `novelty` | `bool` | `False` | Enable prediction and scoring on unseen testing data. |

## 21. Sovereign Edge-to-Cloud Serving, Hybrid State-Space, Continual Learning & Optimal Transport (Milestone 18)

### 21.1 Sovereign Edge-to-Cloud Serving & Paged Attention (`PagedKVCache`, `PagedAttention`, `ContinuousBatcher`)

```python
import numpy as np
from chokkhu import PagedKVCache, PagedAttention, ContinuousBatcher
from chokkhu.core.tensor import Tensor

# 1. Paged KV Memory Pool (vLLM-style non-contiguous block allocation)
cache = PagedKVCache(num_blocks=32, block_size=4, num_heads=4, head_dim=16)
seq_id = "seq_42"
cache.allocate_sequence(seq_id, num_tokens=0)

# Simulate streaming KV writes
for t in range(10):
    k_t = np.random.randn(4, 16).astype(np.float32)
    v_t = np.random.randn(4, 16).astype(np.float32)
    cache.append_token(seq_id, k_t, v_t)

# 2. Paged Attention Kernel Computation
attn = PagedAttention()
query = np.random.randn(4, 16).astype(np.float32)
context = attn.forward(query, seq_id, cache)
print(f"Paged Attention Output Context Shape: {context.shape}")
stats = cache.memory_usage()
print(f"Cache Utilization: {stats['utilization']:.1%}, Used Blocks: {stats['used_blocks']}")

# 3. Continuous Batching Iteration Loop
class MockTokenizer:
    def encode(self, text, bos=True): return [1, 5, 8]
    def decode(self, tokens, skip_special_tokens=True): return "Generated response text."

def mock_llm(tensor_in):
    return Tensor(np.random.randn(1, 1, 50).astype(np.float32), requires_grad=False)

batcher = ContinuousBatcher(model=mock_llm, tokenizer=MockTokenizer(), max_batch_size=4)
req1 = batcher.add_request("Explain quantum gravity", max_new_tokens=4)
req2 = batcher.add_request("Write python quicksort", max_new_tokens=3)
batch_results = batcher.generate_all()
print(f"Continuous Batching Completed {len(batch_results)} Requests.")
```

#### Parameter Breakdown: `PagedKVCache`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `num_blocks` | `int` | `128` | Total number of preallocated physical memory blocks in the global pool. |
| `block_size` | `int` | `16` | Number of token slots per physical memory block. |
| `num_heads` | `int` | `4` | Number of attention key/value heads. |
| `head_dim` | `int` | `32` | Dimensionality per attention head. |

#### Parameter Breakdown: `ContinuousBatcher`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `model` | `Any` | *Required* | Causal language model callable accepting input token tensors. |
| `tokenizer` | `Any` | *Required* | Tokenizer instance providing `encode` and `decode` methods. |
| `cache` | `PagedKVCache | None` | `None` | Optional preallocated `PagedKVCache` instance. |
| `max_batch_size` | `int` | `8` | Maximum concurrent active decoding sequences scheduled per iteration. |

---

### 21.2 Structured State-Space Duality & Hybrid LLM Architectures (`Mamba2SSD`, `JambaHybridBlock`)

```python
import numpy as np
from chokkhu import Mamba2SSD, JambaHybridBlock

# 1. Mamba-2 Structured State Space Duality (SSD) Layer
mamba2 = Mamba2SSD(d_model=32, d_state=16, d_conv=4, expand=2, seed=42)
tokens_emb = np.random.randn(2, 8, 32).astype(np.float32)
mamba_out = mamba2.forward(tokens_emb)
print(f"Mamba-2 SSD Forward Output Shape: {mamba_out.shape}")

# 2. Jamba Hybrid (Mamba-2 SSM + MHA + Sparse MoE FFN)
jamba = JambaHybridBlock(d_model=32, num_heads=4, num_experts=4, top_k=2, seed=42)
jamba_out = jamba.forward(tokens_emb)
print(f"Jamba Hybrid Layer Output Shape: {jamba_out.shape}")
```

#### Parameter Breakdown: `Mamba2SSD`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `d_model` | `int` | `64` | Model input and output feature dimensionality. |
| `d_state` | `int` | `64` | Latent SSM state dimension ($N$). |
| `d_conv` | `int` | `4` | 1D causal convolution receptive field width. |
| `expand` | `int` | `2` | Internal hidden expansion ratio ($d_{inner} = d_{model} 	imes 	ext{expand}$). |
| `chunk_size` | `int` | `8` | Block chunk size for 1D SSD matrix multiplication. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

#### Parameter Breakdown: `JambaHybridBlock`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `d_model` | `int` | `64` | Input and output feature dimensionality. |
| `num_heads` | `int` | `4` | Number of multi-head self-attention heads. |
| `num_experts` | `int` | `4` | Number of parallel feed-forward expert MLPs. |
| `top_k` | `int` | `2` | Number of active experts routed per token. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

---

### 21.3 Continual Lifelong Learning & Catastrophic Forgetting Mitigation (`ElasticWeightConsolidation`, `DarkExperienceReplay`)

```python
import numpy as np
from chokkhu import ElasticWeightConsolidation, DarkExperienceReplay

# 1. Elastic Weight Consolidation (Fisher Quadratic Constraint)
class SimpleClassifier:
    def __init__(self):
        self.weights = np.random.randn(4, 2).astype(np.float32)
    def predict(self, x):
        return np.dot(x, self.weights)

model_cont = SimpleClassifier()
X_task1 = np.random.randn(30, 4).astype(np.float32)
y_task1 = np.random.randint(0, 2, size=30)

ewc = ElasticWeightConsolidation(importance=50.0)
ewc.register_task(model_cont, X_task1, y_task1)

# Compute EWC consolidation loss after slight parameter update
model_cont.weights += 0.05
pen_loss = ewc.penalty_loss(model_cont)
print(f"EWC Regularization Penalty: {pen_loss:.4f}")

# 2. Dark Experience Replay (DER++ Knowledge Distillation)
der = DarkExperienceReplay(buffer_capacity=100, alpha=0.5, beta=0.5)
logits_task1 = model_cont.predict(X_task1)
der.add_batch(X_task1, y_task1, logits_task1)

# Sample mini-batch of past exemplars and calculate DER++ loss
buf_x, buf_y, buf_logits = der.sample(batch_size=8)
student_logits = model_cont.predict(buf_x)
total_loss = der.compute_loss(current_task_loss=0.85, student_replay_logits=student_logits, teacher_replay_logits=buf_logits)
print(f"DER++ Total Regularized Loss: {total_loss:.4f}")
```

#### Parameter Breakdown: `ElasticWeightConsolidation`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `importance` | `float` | `100.0` | Quadratic penalty multiplier ($\lambda$) weighting historical Fisher Information. |

#### Parameter Breakdown: `DarkExperienceReplay`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `buffer_capacity` | `int` | `500` | Maximum number of exemplar samples and teacher logits stored in reservoir memory. |
| `alpha` | `float` | `0.5` | Weight multiplier for MSE logit distillation loss. |
| `beta` | `float` | `0.5` | Weight multiplier for historical replay classification loss. |
| `seed` | `int` | `42` | Random seed for reservoir replacement sampling. |

---

### 21.4 Heterogeneous & Spatio-Temporal Graph Networks (`HeteroGCN`, `SpatioTemporalGCN`)

```python
import numpy as np
from chokkhu import HeteroGCN, SpatioTemporalGCN

# 1. HeteroGCN Multi-Relational Message Passing
in_dims = {"user": 8, "item": 16}
relations = [("user", "buys", "item"), ("item", "bought_by", "user")]
hetero = HeteroGCN(in_channels_dict=in_dims, out_channels=16, relations=relations, seed=42)

x_nodes = {"user": np.random.randn(6, 8).astype(np.float32), "item": np.random.randn(4, 16).astype(np.float32)}
edge_indices = {
    ("user", "buys", "item"): np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
    ("item", "bought_by", "user"): np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
}
node_embeddings = hetero.forward(x_nodes, edge_indices)
print(f"HeteroGCN User Embeddings: {node_embeddings['user'].shape}, Item: {node_embeddings['item'].shape}")

# 2. SpatioTemporalGCN Spatial Graph & 1D Temporal Convolutions
st_gcn = SpatioTemporalGCN(in_channels=3, out_channels=8, temporal_kernel_size=3, seed=42)
st_tensor = np.random.randn(2, 10, 5, 3).astype(np.float32)  # (Batch, Time, Nodes, Channels)
adj_graph = np.eye(5) + 0.2
st_out = st_gcn.forward(st_tensor, adj_graph)
print(f"Spatio-Temporal Output Tensor Shape: {st_out.shape}")
```

#### Parameter Breakdown: `HeteroGCN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_channels_dict` | `Dict[str, int]` | *Required* | Mapping of node type identifiers to their input feature dimensions. |
| `out_channels` | `int` | *Required* | Output representation dimension for all node types. |
| `relations` | `Sequence[Tuple[str, str, str]]` | *Required* | List of relational triplets: `(src_node_type, relation_type, dst_node_type)`. |
| `aggregate` | `str` | `"sum"` | Message aggregation strategy across multiple relation types (`"sum"` or `"mean"`). |
| `seed` | `int` | `42` | Random seed for relational transformation matrices. |

#### Parameter Breakdown: `SpatioTemporalGCN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | *Required* | Input feature dimensionality per graph node. |
| `out_channels` | `int` | *Required* | Output feature dimensionality per graph node. |
| `temporal_kernel_size` | `int` | `9` | Kernel width across the sequence time dimension. |
| `stride` | `int` | `1` | Stride step for temporal convolution. |
| `seed` | `int` | `42` | Random seed for spatial and temporal convolution weights. |

---

### 21.5 Geometric Optimal Transport & Wasserstein Barycenters (`SinkhornOptimalTransport`, `WassersteinBarycenter`)

```python
import numpy as np
from chokkhu import SinkhornOptimalTransport, WassersteinBarycenter, sinkhorn_distance, wasserstein_barycenter

# 1. Entropic Regularized Optimal Transport Coupling & Distance
X_source = np.random.randn(15, 2)
X_target = np.random.randn(20, 2)
w_dist = sinkhorn_distance(X_source, X_target, reg=0.05)
print(f"Sinkhorn Regularized Wasserstein Distance: {w_dist:.4f}")

# 2. Wasserstein Geometric Barycenter (Frechet Mean of Distributions)
grid = np.linspace(0, 1, 25)
p1 = np.exp(-((grid - 0.2)**2) / 0.02)
p2 = np.exp(-((grid - 0.8)**2) / 0.02)
bary = wasserstein_barycenter([p1, p2], (grid[:, None] - grid[None, :])**2, weights=[0.5, 0.5], reg=0.02)
print(f"Wasserstein Barycenter Computed: Shape={bary.shape}, Sum={np.sum(bary):.2f}")
```

#### Parameter Breakdown: `SinkhornOptimalTransport`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `reg` | `float` | `0.05` | Entropic regularization coefficient ($\epsilon$). |
| `max_iter` | `int` | `100` | Maximum Sinkhorn-Knopp scaling iterations. |
| `tol` | `float` | `1e-6` | Marginal error convergence tolerance. |

#### Parameter Breakdown: `WassersteinBarycenter`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `reg` | `float` | `0.01` | Entropic regularization strength ($\epsilon$). |
| `max_iter` | `int` | `50` | Maximum iterative barycentric scaling iterations. |
| `tol` | `float` | `1e-5` | Convergence tolerance for barycenter updates. |

---

### 21.6 Automated Symbolic Feature Synthesis (`SymbolicFeatureSynthesizer`, `SymbolicProgram`)

```python
import numpy as np
from chokkhu import SymbolicFeatureSynthesizer

X_data = np.random.randn(40, 4)
y_target = X_data[:, 0] * X_data[:, 1] + np.sin(X_data[:, 2])

synthesizer = SymbolicFeatureSynthesizer(n_features=3, generations=3, population_size=15, random_state=42)
synthesizer.fit(X_data, y_target, feature_names=["f1", "f2", "f3", "f4"])
evolved_features = synthesizer.transform(X_data)
print(f"Synthesized Features Shape: {evolved_features.shape}")
print(f"Discovered Formulas: {synthesizer.get_feature_names()}")
```

#### Parameter Breakdown: `SymbolicFeatureSynthesizer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_features` | `int` | `5` | Number of top synthesized feature columns retained. |
| `generations` | `int` | `4` | Number of evolutionary genetic programming generations. |
| `population_size` | `int` | `20` | Candidate expression trees per generation. |
| `tournament_size` | `int` | `3` | Size of Pareto tournament selection pool. |
| `random_state` | `int | None` | `42` | Random seed for evolution reproducibility. |

---

## 22. Robotics, World Models, Multi-Agent RL, Genomic AI & Hyperdimensional Computing (Milestone 19)

### 22.1 Robotics, Diffusion Policy & Latent World Models (`DiffusionPolicy`, `RecurrentWorldModel`, `MPPITrajectoryOptimizer`)

```python
import numpy as np
from chokkhu import DiffusionPolicy, RecurrentWorldModel, MPPITrajectoryOptimizer

# 1. Diffusion Policy for Continuous Action Trajectory Generation
policy = DiffusionPolicy(action_dim=4, pred_horizon=6, obs_dim=8, num_timesteps=20, hidden_dim=32, seed=42)
obs_state = np.random.randn(2, 8).astype(np.float32)
action_traj = policy.sample(obs_state, n_samples=2)
print(f"Generated Action Trajectory Shape: {action_traj.shape}")

# 2. Recurrent World Model (RSSM Imagination Rollouts)
world_model = RecurrentWorldModel(obs_dim=8, action_dim=2, deter_dim=16, stoch_dim=8, seed=42)
h0, z0 = world_model.initial_state(batch_size=2)
rollouts = world_model.imagine_trajectory(h0, z0, lambda h, z: np.random.randn(len(h), 2).astype(np.float32), horizon=5)
print(f"Imagined Trajectory Latent States Shape: {rollouts['h'].shape}, Rewards: {rollouts['rewards'].shape}")

# 3. Model Predictive Path Integral (MPPI) Trajectory Optimizer
mppi = MPPITrajectoryOptimizer(action_dim=2, horizon=8, num_samples=32, temperature=1.0, seed=42)
best_action, planned_traj = mppi.optimize(np.zeros(4), lambda s, a: np.mean((a - np.array([1.0, -0.5]))**2, axis=(1, 2)))
print(f"MPPI Optimal Control Action: {best_action}, Planned Horizon Shape: {planned_traj.shape}")
```

#### Parameter Breakdown: `DiffusionPolicy`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `action_dim` | `int` | `4` | Dimensionality of robotic action control vector ($D_a$). |
| `pred_horizon` | `int` | `8` | Lookahead planning horizon ($T_a$). |
| `obs_dim` | `int` | `16` | Observation feature conditioning dimensionality. |
| `num_timesteps` | `int` | `50` | Number of reverse diffusion denoising steps. |
| `seed` | `int` | `42` | Random seed for weight initialization. |

#### Parameter Breakdown: `RecurrentWorldModel`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `obs_dim` | `int` | `16` | Observation feature dimensionality. |
| `action_dim` | `int` | `4` | Continuous control action dimension. |
| `deter_dim` | `int` | `32` | Deterministic GRU recurrent state size ($h_t$). |
| `stoch_dim` | `int` | `16` | Stochastic latent Gaussian state size ($z_t$). |
| `seed` | `int` | `42` | Random seed for latent transition parameters. |

#### Parameter Breakdown: `MPPITrajectoryOptimizer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `action_dim` | `int` | *Required* | Dimensionality of control action vector. |
| `horizon` | `int` | `15` | Trajectory planning lookahead steps ($T$). |
| `num_samples` | `int` | `64` | Number of parallel perturbation rollouts ($K$). |
| `temperature` | `float` | `1.0` | Return weighting temperature ($\lambda$). |
| `noise_sigma` | `float` | `0.5` | Exploration perturbation standard deviation. |

---

### 22.2 Multi-Agent Reinforcement Learning & Game Theory (`QMIX`, `VDN`, `NashEquilibriumSolver`)

```python
import numpy as np
from chokkhu import QMIX, VDN, NashEquilibriumSolver

# 1. QMIX Monotonic Value Factorization
qmix = QMIX(n_agents=3, state_dim=12, mixing_embed_dim=16, seed=42)
agent_qs = np.random.randn(4, 3).astype(np.float32)
global_states = np.random.randn(4, 12).astype(np.float32)
q_tot = qmix.forward(agent_qs, global_states)
print(f"QMIX Centralized Joint Q_tot Shape: {q_tot.shape}")

# 2. VDN Additive Utility Decomposition
vdn = VDN(n_agents=3)
vdn_tot = vdn.forward(agent_qs)
print(f"VDN Joint Team Value: {vdn_tot[:2].flatten()}")

# 3. Game-Theoretic Nash Equilibrium Solver (Zero-Sum & General-Sum)
solver = NashEquilibriumSolver(max_iter=500)
rps_payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
p_row, p_col, val = solver.solve_zero_sum(rps_payoff)
print(f"Nash Equilibrium Mixed Strategy: Row={p_row.round(2)}, Col={p_col.round(2)}, Value={val:.2f}")
```

#### Parameter Breakdown: `QMIX` & `VDN`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `n_agents` | `int` | `4` | Number of cooperative agents in multi-agent team. |
| `state_dim` | `int` | `32` | Global environmental state dimension ($S$). |
| `mixing_embed_dim` | `int` | `32` | Internal hypernetwork mixing capacity. |
| `seed` | `int` | `42` | Random seed for hypernetwork weights. |

---

### 22.3 Genomic & Bio-Molecular Sequence AI (`GenomicTokenizer`, `GenomicBERT`, `ProteinContactMap`)

```python
import numpy as np
from chokkhu import GenomicTokenizer, GenomicBERT, ProteinContactMap

# 1. DNA/RNA k-mer Tokenization & Reverse Complement
tok = GenomicTokenizer(k=3, stride=1, is_rna=False)
dna_seq = "ATGCGATCG"
token_ids = tok.encode(dna_seq, add_special_tokens=True)
rev_comp = GenomicTokenizer.reverse_complement("ATGC")
print(f"Encoded Token IDs: {token_ids}, Reverse Complement: {rev_comp}")

# 2. GenomicBERT Sequence Modeling & Variant Effect Scoring
gen_bert = GenomicBERT(vocab_size=len(tok.vocab), d_model=32, num_heads=4, num_layers=2, max_len=32, num_classes=2, seed=42)
hidden, cls_logits, mlm_logits = gen_bert.forward(np.array([token_ids]))
var_score = gen_bert.score_variant(np.array([token_ids]), np.array([token_ids]))
print(f"GenomicBERT Sequence Embeddings: {hidden.shape}, Variant Effect Score: {var_score:.4f}")

# 3. Protein Residue Contact Map via Direct Coupling Analysis (DCA)
dca = ProteinContactMap(pseudocount_weight=0.5, apc=True)
dummy_msa = np.random.randint(0, 20, size=(25, 12))
contact_map = dca.compute_contact_map(dummy_msa)
print(f"Residue Contact Matrix Shape: {contact_map.shape}, Symmetric: {np.allclose(contact_map, contact_map.T)}")
```

#### Parameter Breakdown: `GenomicTokenizer` & `GenomicBERT`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `k` | `int` | `6` | Length of contiguous nucleotide $k$-mer tokens. |
| `stride` | `int` | `1` | Tokenization sliding window step. |
| `is_rna` | `bool` | `False` | Toggle between DNA (T) and RNA (U) alphabets. |
| `d_model` | `int` | `64` | Transformer representation width. |
| `num_heads` | `int` | `4` | Number of self-attention heads. |
| `num_layers` | `int` | `2` | Number of bidirectional encoder layers. |

---

### 22.4 Neuro-Symbolic Reasoning & Knowledge Graph Embeddings (`DifferentiableLogicEngine`, `RotatE`, `TransE`)

```python
import numpy as np
from chokkhu import DifferentiableLogicEngine, RotatE, TransE

# 1. Continuous Fuzzy Differentiable Logic Engine
engine = DifferentiableLogicEngine(t_norm="product")
a_truth = np.array([0.9, 0.4])
b_truth = np.array([0.8, 0.7])
and_val = engine.conjunction(a_truth, b_truth)
implies_val = engine.implication(a_truth, b_truth)
sat_loss = engine.satisfaction_loss(implies_val)
print(f"Fuzzy Conjunction: {and_val}, Implication: {implies_val}, Satisfaction Loss: {sat_loss:.4f}")

# 2. RotatE & TransE Knowledge Graph Embeddings
rotate = RotatE(num_entities=20, num_relations=5, embedding_dim=16, gamma=12.0, seed=42)
transe = TransE(num_entities=20, num_relations=5, embedding_dim=16, margin=1.0, seed=42)
h_ids, r_ids, t_ids = np.array([0, 1]), np.array([2, 3]), np.array([4, 5])
rot_scores = rotate.score_triplets(h_ids, r_ids, t_ids)
trans_scores = transe.score_triplets(h_ids, r_ids, t_ids)
print(f"RotatE Triplet Distance Scores: {rot_scores}, TransE: {trans_scores}")
```

#### Parameter Breakdown: `RotatE` & `TransE`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `num_entities` | `int` | *Required* | Total number of entities in knowledge graph. |
| `num_relations` | `int` | *Required* | Total number of relation types. |
| `embedding_dim` | `int` | `64` | Latent space dimensionality. |
| `gamma` / `margin` | `float` | `12.0` / `1.0` | Distance margin parameter for contrastive ranking loss. |

---

### 22.5 Hyperdimensional Computing & Vector Symbolic Architectures (`HyperdimensionalVector`, `HDCClassifier`)

```python
import numpy as np
from chokkhu import HyperdimensionalVector, HDCClassifier

# 1. Hypervector Primitive Algebra (Binding, Bundling, Permutation)
v1 = HyperdimensionalVector.random_bipolar(dim=2000, seed=42)
v2 = HyperdimensionalVector.random_bipolar(dim=2000, seed=43)
bound = v1.bind(v2)
bundled = HyperdimensionalVector.bundle([v1, v2], binarize=True)
permuted = v1.permute(shift=2)
print(f"HDC Similarity (v1 vs v2): {v1.similarity(v2):.4f}, (v1 vs bundled): {v1.similarity(bundled):.4f}")

# 2. Zero-Gradient One-Shot HDC Associative Memory Classifier
hdc_clf = HDCClassifier(dim=2000, num_levels=16, seed=42)
X_train = np.random.randn(20, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
hdc_clf.fit(X_train, y_train)
acc = hdc_clf.score(X_train, y_train)
print(f"HDC Associative Memory Training Accuracy: {acc:.1%}")
```

#### Parameter Breakdown: `HDCClassifier`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dim` | `int` | `5000` | Dimensionality of hypervector cognitive space ($D$). |
| `num_levels` | `int` | `32` | Number of continuous feature quantization level vectors. |
| `seed` | `int` | `42` | Random seed for orthogonal basis vectors. |

---

## 23. The Grand Frontier: Discrete Diffusion, Differentiable Physics, LLM Safety, Causal DAGs & Hyperbolic Geometry (Milestone 20)

### 23.1 Discrete Text Diffusion Models (`DiscreteTextDiffusion`)

```python
import numpy as np
from chokkhu import DiscreteTextDiffusion

# 1. Initialize Discrete Absorbing-State Categorical Diffusion Model
model = DiscreteTextDiffusion(
    vocab_size=500,
    max_seq_len=32,
    num_timesteps=20,
    d_model=64,
    num_heads=4,
    num_layers=2,
    schedule="linear",
    seed=42
)

# 2. Forward Categorical Token Masking & ELBO Loss Calculation
x_clean = np.random.randint(0, 500, size=(2, 16))
timesteps = np.array([5, 12])
x_corrupted = model.q_sample(x_clean, timesteps)
loss = model.compute_loss(x_clean, timesteps)
print(f"Corrupted Tokens Shape: {x_corrupted.shape}, Variational Loss: {loss:.4f}")

# 3. Ancestral Iterative Reverse Sampling from All [MASK] to Clean Text
generated_tokens = model.sample(batch_size=2, seq_len=12, num_steps=10, temperature=0.8)
print(f"Generated Token Sequences: {generated_tokens.shape}")
```

#### Parameter Breakdown: `DiscreteTextDiffusion`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `1000` | Total discrete vocabulary size. |
| `max_seq_len` | `int` | `64` | Maximum allowable sequence token length. |
| `num_timesteps` | `int` | `50` | Total diffusion Markov chain step count ($T$). |
| `mask_token_id` | `Optional[int]` | `None` | Absorbing mask token ID (defaults to `vocab_size`). |
| `d_model` | `int` | `64` | Bidirectional transformer hidden dimension. |
| `num_heads` | `int` | `4` | Number of self-attention heads. |
| `num_layers` | `int` | `2` | Number of transformer encoder layers. |
| `schedule` | `str` | `"linear"` | Noise transition schedule: `"linear"` or `"cosine"`. |

---

### 23.2 Differentiable Physics & Robotics Kinematics (`RobotArmKinematics`, `DifferentiableParticleFluid`)

```python
import numpy as np
from chokkhu import RobotArmKinematics, DifferentiableParticleFluid

# 1. 2-Link Planar Robot Arm Forward & Damped Least Squares (DLS) Inverse Kinematics
arm = RobotArmKinematics.planar_2d(link_lengths=[1.0, 1.0])
target_pos = np.array([1.0, 1.0, 0.0])
solved_angles, converged, iters = arm.inverse_kinematics(target_position=target_pos, tolerance=1e-3)
actual_pos = arm.get_end_effector_position(solved_angles)
print(f"IK Solved Angles: {solved_angles}, Target: {target_pos[:2]}, Reached: {actual_pos[:2]}")

# 2. Smoothed Particle Hydrodynamics (SPH) Differentiable Fluid Dynamics
fluid = DifferentiableParticleFluid(
    num_particles=30, dim=2, rest_density=1000.0, stiffness=150.0, viscosity=0.05, dt=0.005, seed=42
)
pos_init = np.random.uniform(-0.4, 0.4, size=(30, 2))
pos_hist, vel_hist = fluid.simulate(pos_init, num_steps=20)
print(f"Fluid Particle Trajectory History: {pos_hist.shape}")
```

#### Parameter Breakdown: `RobotArmKinematics` & `DifferentiableParticleFluid`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dh_params` | `np.ndarray` | *Required* | Denavit-Hartenberg parameter table `(N, 4)`: `[a, alpha, d, theta_offset]`. |
| `joint_limits` | `Optional[List[Tuple[float, float]]]` | `None` | Per-joint rotation angle bounds $[-\pi, \pi]$. |
| `num_particles` | `int` | `100` | Total simulated fluid particle count. |
| `rest_density` | `float` | `1000.0` | Equilibrium rest mass density $\rho_0$ ($\text{kg/m}^3$). |
| `stiffness` | `float` | `200.0` | Tait equation of state gas stiffness constant $k$. |
| `viscosity` | `float` | `0.1` | Dynamic viscosity coefficient $\mu$. |

---

### 23.3 AI Safety & LLM Watermarking (`StatisticalTextWatermark`, `RefusalDirectionProbe`)

```python
import numpy as np
from chokkhu import StatisticalTextWatermark, RefusalDirectionProbe

# 1. Statistical Token Watermarking & Z-Score Hypothesis Testing
watermark = StatisticalTextWatermark(vocab_size=1000, gamma=0.5, delta=3.0, hash_key=15485863)
prefix = np.array([42, 108])
logits = np.random.randn(1000)
biased_logits = watermark.apply_watermark_bias(logits, prefix)
det_result = watermark.detect(np.array([42, 108, 205, 312, 401]))
print(f"Watermark Detection Z-Score: {det_result['z_score']:.2f}, Is Watermarked: {det_result['is_watermarked']}")

# 2. Refusal Direction Probe & Representation Steering
probe = RefusalDirectionProbe(dim=64)
harmful_acts = np.random.randn(20, 64) + 1.5
harmless_acts = np.random.randn(20, 64) - 1.5
probe.fit(harmful_acts, harmless_acts)
steered_acts = probe.steer(harmless_acts[:2], alpha=1.0)
ablated_acts = probe.ablate(harmful_acts[:2])
print(f"Refusal Intent Score (Harmful): {probe.score_refusal_intent(harmful_acts[:2])}")
```

#### Parameter Breakdown: `StatisticalTextWatermark` & `RefusalDirectionProbe`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `1000` | LLM token vocabulary size. |
| `gamma` | `float` | `0.5` | Fraction of vocabulary partitioned into the green list ($\gamma \in (0, 1)$). |
| `delta` | `float` | `2.0` | Logit perturbation bias injected into green tokens ($\delta > 0$). |
| `hash_key` | `int` | `15485863` | Cryptographic prime seed for deterministic pseudo-random hashing. |
| `dim` | `int` | `64` | Residual stream activation embedding dimensionality. |

---

### 23.4 Causal DAG Discovery (`NOTEARSCausalDiscovery`, `PCAlgorithm`)

```python
import numpy as np
from chokkhu import NOTEARSCausalDiscovery, PCAlgorithm

# 1. Continuous Structure Learning with Smooth Matrix Exponential Acyclicity (NOTEARS)
rng = np.random.RandomState(42)
n_samples = 300
x0 = rng.randn(n_samples)
x1 = 0.7 * x0 + 0.3 * rng.randn(n_samples)
x2 = 0.8 * x1 + 0.2 * rng.randn(n_samples)
X_causal = np.column_stack([x0, x1, x2])

notears = NOTEARSCausalDiscovery(lambda1=0.05, max_iter=25, w_threshold=0.2)
notears.fit(X_causal)
print(f"NOTEARS Learned DAG Matrix:\n{notears.adjacency_matrix_}\nIs Strict DAG: {notears.is_dag()}")

# 2. Constraint-Based Conditional Independence Discovery (PC Algorithm)
pc = PCAlgorithm(alpha=0.05)
pc.fit(X_causal)
print(f"PC Algorithm Discovered CPDAG Edges: {pc.get_edges()}")
```

#### Parameter Breakdown: `NOTEARSCausalDiscovery` & `PCAlgorithm`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `lambda1` | `float` | `0.1` | $L_1$ sparsity penalty coefficient on DAG adjacency weights. |
| `max_iter` | `int` | `100` | Maximum Augmented Lagrangian outer optimization iterations. |
| `h_tol` | `float` | `1e-8` | Matrix exponential acyclicity trace tolerance $h(W) = \text{tr}(e^{W \odot W}) - d$. |
| `w_threshold` | `float` | `0.3` | Threshold for pruning spurious small edges. |
| `alpha` | `float` | `0.05` | Statistical significance level for partial correlation Fisher $z$-tests. |

---

### 23.5 Hyperbolic Manifold Geometry (`PoincareBallEmbedding`, `LorentzManifold`)

```python
import numpy as np
from chokkhu import PoincareBallEmbedding, LorentzManifold

# 1. Poincaré Ball Model & Riemannian Graph Representation Learning
poincare = PoincareBallEmbedding(dim=3, c=1.0, seed=42)
u = np.array([0.2, 0.1, -0.3])
v = np.array([-0.1, 0.4, 0.2])
geo_dist = poincare.distance(u, v)
mob_sum = poincare.mobius_add(u, v)
print(f"Poincare Geodesic Distance: {geo_dist:.4f}, Mobius Addition Norm: {np.linalg.norm(mob_sum):.4f}")

# 2. Lorentz Hyperboloid Model & Isomorphic Diffeomorphisms
lorentz = LorentzManifold(dim=2, c=1.0)
x_poincare = np.array([0.3, -0.2])
x_lorentz = lorentz.poincare_to_lorentz(x_poincare)
minkowski_norm = lorentz.minkowski_dot(x_lorentz, x_lorentz)
print(f"Lorentz Coordinates: {x_lorentz}, Minkowski Constraint <x, x>_L: {minkowski_norm:.4f}")
```

#### Parameter Breakdown: `PoincareBallEmbedding` & `LorentzManifold`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `dim` | `int` | `2` | Hyperbolic manifold spatial embedding dimension $d$. |
| `c` | `float` | `1.0` | Negative sectional curvature parameter ($\kappa = -c$). |
| `eps` | `float` | `1e-5` | Boundary safety margin for numerical ball projection $\|x\| \le 1/\sqrt{c} - \epsilon$. |

---


---

## 24. Multi-Agent Economy, Mechanistic Interpretability, Guided Diffusion Inpainting, Byte Transformers & Differentiable Ranking

### 24.1 Replicator Dynamics & Evolutionary Game Theory (`ReplicatorDynamics`)

Models continuous-time frequency-dependent evolutionary selection and mutator dynamics in multi-agent game-theoretic populations $\dot{x}_i = x_i (f_i(x) - ar{f}(x)) + \sum_j (\mu_{ji} x_j - \mu_{ij} x_i)$ integrated via 4th-order Runge-Kutta (RK4) projected onto the probability simplex.

```python
import numpy as np
from chokkhu import ReplicatorDynamics

# 1. Classical Hawk-Dove Game Simulation
# Payoff: V=2.0 (resource value), C=4.0 (fight cost) -> Evolutionary Stable Strategy (ESS) = [0.5, 0.5]
hd = ReplicatorDynamics.hawks_doves(v=2.0, c=4.0, dt=0.01)
initial_population = np.array([0.9, 0.1])  # 90% Hawks, 10% Doves
pop_trajectory = hd.simulate(initial_population, num_steps=200)
print(f"Hawk-Dove Initial: {initial_population} -> Final ESS: {pop_trajectory[-1]}")

# 2. Rock-Paper-Scissors Cyclic Dynamics
rps = ReplicatorDynamics.rock_paper_scissors(dt=0.01)
rps_trajectory = rps.simulate(np.array([0.5, 0.3, 0.2]), num_steps=100)
print(f"RPS Step 100 Distribution: {rps_trajectory[-1]}")
```

#### Parameter Breakdown: `ReplicatorDynamics`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `payoff_matrix` | `np.ndarray` | *Required* | Square payoff matrix $A \in \mathbb{R}^{K 	imes K}$ where $A_{ij}$ is payoff of strategy $i$ against $j$. |
| `mutation_rate` | `float` | `0.0` | Uniform mutation probability $\mu \in [0, 1)$ between strategies. |
| `dt` | `float` | `0.01` | Continuous numerical integration step size for the RK4 solver. |

---

### 24.2 Combinatorial Bundle Auctions & VCG Mechanism (`CombinatorialAuction`)

Implements the generalized Vickrey-Clarke-Groves (VCG) mechanism for combinatorial multi-item auctions, guaranteeing dominant-strategy incentive compatibility (DSIC / truth-telling) and social welfare maximization with Clarke pivot rule taxation.

```python
from chokkhu import CombinatorialAuction

# Create combinatorial auction for spectrum frequency bands A and B
auction = CombinatorialAuction(items=["A", "B"], bidders=["Alice", "Bob", "Charlie"])

# Agents submit bids over item combinations
auction.add_bid("Alice", ["A", "B"], 10.0)  # Alice bids $10 for the combined bundle {A, B}
auction.add_bid("Bob", ["A"], 6.0)           # Bob bids $6 for band A alone
auction.add_bid("Charlie", ["B"], 5.0)       # Charlie bids $5 for band B alone

# Solve Winner Determination Problem (WDP) & compute Clarke pivot payments
result = auction.solve()
print(f"Winning Allocations: {result['allocations']}")
print(f"Clarke Pivot Payments: {result['payments']}")
print(f"Total Social Welfare: {result['social_welfare']}")
```

#### Parameter Breakdown: `CombinatorialAuction`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `items` | `List[str]` | *Required* | List of distinct item and good identifiers up for auction. |
| `bidders` | `List[str]` | *Required* | List of participating agent/bidder identifiers. |

---

### 24.3 Mechanistic Interpretability & Interventional Causal Patching (`ActivationPatchingEngine`)

Performs interventional causal tracing and activation patching across transformer layers and attention heads to localize factual knowledge circuits without backpropagation.

```python
import numpy as np
from chokkhu import ActivationPatchingEngine

# Define forward pass with activation caching hook
def model_forward_with_cache(x, patch_dict=None):
    layer_1 = x * 2.0
    if patch_dict and "layer_1" in patch_dict:
        layer_1 = patch_dict["layer_1"]
    logits = np.sum(layer_1, axis=-1, keepdims=True)
    cache = {"layer_1": x * 2.0}
    return logits, cache

engine = ActivationPatchingEngine(model_forward_with_cache)
clean_prompt = np.array([[5.0, 5.0]])
corrupted_prompt = np.array([[1.0, 1.0]])

# Measure normalized causal recovery ratio of patching layer_1
causal_effect = engine.compute_causal_effect(clean_prompt, corrupted_prompt, "layer_1")
print(f"Layer 1 Causal Mediation Effect: {causal_effect:.4f}")

# Full circuit sweep across all model layers
circuit_map = engine.trace_circuit(clean_prompt, corrupted_prompt, ["layer_1"])
print(f"Circuit Trace Attribution: {circuit_map}")
```

#### Parameter Breakdown: `ActivationPatchingEngine`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `forward_with_cache` | `Callable` | *Required* | Function accepting `(x, patch_dict)` returning `(logits, cache)`. |
| `clean_input` | `np.ndarray` | *Required* | Clean prompt input that produces correct ground-truth outputs. |
| `corrupted_input` | `np.ndarray` | *Required* | Corrupted/perturbed input where factual recall fails. |
| `layer_name` | `str` | *Required* | Hook identifier of the layer/head activation tensor to intervene upon. |

---

### 24.4 Classifier-Free Guidance & Masked Diffusion Inpainting (`ClassifierFreeGuidance`, `DiffusionInpainter`)

Enables controllable generation via Classifier-Free Guidance (CFG) score extrapolation $	ilde{\epsilon}_	heta(x_t, c) = \epsilon_	heta(x_t, \emptyset) + s (\epsilon_	heta(x_t, c) - \epsilon_	heta(x_t, \emptyset))$ and RePaint masked diffusion inpainting with boundary time-travel harmonization.

```python
import numpy as np
from chokkhu import ClassifierFreeGuidance, DiffusionInpainter

# 1. Classifier-Free Guidance (CFG) Noise Extrapolation
cfg = ClassifierFreeGuidance(guidance_scale=3.5)
unconditional_noise = np.array([0.1, 0.2])
conditional_noise = np.array([0.5, 0.8])
guided_noise = cfg.combine_scores(unconditional_noise, conditional_noise)
print(f"CFG Extrapolated Noise Estimate: {guided_noise}")

# 2. RePaint Masked Diffusion Inpainting Engine
class DiffusionScoreModel:
    num_timesteps = 10
    alphas_cumprod = np.linspace(0.99, 0.05, 10)
    def predict_noise(self, x_t, t): return 0.05 * x_t

inpainter = DiffusionInpainter(DiffusionScoreModel(), num_resample_steps=2, seed=42)
clean_image = np.ones((1, 8, 8))
mask = np.ones((1, 8, 8))
mask[:, 2:6, 2:6] = 0.0  # Mask out central 4x4 hole (0.0 = hole to inpaint)

inpainted_result = inpainter.inpaint(clean_image, mask, num_steps=5)
print(f"Inpainted Image Output Shape: {inpainted_result.shape}")
```

#### Parameter Breakdown: `ClassifierFreeGuidance` & `DiffusionInpainter`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `guidance_scale` | `float` | `3.0` | Conditioning scale multiplier $s$ ($s > 1.0$ amplifies conditional steering). |
| `ddpm_model` | `Any` | *Required* | Base diffusion model providing noise schedule and score evaluation. |
| `num_resample_steps` | `int` | `1` | Time-travel resampling passes per reverse timestep for boundary harmonization. |

---

### 24.5 Token-Free Raw UTF-8 Byte Transformer (`ByteTransformer`)

Operates directly on raw binary byte streams ($0 - 255$) with zero tokenization artifacts, eliminating out-of-vocabulary failures via local patch convolutional pooling, multi-head self-attention, and byte-level unpooling.

```python
import numpy as np
from chokkhu import ByteTransformer

# Initialize Sovereign Byte Transformer (ByT5 / MambaByte architecture)
byte_model = ByteTransformer(
    d_model=32, patch_size=2, num_heads=2, num_layers=1, max_seq_len=64, seed=42
)

# Encode arbitrary raw UTF-8 string directly into uint8 byte array
text = "Hello Sovereign AI! 🇧🇩"
byte_array = ByteTransformer.text_to_bytes(text)
print(f"Raw UTF-8 Bytes: {byte_array[:6]}... Decoded: {ByteTransformer.bytes_to_text(byte_array)}")

# Next-byte probability distribution prediction
logits = byte_model.forward(np.array([byte_array[:8]]))
print(f"Next-Byte Output Logits Tensor: {logits.shape}")

# Autoregressive generation directly in raw byte space
generated_str = byte_model.generate("AI", max_new_bytes=4, temperature=0.8)
print(f"Generated Continuation: {repr(generated_str)}")
```

#### Parameter Breakdown: `ByteTransformer`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `d_model` | `int` | `64` | Internal transformer representation dimensionality. |
| `patch_size` | `int` | `4` | Local byte compression factor per patch projection. |
| `num_heads` | `int` | `4` | Number of multi-head self-attention heads. |
| `num_layers` | `int` | `2` | Number of stacked transformer layers. |
| `max_seq_len` | `int` | `256` | Maximum allowable raw byte sequence length. |

---

### 24.6 Differentiable Neural Sorting & Soft Ranking Losses (`NeuralSort`, `DifferentiableRankingLoss`)

Provides continuous relaxations of the non-differentiable $	ext{argsort}$ operator via temperature-controlled doubly-stochastic permutation matrices $P_	au(s)$ alongside differentiable Spearman rank correlation and Soft NDCG losses for Learning-to-Rank (LTR).

```python
import numpy as np
from chokkhu import NeuralSort, DifferentiableRankingLoss

# 1. NeuralSort Continuous Relaxed Permutations
sorter = NeuralSort(tau=0.5)
continuous_scores = np.array([12.0, 45.0, 23.0])

# Compute continuous 1-based soft ranks
soft_ranks = sorter.soft_ranks(continuous_scores)
print(f"Soft Continuous Ranks: {soft_ranks}")

# Differentiable soft sort
values = np.array([100.0, 500.0, 200.0])
sorted_values = sorter.soft_sort(values, scores=continuous_scores)
print(f"Softly Sorted Output: {sorted_values}")

# 2. Differentiable Surrogate Ranking Losses
pred_scores = np.array([10.0, 50.0, 20.0])
ground_truth_relevance = np.array([1.0, 3.0, 2.0])

spearman_loss = DifferentiableRankingLoss.spearman_loss(pred_scores, ground_truth_relevance, tau=0.5)
ndcg_loss = DifferentiableRankingLoss.soft_ndcg_loss(pred_scores, ground_truth_relevance, tau=0.5)
print(f"Differentiable Spearman Loss: {spearman_loss:.4f} | Soft NDCG Loss: {ndcg_loss:.4f}")
```

#### Parameter Breakdown: `NeuralSort` & `DifferentiableRankingLoss`
| Parameter Name | Data Type | Default Value | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `tau` | `float` | `1.0` | Temperature relaxation parameter ($	au 	o 0$ approaches exact discrete hard sort). |
| `pred_scores` | `np.ndarray` | *Required* | Model output relevance score predictions. |
| `true_relevance` | `np.ndarray` | *Required* | Ground-truth relevance or ranking labels. |


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
