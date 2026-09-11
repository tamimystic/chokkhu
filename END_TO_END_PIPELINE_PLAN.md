# CHOKKHU (চক্ষু) — Master Architectural Blueprint & Universal AI Specification
## The Sovereign All-in-One AI & Machine Learning Pipeline Framework (Grand Unified Universe)
### *"Minimalistic Code. Maximum Output. Zero Heavy Dependencies. Infinite Intelligence."*

---

## 📑 Comprehensive Master Table of Contents

- [1. Executive Vision, 9-Layer Sovereign Architecture & Zero-Dependency Principles](#1-executive-vision-9-layer-sovereign-architecture--zero-dependency-principles)
  - [1.1 The Mission of Chokkhu & The Sovereign Manifesto](#11-the-mission-of-chokkhu--the-sovereign-manifesto)
  - [1.2 The 9 Sovereign Architectural Layers](#12-the-9-sovereign-architectural-layers)
  - [1.3 Strict Operational & Engineering Directives](#13-strict-operational--engineering-directives)
- [2. Complete Sovereign Codebase Directory Map (All 31 Packages & Submodules)](#2-complete-sovereign-codebase-directory-map-all-31-packages--submodules)
- [3. Pillar I: Core Tensor DAG Engine, Dynamic Autograd, Memory Arena & Multi-Modal IO](#3-pillar-i-core-tensor-dag-engine-dynamic-autograd-memory-arena--multi-modal-io)
  - [3.1 Tape-Based Reverse Autodiff Engine (`chokkhu.core.tensor`, `chokkhu.core.ops`)](#31-tape-based-reverse-autodiff-engine-chokkhucoretensor-chokkhuops)
  - [3.2 Zero-Allocation Memory Arena Pool (`chokkhu.core.arena`)](#32-zero-allocation-memory-arena-pool-chokkhucorearena)
  - [3.3 Deterministic Dynamic Config & Context Management (`chokkhu.core.config`)](#33-deterministic-dynamic-config--context-management-chokkhuconfig)
  - [3.4 Zero-Copy SafeTensors Binary Serializer (`chokkhu.io.safetensors`)](#34-zero-copy-safetensors-binary-serializer-chokkhuiosafetensors)
  - [3.5 Multi-Modal Streaming & Domain Binary Parsers (DICOM, Geospatial, 3D Mesh)](#35-multi-modal-streaming--domain-binary-parsers-dicom-geospatial-3d-mesh)
- [4. Pillar II: End-to-End Data Lifecycle (Cleaning, Preprocessing, Transformation & Splitting)](#4-pillar-ii-end-to-end-data-lifecycle-cleaning-preprocessing-transformation--splitting)
  - [4.1 Advanced Missing Value Imputation (KNN, Iterative MICE, Nuclear-Norm Matrix Factorization)](#41-advanced-missing-value-imputation-knn-iterative-mice-nuclear-norm-matrix-factorization)
  - [4.2 Statistical & Geometric Outlier Detection (Isolation Forest, Mahalanobis MCD, LOF, Seasonal-ESD)](#42-statistical--geometric-outlier-detection-isolation-forest-mahalanobis-mcd-lof-seasonal-esd)
  - [4.3 Fuzzy Deduplication & Entity Resolution (MinHash LSH)](#43-fuzzy-deduplication--entity-resolution-minhash-lsh)
  - [4.4 Non-Linear & Variance-Stabilizing Scalers (Yeo-Johnson, Box-Cox, Quantile, Robust)](#44-non-linear--variance-stabilizing-scalers-yeo-johnson-box-cox-quantile-robust)
  - [4.5 High-Cardinality Categorical Encoders (Out-of-Fold Target, CatBoost, WoE/IV, Hash)](#45-high-cardinality-categorical-encoders-out-of-fold-target-catboost-woeiv-hash)
  - [4.6 Non-Parametric Feature Selection (Mutual Information, ANOVA, RFECV, SFS/SBS, L1)](#46-non-parametric-feature-selection-mutual-information-anova-rfecv-sfssbs-l1)
  - [4.7 Non-Linear Manifold Projections (UMAP, PacMAP, TriMAP, t-SNE, KernelPCA, FastICA)](#47-non-linear-manifold-projections-umap-pacmap-trimap-t-sne-kernelpca-fastica)
  - [4.8 Class Imbalance Resampling (SMOTE, Borderline-SMOTE, ADASYN, Tomek Links, ENN)](#48-class-imbalance-resampling-smote-borderline-smote-adasyn-tomek-links-enn)
  - [4.9 Deep Feature Synthesis (DFS, Splines, Harmonic Cyclical Embeddings)](#49-deep-feature-synthesis-dfs-splines-harmonic-cyclical-embeddings)
  - [4.10 Leakage-Free Cross-Validation & Purged Group TimeSeries with Embargo](#410-leakage-free-cross-validation--purged-group-timeseries-with-embargo)
- [5. Pillar III: Classical Machine Learning, Decision Trees, Ensembles & Ranking](#5-pillar-iii-classical-machine-learning-decision-trees-ensembles--ranking)
  - [5.1 Generalized Linear Models, Robust Estimators & Kernel SVMs (`chokkhu.models.ml`)](#51-generalized-linear-models-robust-estimators--kernel-svms-chokkhumodelsml)
  - [5.2 Tree Ensembles & Piecewise Linear Model Trees (`chokkhu.models.trees`)](#52-tree-ensembles--piecewise-linear-model-trees-chokkhumodelstrees)
  - [5.3 Density Estimation, Hierarchical & Spectral Clustering (`chokkhu.clustering`)](#53-density-estimation-hierarchical--spectral-clustering-chokkhuclustering)
  - [5.4 Learning-to-Rank Ecosystem (LambdaMART, ListNet, SoftSort) (`chokkhu.ranking`)](#54-learning-to-rank-ecosystem-lambdamart-listnet-softsort-chokkhuranking)
- [6. Pillar IV: Sovereign Deep Learning, Optimizers, Compression & Biological Alternatives](#6-pillar-iv-sovereign-deep-learning-optimizers-compression--biological-alternatives)
  - [6.1 Dynamic Computational Graph, Neural Layers, Norms & Losses (`chokkhu.models.dl`)](#61-dynamic-computational-graph-neural-layers-norms--losses-chokkhumodelsdl)
  - [6.2 Advanced Optimization, Second-Order Solvers & OptNet (`chokkhu.optimization`)](#62-advanced-optimization-second-order-solvers--optnet-chokkhuoptimization)
  - [6.3 Biologically Plausible Learning (Direct Feedback Alignment, DTP, SNN, DEQ)](#63-biologically-plausible-learning-direct-feedback-alignment-dtp-snn-deq)
  - [6.4 Sovereign Model Compression & Quantization (INT8/INT4 PTQ, AWQ, 2:4 Sparsity)](#64-sovereign-model-compression--quantization-int8int4-ptq-awq-24-sparsity)
- [7. Pillar V: Complete Computer Vision Universe, 3D Vision & Robotics](#7-pillar-v-complete-computer-vision-universe-3d-vision--robotics)
  - [7.1 2D CNN Backbones (ConvNeXt V1/V2 with GRN, ResNet, DenseNet, MobileNet V1-V4)](#71-2d-cnn-backbones-convnext-v1v2-with-grn-resnet-densenet-mobilenet-v1-v4)
  - [7.2 Vision State-Space & Transformers (Vision Mamba Vim, VMamba CSMS, ViT, Swin V1/V2, DINOv2)](#72-vision-state-space--transformers-vision-mamba-vim-vmamba-csms-vit-swin-v1v2-dinov2)
  - [7.3 Dense Segmentation & Real-Time Object Detection (UNet++, SAM/SAM-2, YOLOv1-v11, RT-DETR)](#73-dense-segmentation--real-time-object-detection-unet-samsam-2-yolov1-v11-rt-detr)
  - [7.4 Visual Explainability (GradCAM, GradCAM++, Score-CAM, Guided Backprop)](#74-visual-explainability-gradcam-gradcam-score-cam-guided-backprop)
  - [7.5 3D Computer Vision, Neural Radiance & Gaussian Splatting (PointNet++, NeRF, 3DGS, 4DGS)](#75-3d-computer-vision-neural-radiance--gaussian-splatting-pointnet-nerf-3dgs-4dgs)
  - [7.6 Embodied AI, Robotics & VLA (OpenVLA 7-DoF Tokenization, ACT, Learned Energy MPPI)](#76-embodied-ai-robotics--vla-openvla-7-dof-tokenization-act-learned-energy-mppi)
- [8. Pillar VI: NLP, Frontier Large Language Models, Reasoning & Autonomous Agents](#8-pillar-vi-nlp-frontier-large-language-models-reasoning--autonomous-agents)
  - [8.1 Pure Python Tokenization & Frontier Positional Encodings (BPE, RoPE, YaRN 128k, CoPE)](#81-pure-python-tokenization--frontier-positional-encodings-bpe-rope-yarn-128k-cope)
  - [8.2 State-Space Models & Linear Attention (Mamba S6, Mamba-2 SSD, RWKV-v7, RetNet)](#82-state-space-models--linear-attention-mamba-s6-mamba-2-ssd-rwkv-v7-retnet)
  - [8.3 Frontier LLM Architectures & MoE (DeepSeek MLA, DeepSeek-V3 256-Expert MoE, LLaMA 3.3, BitNet 1.58b)](#83-frontier-llm-architectures--moe-deepseek-mla-deepseek-v3-256-expert-moe-llama-33-bitnet-158b)
  - [8.4 Test-Time Search, Reasoning & Process Reward Models (MCTS, PRM, STaR, Quiet-STaR)](#84-test-time-search-reasoning--process-reward-models-mcts-prm-star-quiet-star)
  - [8.5 Decoding Strategies & Alignment (Speculative Decoding, Medusa, CFG JSON, RLHF, DPO, SimPO)](#85-decoding-strategies--alignment-speculative-decoding-medusa-cfg-json-rlhf-dpo-simpo)
  - [8.6 Autonomous AI Agents & Coordination (ReAct, Toolformer, Reflexion, Multi-Agent Engine)](#86-autonomous-ai-agents--coordination-react-toolformer-reflexion-multi-agent-engine)
- [9. Pillar VII: Multimodal, Audio & Generative Flow Modeling](#9-pillar-vii-multimodal-audio--generative-flow-modeling)
  - [9.1 Multimodal Vision-Language & Cross-Modal AI (SigLIP, CLIP, ImageBind, LLaVA OneVision)](#91-multimodal-vision-language--cross-modal-ai-siglip-clip-imagebind-llava-onevision)
  - [9.2 Neural Audio & Speech Processing (STFT, Mel, RVQ EnCodec/DAC, Conformer, Whisper, VITS)](#92-neural-audio--speech-processing-stft-mel-rvq-encodecdac-conformer-whisper-vits)
  - [9.3 Generative Diffusion, Flow Matching & Consistency Models (DiT, Continuous Velocity Flux.1, Reflow)](#93-generative-diffusion-flow-matching--consistency-models-dit-continuous-velocity-flux1-reflow)
  - [9.4 Score-Matching Energy Models & GFlowNets (Annealed Langevin Dynamics, Trajectory Balance)](#94-score-matching-energy-models--gflownets-annealed-langevin-dynamics-trajectory-balance)
  - [9.5 Parameter-Efficient Fine-Tuning & Control Adapters (LoRA, QLoRA, ControlNet, IP-Adapter)](#95-parameter-efficient-fine-tuning--control-adapters-lora-qlora-controlnet-ip-adapter)
- [10. Pillar VIII: Geometric Deep Learning, GNNs, Topology & Non-Euclidean Manifolds](#10-pillar-viii-geometric-deep-learning-gnns-topology--non-euclidean-manifolds)
  - [10.1 Spatial & Spectral Graph Neural Networks (GCN, GATv2, GraphSAGE, GIN, Graphormer)](#101-spatial--spectral-graph-neural-networks-gcn-gatv2-graphsage-gin-graphormer)
  - [10.2 Equivariant Graph Networks, Cellular Sheaves & Clifford Algebras (EGNN, Sheaf Laplacian, GANN)](#102-equivariant-graph-networks-cellular-sheaves--clifford-algebras-egnn-sheaf-laplacian-gann)
  - [10.3 Non-Euclidean Hyperbolic Manifolds (Poincaré Ball, Lorentz Hyperboloid, HGCN)](#103-non-euclidean-hyperbolic-manifolds-poincar-ball-lorentz-hyperboloid-hgcn)
  - [10.4 Topological Data Analysis & Persistence Vectorization (Vietoris-Rips, Persistence Landscapes)](#104-topological-data-analysis--persistence-vectorization-vietoris-rips-persistence-landscapes)
- [11. Pillar IX: SciML, Dynamical Systems, Time Series, Causal & Survival Analysis](#11-pillar-ix-sciml-dynamical-systems-time-series-causal--survival-analysis)
  - [11.1 Scientific Machine Learning & PDE Solvers (PINNs Navier-Stokes, Neural ODE RK4, FNO 2D, Sparse GP)](#111-scientific-machine-learning--pde-solvers-pinns-navier-stokes-neural-ode-rk4-fno-2d-sparse-gp)
  - [11.2 Time Series Forecasting & Non-Linear Filtering (Auto-ARIMA, UKF, Particle Filters, PatchTST)](#112-time-series-forecasting--non-linear-filtering-auto-arima-ukf-particle-filters-patchtst)
  - [11.3 Causal Inference & Continuous DAG Discovery (DoubleML, Rubin CATE, NOTEARS, LiNGAM)](#113-causal-inference--continuous-dag-discovery-doubleml-rubin-cate-notears-lingam)
  - [11.4 Survival & Reliability Modeling (Kaplan-Meier, Cox-PH, DeepSurv, Random Survival Forests)](#114-survival--reliability-modeling-kaplan-meier-cox-ph-deepsurv-random-survival-forests)
- [12. Pillar X: Specialized Frontiers (Quantum, HDC, Safety, Privacy, Merging & Valuation)](#12-pillar-x-specialized-frontiers-quantum-hdc-safety-privacy-merging--valuation)
  - [12.1 AI Safety & LLM Watermarking (Kirchenbauer Green/Red Partition, Refusal Direction Probes)](#121-ai-safety--llm-watermarking-kirchenbauer-greenred-partition-refusal-direction-probes)
  - [12.2 Privacy, Federated Learning & Machine Unlearning (DP-SGD, Byzantine Krum, SISA, SCRUB)](#122-privacy-federated-learning--machine-unlearning-dp-sgd-byzantine-krum-sisa-scrub)
  - [12.3 Model Merging, Weight Surgery & Task Arithmetic (TIES-Merging, DARE, SLERP, RegMean)](#123-model-merging-weight-surgery--task-arithmetic-ties-merging-dare-slerp-regmean)
  - [12.4 Mechanistic Interpretability & Circuit Analysis (Sparse Autoencoders SAE, DLA, Activation Patching)](#124-mechanistic-interpretability--circuit-analysis-sparse-autoencoders-sae-dla-activation-patching)
  - [12.5 Quantum ML, Hyperdimensional Computing & Genomics (VQC Parameter-Shift, HDC VSA, GenomicBERT)](#125-quantum-ml-hyperdimensional-computing--genomics-vqc-parameter-shift-hdc-vsa-genomicbert)
  - [12.6 Uncertainty Quantification, Data Valuation & Meta-Learning (Conformal CQR, Data Shapley, MMD, MAML)](#126-uncertainty-quantification-data-valuation--meta-learning-conformal-cqr-data-shapley-mmd-maml)
- [13. Pillar XI: The Grand Unified Universal Pipeline Engine (`chokkhu.pipeline`)](#13-pillar-xi-the-grand-unified-universal-pipeline-engine-chokkhupipeline)
  - [13.1 Universal 1-Line Pipeline Architecture (`ck.pipeline(...)`)](#131-universal-1-line-pipeline-architecture-ckpipeline)
  - [13.2 Fluent Composable Chaining API (`ChokkhuPipeline`)](#132-fluent-composable-chaining-api-chokkhupipeline)
  - [13.3 Multi-Task Universal Pipeline Dispatcher](#133-multi-task-universal-pipeline-dispatcher)
  - [13.4 Strict Zero-Leakage State Preservation Engine](#134-strict-zero-leakage-state-preservation-engine)
  - [13.5 Automated Deep Feature Synthesis (DFS) & Interaction Generator](#135-automated-deep-feature-synthesis-dfs--interaction-generator)
  - [13.6 SuperLearner Multi-Layer Out-of-Fold Stacking](#136-superlearner-multi-layer-out-of-fold-stacking)
  - [13.7 Standalone Dark-Mode Interactive HTML Diagnostic Dashboard Generator](#137-standalone-dark-mode-interactive-html-diagnostic-dashboard-generator)
- [14. Verification, Quality Assurance & Release Protocol](#14-verification-quality-assurance--release-protocol)
  - [14.1 Continuous Testing & Verification (430+ Test Suites)](#141-continuous-testing--verification-430-test-suites)
  - [14.2 First-Principles Numerical Invariance Protocol](#142-first-principles-numerical-invariance-protocol)
  - [14.3 Code Quality & Automated PyPI Release](#143-code-quality--automated-pypi-release)

---

## 1. Executive Vision, 9-Layer Sovereign Architecture & Zero-Dependency Principles

### 1.1 The Mission of Chokkhu & The Sovereign Manifesto
**CHOKKHU (চক্ষু)** is an uncompromising, sovereign, all-in-one artificial intelligence and machine learning framework built entirely from first mathematical principles in **pure Python and NumPy/SciPy**.

#### Core Sovereign Tenets:
1. **Zero Heavy External Dependencies**: Strictly no `torch`, `tensorflow`, `jax`, `sklearn`, `transformers`, `diffusers`, `timm`, `xgboost`, `lightgbm`, or `statsmodels`. Every tensor operator, computational graph, backward pass, optimizer, tree partition, token sampler, and PDE solver is written natively.
2. **One-Line Elegance with Complete Depth**: Any workflow—from tabular regression to 3D Gaussian Splatting, causal inference, and multi-agent reasoning—can be trained, evaluated, and deployed in a single Python call (`ck.pipeline(...)`) or customized with fluent, type-safe composable blocks (`ck.Pipeline()`).
3. **Deterministic & Strict Zero Data Leakage**: Transformations, imputations, target encodings, out-of-fold statistics, and feature selection are computed strictly on training folds and frozen for test inference.
4. **Universal Multi-Task Coverage**: Seamless support for Tabular, Vision, 3D Vision, NLP, Audio, Multimodal, Generative, Time Series, Graph, Causal, Survival, Scientific ML, Robotics, Quantum ML, and Reinforcement Learning.

### 1.2 The 9 Sovereign Architectural Layers

```
+========================================================================================+
| LAYER 9: Universal 1-Line Pipeline & Fluent Composable Engine (ck.pipeline, ck.Pipeline) |
+========================================================================================+
| LAYER 8: Multi-Modal Diagnostics & Interactive Dark-Mode HTML Report Builder           |
+========================================================================================+
| LAYER 7: Autonomous Multi-Task Auto-Dispatcher & SuperLearner Stacking                  |
+========================================================================================+
| LAYER 6: Comprehensive Domain Model Universes                                          |
|  [Classical ML | DL | Vision | 3D Vision | LLMs | Audio | GenAI | GNNs | Causal | SciML] |
+========================================================================================+
| LAYER 5: Leakage-Free Splitting, Cross-Validation & Out-of-Fold State Preservers       |
+========================================================================================+
| LAYER 4: Space Projections, Manifold Learning, Resampling & Deep Feature Synthesis     |
+========================================================================================+
| LAYER 3: Feature Preprocessing, Scaling & Advanced Encodings                           |
+========================================================================================+
| LAYER 2: Data Quality, MICE Imputation, Entity Resolution & Anomaly Cleaning           |
+========================================================================================+
| LAYER 1: Core Foundation (Dynamic Autograd DAG, Memory Buffer Arena, SafeTensors IO)   |
+========================================================================================+
```

### 1.3 Strict Operational & Engineering Directives
- **Type Safety**: 100% strict type annotations (`mypy --strict`).
- **Code Quality**: 100% compliant with PEP 8 (`flake8`, `black`).
- **Numerical Robustness**: Log-sum-exp stabilization, dynamic epsilons ($\epsilon = 10^{-12}$), and matrix conditioning checks.
- **Fast Pure NumPy Vectorization**: Multi-threading SIMD alignment, memory strides, and zero Python loop bottlenecks on inner tensor operations.

---

## 2. Complete Sovereign Codebase Directory Map (All 31 Packages & Submodules)

```
chokkhu/
├── src/
│   └── chokkhu/
│       ├── __init__.py                     # Sovereign 1-line top-level API (ck.pipeline, ck.Pipeline)
│       ├── core/                           # Foundation: Tensor Autograd DAG, Memory Arena, Config
│       │   ├── __init__.py
│       │   ├── tensor.py                   # Dynamic Autograd Tensor with Reverse Autodiff
│       │   ├── ops.py                      # Pure NumPy vector math & autodiff primitive operations
│       │   ├── arena.py                    # Zero-allocation contiguous memory pool manager
│       │   └── config.py                   # Thread-safe global seed and runtime execution tracer
│       ├── io/                             # Multi-modal ingestion, SafeTensors, Streaming
│       │   ├── __init__.py
│       │   ├── safetensors.py              # Zero-copy binary tensor serializer & deserializer
│       │   ├── streaming.py                # Lazy memory-mapped chunked data loader
│       │   ├── dicom.py                    # Medical imaging binary DICOM parser
│       │   ├── geospatial.py               # GeoJSON, Shapefile vectors, and GeoTIFF raster parser
│       │   └── mesh_3d.py                  # PLY binary/ASCII, Wavefront OBJ, PCD 3D mesh parser
│       ├── cleaning/                       # Data cleaning, MICE Imputation, Outlier detection
│       │   ├── __init__.py
│       │   ├── imputation.py               # KNNImputer, Iterative MICE, Nuclear Matrix Factorization
│       │   ├── outliers.py                 # Isolation Forest, Mahalanobis MCD, LOF, Seasonal-ESD
│       │   └── deduplication.py            # MinHash LSH fuzzy duplicate & entity resolution
│       ├── preprocessing/                  # Scaling, Encoding, Feature Selection
│       │   ├── __init__.py
│       │   ├── scalers.py                  # RobustScaler, Box-Cox, Yeo-Johnson, QuantileScaler
│       │   ├── encoders.py                 # TargetEncoder (OOF), CatBoostEncoder, WoE/IV, HashEncoder
│       │   └── selection.py                # VarianceThreshold, MutualInfo, ANOVA, RFECV, L1 Sparsity
│       ├── transformation/                 # Manifolds, Resampling, Augmentation
│       │   ├── __init__.py
│       │   ├── manifold.py                 # UMAP, PacMAP, TriMAP, t-SNE, KernelPCA, FastICA, Isomap
│       │   ├── resampling.py               # SMOTE, Borderline-SMOTE, ADASYN, Tomek Links, ENN
│       │   └── feature_engineering.py      # Deep Feature Synthesis (DFS), Spline, Harmonic Cyclical
│       ├── splitting/                      # Leak-free Cross-Validation
│       │   ├── __init__.py
│       │   ├── kfold.py                    # Stratified, Group, Multi-Label Stratified K-Fold
│       │   └── time_spatial.py             # Purged Group TimeSeries (with Embargo), Spatial Block CV
│       ├── models/                         # Unified Domain Model Universes
│       │   ├── ml/                         # Classical ML (GLM, Ridge, Lasso, SVM, KNN, Naive Bayes)
│       │   ├── trees/                      # CART, Model Trees (Linear Leaves), RF, ExtraTrees, HistGBM
│       │   ├── dl/                         # DL Layers, Norms, Losses, Activations, Biological DFA
│       │   ├── vision/                     # CNNs, ViT, Swin, Mamba-Vision, SAM, YOLOv11, RT-DETR
│       │   ├── vision_3d/                  # PointNet++, Point Transformer, NeRF, 3DGS, 4DGS, DeepSDF
│       │   ├── nlp/                        # BPE, Mamba, DeepSeek MLA/MoE, LLaMA, Reasoning MCTS, Agents
│       │   ├── multimodal/                 # CLIP, SigLIP, ImageBind, LLaVA OneVision, BLIP-2 Q-Former
│       │   ├── audio/                      # STFT, Mel, RVQ Codec (EnCodec/DAC), Conformer, Whisper, VITS
│       │   ├── generative/                 # VQ-GAN, StyleGAN, SDXL/SD3 MMDiT, DiT, Continuous Flow
│       │   ├── gnn/                        # GCN, GATv2, GraphSAGE, Graphormer, GPS, GraphCL
│       │   ├── timeseries/                 # Auto-ARIMA, Kalman, EKF/UKF, Particle Filters, PatchTST
│       │   ├── retrieval/                  # HNSW, IVF-PQ, ScaNN, ColBERT MaxSim, Okapi BM25+
│       │   ├── recommendation/             # SVD++, iALS, BPR-MF, DLRM, AutoInt, SASRec, BERT4Rec
│       │   ├── causal/                     # S/T/X/DR-Learners, DoubleML PLR, NOTEARS DAG, LiNGAM
│       │   ├── survival/                   # Kaplan-Meier, Nelson-Aalen, Cox-PH, DeepSurv, RSF
│       │   ├── sciml/                      # PINNs (Navier-Stokes), Neural ODE, FNO, Sparse GP, Koopman
│       │   ├── robotics/                   # VLA OpenVLA, Action Chunking ACT, MPPI, SPH Fluids
│       │   ├── merging/                    # TIES-Merging, DARE, SLERP, Task Vectors, RegMean
│       │   └── meta/                       # MAML, Reptile, MAP-Elites, CMA-ES
│       ├── optimization/                   # K-FAC, OptNet, INT8/INT4 PTQ, AWQ/GPTQ, 2:4 Sparsity
│       ├── uncertainty/                    # Conformalized Quantile Regression (CQR), Split Conformal
│       ├── ranking/                        # LambdaMART, ListNet, SoftSort
│       ├── safety/                         # LLM Statistical Watermarking, Refusal Direction Probes
│       ├── privacy/                        # DP-SGD, Federated Byzantine, SISA Unlearning, SCRUB
│       ├── geometry/                       # Hyperbolic Poincaré/Lorentz, Cellular Sheaf Laplacian
│       ├── hdc/                            # Hyperdimensional Vector Symbolic Architecture
│       ├── quantum/                        # Variational Quantum Classifier (VQC Parameter-Shift Rule)
│       ├── tda/                            # Persistent Homology, Persistence Landscapes & Images
│       ├── evaluation/                     # Multi-task Metrics, Calibration, Drift Detectors (MMD)
│       ├── explainability/                 # SHAP, LIME, Sparse Autoencoders (SAE), DLA, GradCAM++
│       ├── pipeline/                       # Grand Unified Pipeline Engine, Zero-Leakage State Preservers
│       └── reports/                        # Standalone Dark-Mode Interactive HTML Dashboard Builder
└── tests/                                  # 430+ Strict First-Principles Numerical & Integration Tests
```

---
## 3. Pillar I: Core Tensor DAG Engine, Dynamic Autograd, Memory Arena & Multi-Modal IO

### 3.1 Tape-Based Reverse Autodiff Engine (`chokkhu.core.tensor`, `chokkhu.core.ops`)
The autograd core operates as a dynamic, tape-based Directed Acyclic Graph (DAG) built directly on pure NumPy ndarrays.

#### 1. Tensor Node Data Structure:
```python
class Tensor:
    data: np.ndarray             # Underlying N-dimensional array (float32/float64)
    grad: np.ndarray | None      # Accumulated partial derivative (same shape as data)
    creator: Function | None     # Primitive operation that generated this tensor
    requires_grad: bool          # Flag determining whether to trace gradients
    is_leaf: bool                # True if user-created or parameter tensor
    retain_grad: bool            # If True, retains grad on non-leaf nodes
    _arena_block: MemoryBlock    # Reference to memory buffer pool
```

#### 2. Topological Sort & Reverse-Mode Gradient Propagation:
For any scalar loss $\mathcal{L}$, gradient propagation follows the chain rule over the DAG:
$$\frac{\partial \mathcal{L}}{\partial v_j} = \sum_{i \in \text{Children}(v_j)} \frac{\partial \mathcal{L}}{\partial v_i} \frac{\partial v_i}{\partial v_j}$$
The topological ordering is computed dynamically via depth-first search:
```python
def backward(self, gradient: np.ndarray | None = None) -> None:
    if self.grad is None:
        self.grad = np.ones_like(self.data) if gradient is None else gradient
    
    # Dynamic Topological Sort
    topo: list[Tensor] = []
    visited: set[Tensor] = set()
    def build_topo(v: Tensor):
        if v not in visited:
            visited.add(v)
            if v.creator is not None:
                for parent in v.creator.inputs:
                    build_topo(parent)
                topo.append(v)
    build_topo(self)
    
    # Reverse Execution
    for node in reversed(topo):
        if node.creator is not None:
            grads = node.creator.backward(node.grad)
            for parent, g in zip(node.creator.inputs, grads):
                if parent.requires_grad:
                    parent.grad = g if parent.grad is None else parent.grad + g
```

#### 3. Core Autograd Primitives & Vectorized Derivatives (`ops.py`):
- **Matrix Multiplication ($C = A B$)**:
  $$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial C} B^\top, \quad \frac{\partial \mathcal{L}}{\partial B} = A^\top \frac{\partial \mathcal{L}}{\partial C}$$
- **Convolution 2D ($Y = X \star W$)**:
  $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \star_{\text{full}} W^{\text{rot180}}, \quad \frac{\partial \mathcal{L}}{\partial W} = X \star \frac{\partial \mathcal{L}}{\partial Y}$$
  Vectorized using pure NumPy `lib.stride_tricks.as_strided` for zero-copy `im2col` matrix multiplication.
- **Log-Sum-Exp Softmax**:
  $$p_i = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}, \quad \frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)$$

### 3.2 Zero-Allocation Memory Arena Pool (`chokkhu.core.arena`)
To eliminate Python garbage collection latency during high-frequency tensor allocation, Chokkhu implements a contiguous buffer pool:
- **Bump-Pointer Memory Allocation**: Allocates large contiguous NumPy byte buffers (`np.empty(size, dtype=np.uint8)`).
- **Sub-Array Slicing**: Reallocates views onto buffer slices with shape and stride specifications without triggering `malloc()`.
- **Arena Reset & Block Recycling**: Instant bulk-reset of memory pointers at the end of each training step / forward-backward iteration.

### 3.3 Multi-Modal IO & Domain Parsers (`chokkhu.io`)
- **SafeTensors Serializer (`safetensors.py`)**:
  - Implements the SafeTensors specification in pure Python: 8-byte unsigned little-endian integer $N$ (header length) + $N$-byte JSON header describing tensor shapes, dtypes, and byte offsets + contiguous binary tensor payload.
  - Zero-copy deserialization using Python's `mmap.mmap` directly to `np.ndarray`.
- **Lazy Streaming Chunker (`streaming.py`)**:
  - Memory-efficient streaming iterator supporting CSV, JSON Lines, Parquet, and binary chunks with multi-threaded prefetching.
- **DICOM Parser (`dicom.py`)**:
  - Reads medical imaging headers, pixel array decompresion, Rescale Slope ($m$) and Rescale Intercept ($b$): $\text{HU} = m \cdot \text{pixel} + b$, and dynamic Window Width/Center clipping.
- **Geospatial Parser (`geospatial.py`)**:
  - Parses Shapefile ESRI geometry primitives (Points, Polylines, Polygons) and GeoTIFF raster georeferencing metadata (Affine transform, TiePoints, PixelScale).
- **3D Mesh & Point Cloud Parser (`mesh_3d.py`)**:
  - Binary and ASCII parsers for Stanford PLY (element vertex, element face), Wavefront OBJ ($v, vn, vt, f$), and PCD (Point Cloud Data) formats.

---

## 4. Pillar II: End-to-End Data Lifecycle (Cleaning, Preprocessing, Transformation & Splitting)

```
[Raw Multi-Modal Data] 
       │
       ▼
[Pillar II-A: Data Cleaning] ──► MICE / Matrix Factorization / Isolation Forest / MinHash
       │
       ▼
[Pillar II-B: Preprocessing]  ──► Yeo-Johnson / OOF Target / WoE / RFECV Selection
       │
       ▼
[Pillar II-C: Transformation] ──► UMAP / PacMAP / SMOTE-Tomek / Deep Feature Synthesis
       │
       ▼
[Pillar II-D: Safe Splitting] ──► Purged Group TimeSeries (Embargo) / Spatial Block CV
```

### 4.1 Advanced Missing Value Imputation (`chokkhu.cleaning.imputation`)
- **KNN Imputer**: Computes nan-Euclidean distance between samples:
  $$d_{\text{nan}}(x, y) = \sqrt{\frac{p}{p_{\text{obs}}} \sum_{i \in \text{obs}} (x_i - y_i)^2}$$
  and imputes missing entries via distance-weighted average of $k$-nearest neighbors.
- **Iterative MICE (Multivariate Imputation by Chained Equations)**:
  - Step 1: Initialize missing values with mean/median.
  - Step 2: For each column $j \in \{1, \dots, p\}$: set missing entries of $y_j$ back to NaN; fit linear/logistic regression model $y_j \sim X_{-j}$; draw imputations from posterior predictive distribution:
    $$y_j^{\text{imputed}} \sim \mathcal{N}\left(X_{-j} \hat{\beta}_j, \hat{\sigma}_j^2\right)$$
  - Step 3: Iterate until convergence (Frobenius norm difference between successive imputed matrices $\|X^{(t)} - X^{(t-1)}\|_F < 10^{-4}$).
- **Nuclear-Norm Matrix Factorization Imputer**:
  Solves the low-rank matrix completion optimization via Singular Value Thresholding (SVT):
  $$\min_M \frac{1}{2} \|\mathcal{P}_\Omega(X) - \mathcal{P}_\Omega(M)\|_F^2 + \lambda \|M\|_*, \quad M^{(k+1)} = \mathcal{D}_\tau\left(M^{(k)} + \delta \mathcal{P}_\Omega(X - M^{(k)})\right)$$
  where $\mathcal{D}_\tau(Y) = U \text{diag}(\max(0, \sigma_i - \tau)) V^\top$.

### 4.2 Statistical & Geometric Outlier Detection (`chokkhu.cleaning.outliers`)
- **Isolation Forest**:
  - Constructs $t$ Isolation Trees via recursive random feature and split value selection.
  - Path length $h(x)$ to isolate a point yields the anomaly score:
    $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}, \quad c(n) = 2\ln(n - 1) + 0.5772156649 - \frac{2(n-1)}{n}$$
    If $s(x, n) \to 1$, $x$ is highly anomalous; if $s(x, n) < 0.5$, $x$ is normal.
- **Mahalanobis Distance with MCD (Minimum Covariance Determinant)**:
  $$D_M(x) = \sqrt{(x - \hat{\mu}_{\text{MCD}})^\top \hat{\Sigma}_{\text{MCD}}^{-1} (x - \hat{\mu}_{\text{MCD}})} \sim \sqrt{\chi_p^2}$$
- **Local Outlier Factor (LOF)**:
  Computes local reachability density $\text{lrd}_k(p) = \left[ \frac{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}{|N_k(p)|} \right]^{-1}$ and LOF ratio:
  $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
- **Seasonal-ESD (Extreme Studentized Deviate)**:
  Applies STL decomposition $Y_t = T_t + S_t + R_t$, running iterative Grubbs test on residuals $R_t$ to detect seasonal anomalous spikes.

### 4.3 Fuzzy Deduplication & Entity Resolution (`chokkhu.cleaning.deduplication`)
- **MinHash Locality-Sensitive Hashing (LSH)**:
  - Generates $K$ universal hash functions $h_k(x) = (a_k x + b_k) \pmod p$.
  - Computes MinHash signatures for string $n$-grams ($n=3$), partitioning signature matrix into $b$ bands of $r$ rows.
  - Candidate duplicate pairs matching within any band bucket are verified with exact Jaccard similarity $J(A, B) = \frac{|A \cap B|}{|A \cup B|} \ge \theta$.

### 4.4 Non-Linear & Variance-Stabilizing Scalers (`chokkhu.preprocessing.scalers`)
- **Yeo-Johnson Power Transform**:
  Optimizes $\lambda$ via maximum profile log-likelihood for strictly continuous zero-mean, unit-variance distributions across all real numbers:
  $$\psi(\lambda, y) = \begin{cases} \frac{(y + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \ge 0 \\ \ln(y + 1) & \text{if } \lambda = 0, y \ge 0 \\ -\frac{(-y + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, y < 0 \\ -\ln(-y + 1) & \text{if } \lambda = 2, y < 0 \end{cases}$$
- **Quantile Transformer**:
  Computes empirical cumulative distribution function $F_n(x)$ and maps to uniform $\mathcal{U}(0, 1)$ or standard normal $\Phi^{-1}(F_n(x))$.
- **RobustScaler**:
  Centers by median and scales by Interquartile Range (IQR): $x' = \frac{x - Q_2(X)}{Q_3(X) - Q_1(X)}$.

### 4.5 High-Cardinality Categorical Encoders (`chokkhu.preprocessing.encoders`)
- **Out-of-Fold (OOF) Smoothed Target Encoder**:
  Computes smoothed posterior conditional expectation strictly inside training cross-validation folds:
  $$S_i = \lambda(n_i) \bar{y}_i + (1 - \lambda(n_i)) \bar{y}_{\text{global}}, \quad \lambda(n_i) = \frac{1}{1 + e^{-(n_i - k)/f}}$$
  where $k$ is min samples threshold, and $f$ is smoothing parameter.
- **CatBoost Encoder**:
  Online Bayesian encoder sorting rows with random permutation $\sigma$, updating running category count:
  $$\hat{x}_{\sigma(i)} = \frac{\sum_{j < i, x_{\sigma(j)} = x_{\sigma(i)}} y_{\sigma(j)} + a \cdot \bar{y}_{\text{global}}}{\sum_{j < i, x_{\sigma(j)} = x_{\sigma(i)}} 1 + a}$$
- **Weight of Evidence (WoE) & Information Value (IV)**:
  $$\text{WoE}_i = \ln\left(\frac{\% \text{Good}_i}{\% \text{Bad}_i}\right), \quad \text{IV} = \sum_{i=1}^B (\% \text{Good}_i - \% \text{Bad}_i) \cdot \text{WoE}_i$$

### 4.6 Non-Parametric Feature Selection (`chokkhu.preprocessing.selection`)
- **Mutual Information (Kraskov $k$-NN Estimator)**:
  Estimates mutual information between continuous variables $X$ and $Y$ without binning:
  $$I(X; Y) = \psi(k) - \frac{1}{N}\sum_{i=1}^N [\psi(n_x(i) + 1) + \psi(n_y(i) + 1)] + \psi(N)$$
- **Recursive Feature Elimination with Cross-Validation (RFECV)**:
  Iteratively trains model, ranks feature importance $|w_i|$ or Gini impurity, prunes worst $p\%$ features, and identifies optimal subset maximizing validation score.
- **Sequential Forward / Backward Selection (SFS / SBS)** & **L1 ElasticNet Sparsity Selection**.

### 4.7 Non-Linear Manifold Projections (`chokkhu.transformation.manifold`)
- **UMAP (Uniform Manifold Approximation and Projection)**:
  - Constructs weighted $k$-NN fuzzy simplicial graph with local connectivity $\sigma_i$:
    $$p_{i|j} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right), \quad p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} p_{j|i}$$
  - Minimizes cross-entropy in low-dimensional space $Y$:
    $$\mathcal{L}_{\text{UMAP}} = \sum_{i \neq j} \left[ p_{ij} \ln\frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \ln\frac{1 - p_{ij}}{1 - q_{ij}} \right], \quad q_{ij} = (1 + a \|y_i - y_j\|_2^{2b})^{-1}$$
- **PacMAP & TriMAP**: Triplet and paired edge embeddings preserving both local topology and global inter-cluster separations.
- **Barnes-Hut t-SNE**: Student-t low-dimensional distribution $q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$ optimized via Barnes-Hut $O(N \log N)$ quadtrees.
- **Kernel PCA & FastICA**: Fast Fixed-Point Negentropy non-Gaussianity maximization for blind source separation.

### 4.8 Class Imbalance Resampling (`chokkhu.transformation.resampling`)
- **SMOTE & Borderline-SMOTE**:
  Generates synthetic samples along line segments connecting $k$-NN minority neighbors:
  $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
  Borderline-SMOTE isolates borderline minority instances where $\frac{k}{2} \le |\text{Majority neighbors}| < k$.
- **ADASYN (Adaptive Synthetic Sampling)**:
  Weights number of synthetic samples per minority instance by its difficulty ratio $\Gamma_i = \frac{r_i}{\sum r_i}$.
- **Tomek Links & Edited Nearest Neighbors (ENN)**: Cleans ambiguous overlapping majority instances.

### 4.9 Deep Feature Synthesis (DFS) & Temporal Embeddings (`chokkhu.transformation.feature_engineering`)
- **Automated Interaction Expansion**:
  Recursively generates pairwise interaction products ($x_i \cdot x_j$), division ratios ($\frac{x_i}{x_j + \epsilon}$), logarithmic ratios ($\ln\left(\frac{x_i + \epsilon}{x_j + \epsilon}\right)$), and difference offsets.
- **Harmonic Cyclical Encodings**:
  Transforms periodic temporal features (hour, day of week, month) into continuous Cartesian harmonics:
  $$x_{\sin} = \sin\left(\frac{2\pi t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi t}{T}\right)$$
- **Spline Transformer**: Generates piecewise B-spline basis function expansions for non-linear regression.

### 4.10 Leakage-Free Cross-Validation (`chokkhu.splitting`)
- **Purged Group TimeSeries Split (with Embargo)**:
  Eliminates lookahead and autocorrelation leakage across overlapping labels:
  $$\text{Train}(k) = \{t : t \le T_k^{\text{train}}\}, \quad \text{Purge} = (T_k^{\text{train}}, T_k^{\text{test}} + h), \quad \text{Test}(k) = \{t : T_k^{\text{test}} \le t \le T_k^{\text{test}} + \Delta\}$$
- **Spatial Block K-Fold**: Hexagonal spatial binning preventing spatial autocorrelation leakage.
- **Multi-Label Stratified K-Fold**: Balances multi-label combination distributions across all $K$ folds.

---
## 5. Pillar III: Classical Machine Learning, Decision Trees, Ensembles & Ranking

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU CLASSICAL ML & ENSEMBLE ENGINE                                      |
+──────────────────────────────────────┬──────────────────────────────────────+
| [GLM / Parametric Models]            | [Tree Ensemble Universe]             |
| • OLS, Ridge, Lasso, ElasticNet      | • Decision Trees (Gini / MSE / MAE)  |
| • Robust Huber & RANSAC              | • Model Trees (Linear Leaf Models)   |
| • Logistic Regression (L-BFGS / CG)  | • Random Forest & ExtraTrees         |
| • SVC / SVR (SMO Solver, Kernels)    | • HistGradientBoosting (Histogram)   |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Clustering & Density]               | [Learning-to-Rank]                   |
| • KMeans++ & KMedoids (PAM)          | • LambdaMART (Lambda Gradients)      |
| • DBSCAN & OPTICS                    | • ListNet (Plackett-Luce Top-1)      |
| • GMM (Expectation-Maximization)     | • Differentiable NeuralSort          |
| • Spectral & Ward Hierarchical       |                                      |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 5.1 Generalized Linear Models, Robust Estimators & Kernel SVMs (`chokkhu.models.ml`)
- **Penalized Linear Models**:
  - **ElasticNet Regression**:
    $$\min_w \frac{1}{2n} \|X w - y\|_2^2 + \alpha \rho \|w\|_1 + \frac{\alpha(1 - \rho)}{2} \|w\|_2^2$$
    Solved via Coordinate Descent with Soft-Thresholding:
    $$w_j \leftarrow \frac{\mathcal{S}\left(\sum_{i=1}^n x_{ij}(y_i - \sum_{k \neq j} x_{ik} w_k), \alpha \rho\right)}{\sum_{i=1}^n x_{ij}^2 + \alpha(1 - \rho)}, \quad \mathcal{S}(z, \gamma) = \text{sign}(z) \max(0, |z| - \gamma)$$
- **Robust Huber & RANSAC Regression**:
  - Huber $M$-estimation with derivative $\psi(r) = \text{clip}(r, -\delta, \delta)$ optimizing resistance to extreme outliers.
  - RANSAC (Random Sample Consensus) iteratively fitting minimum inlier subsets across $N$ iterations.
- **Support Vector Machines (SVC / SVR)**:
  - Solves the dual quadratic programming problem via Sequential Minimal Optimization (SMO):
    $$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \; \sum_{i=1}^n \alpha_i y_i = 0$$
  - Supporting RBF Kernel ($K(x, z) = e^{-\gamma \|x - z\|^2}$), Polynomial Kernel ($(\gamma \langle x, z \rangle + r)^d$), Sigmoid Kernel, and Mahalanobis Kernel.
- **Probabilistic Naive Bayes**: Gaussian, Multinomial, Bernoulli, and Complement Naive Bayes with Dirichlet smoothing.

### 5.2 Tree Ensembles & Piecewise Linear Model Trees (`chokkhu.models.trees`)
- **Fast Histogram Gradient Boosting Classifier & Regressor**:
  - Bins continuous features into 256 integer bins, constructing histogram gradient tables in $O(N)$ time.
  - Evaluates split gain in $O(K)$ time per feature:
    $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
    where $G_L = \sum_{i \in L} g_i$, $H_L = \sum_{i \in L} h_i$, and $\lambda, \gamma$ are L2 regularization and tree complexity penalties.
- **Piecewise Linear Model Trees (Model Trees)**:
  - Instead of predicting a scalar mean value in each leaf node, each leaf fits a local Ridge/OLS regression model:
    $$\hat{y}(x) = w_{\text{leaf}(x)}^\top x + b_{\text{leaf}(x)}$$
  - Produces continuous piecewise linear approximation manifolds with high regression accuracy and full interpretability.
- **Random Forest & ExtraTrees**:
  - Bootstrap aggregating (bagging) with random feature subspace selection ($m = \sqrt{p}$ for classification, $m = p/3$ for regression).
  - Extremely Randomized Trees (ExtraTrees) draws random split thresholds for all candidate features, reducing variance.

### 5.3 Density Estimation, Hierarchical & Spectral Clustering (`chokkhu.clustering`)
- **DBSCAN & OPTICS**:
  - DBSCAN: Identifies Core points ($|N_\epsilon(p)| \ge \text{MinPts}$), Border points, and Noise points.
  - OPTICS: Computes Core Distance and Reachability Distance, generating reachability plot profiles for multi-density clusters.
- **Gaussian Mixture Models (GMM)**:
  - Expectation Step: Posterior responsibility $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$.
  - Maximization Step: Updates parameters $\mu_k, \Sigma_k, \pi_k$ until log-likelihood convergence.
- **Spectral Clustering**: Graph Laplacian $L = D - A$, normalized symmetric Laplacian $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$, and top-$k$ eigenvector projection.

### 5.4 Learning-to-Rank Ecosystem (`chokkhu.ranking`)
- **LambdaMART**:
  Tree boosting optimizing rank order via NDCG-scaled gradient forces:
  $$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta \text{NDCG}_{ij}|, \quad w_{ij} = \sigma \lambda_{ij} (1 - \rho_{ij}) |\Delta \text{NDCG}_{ij}|$$
- **ListNet**: Plackett-Luce probability distribution matching over permutation lists:
  $$P(y) = \prod_{i=1}^n \frac{\exp(s_i)}{\sum_{j=i}^n \exp(s_j)}, \quad \mathcal{L}_{\text{ListNet}} = -\sum_{j=1}^n P(y_j) \ln P(\hat{y}_j)$$
- **Differentiable NeuralSort & SoftSort**: Continuous relaxation of permutation matrices via optimal transport.

---

## 6. Pillar IV: Sovereign Deep Learning, Optimizers, Compression & Biological Alternatives

### 6.1 Dynamic Computational Graph, Neural Layers, Norms & Losses (`chokkhu.models.dl`)
- **Layer Primitives**: Linear, Conv1D/2D/3D, ConvTranspose2D, DepthwiseSeparableConv2D, Embedding.
- **Normalization Modules**:
  - **RMSNorm**: $y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$ (zero-mean assumption for maximum efficiency).
  - **LayerNorm, BatchNorm1D/2D, GroupNorm, InstanceNorm**: Full pure NumPy forward and backward passes.
- **Advanced Activation Functions**:
  - **SwiGLU**: $\text{SwiGLU}(x, W, V, b, c) = (\text{Swish}(x W + b)) \odot (x V + c)$ where $\text{Swish}(z) = z \cdot \sigma(\beta z)$.
  - **GELU, SiLU, Mish, LeakyReLU, PReLU, ELU, SELU**: Numerically stabilized vectorized implementations.
- **Loss Formulations**:
  - **Focal Loss**: $\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$ (counteracting extreme class imbalance).
  - **Dice Loss, Triplet Margin Loss, CTC (Connectionist Temporal Classification) Loss**.

### 6.2 Advanced Optimization, Second-Order Solvers & OptNet (`chokkhu.optimization`)
- **Lion (EvoLved Sign Momentum)**:
  $$c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad \theta_t = \theta_{t-1} - \eta (\text{sign}(c_t) + \lambda \theta_{t-1}), \quad m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t$$
- **AdamW & Prodigy (Adaptive Step-Size Learning-Rate-Free Optimizer)**:
  Automates learning rate tuning via real-time $D$-adaptation estimates of distance to optimum:
  $$D_{t+1} = \max\left(D_t, \frac{\|\theta_0 - \theta_t\|_2}{\sqrt{t}}\right), \quad \eta_t = d_0 \cdot D_t \cdot \frac{\alpha_t}{\sqrt{v_t + \epsilon}}$$
- **K-FAC (Kronecker-factored Approximate Curvature)**:
  Second-order natural gradient descent approximating Fisher Information Matrix as Kronecker products of layer inputs and output gradients:
  $$F_l \approx \mathbb{E}[a_{l-1} a_{l-1}^\top] \otimes \mathbb{E}[g_l g_l^\top] = A_{l-1} \otimes G_l, \quad F_l^{-1} \text{vec}(\nabla_{W_l} \mathcal{L}) = \text{vec}(G_l^{-1} \nabla_{W_l} \mathcal{L} A_{l-1}^{-1})$$
- **OptNet (Differentiable Quadratic Programming Layers)**:
  Solves $\min_z \frac{1}{2} z^\top Q z + q^\top z \; \text{s.t.} \; G z \le h, A z = b$ with implicit differentiation via KKT conditions.

### 6.3 Biologically Plausible Learning Alternatives (`chokkhu.models.dl.biological`)
- **Direct Feedback Alignment (DFA)**:
  Eliminates the weight transport problem ($W^\top$) by projecting top-layer output error $e = \hat{y} - y$ directly to intermediate hidden layers using fixed random feedback matrices $B_l$:
  $$\delta h_l = (e B_l^\top) \odot \sigma'(a_l), \quad \Delta W_l = -\eta \delta h_l h_{l-1}^\top$$
- **Difference Target Propagation (DTP)**:
  Trains layer-wise inverse reconstruction networks $g_l$ to propagate target representations backwards without backpropagation.
- **Decoupled Neural Interfaces (Synthetic Gradients)**:
  Auxiliary neural modules predicting future gradients $\hat{\nabla}_{h_l} \mathcal{L}$ enabling asynchronous forward-backward passes.
- **Spiking Neural Networks (SNN) & Neuromorphic STDP**:
  Leaky Integrate-and-Fire (LIF) neurons with Spike-Timing-Dependent Plasticity:
  $$\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{\text{rest}}) + R I(t), \quad \Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \Delta t > 0 \\ -A_- e^{\Delta t / \tau_-} & \Delta t < 0 \end{cases}$$
- **Deep Equilibrium Models (DEQ)**:
  Root-finding forward passes via Broyden's method solving $z^* = f_\theta(z^*, x)$ with implicit function theorem backward passes.

### 6.4 Sovereign Model Compression & Quantization (`chokkhu.optimization.compression`)
- **Post-Training Quantization (INT8 & INT4)**:
  Symmetric and asymmetric affine mappings: $X_{\text{quant}} = \text{clip}\left(\text{round}\left(\frac{X}{S}\right) + Z, -2^{b-1}, 2^{b-1}-1\right)$.
- **AWQ (Activation-Aware Weight Quantization) & GPTQ**:
  Protects top $1\%$ salient weights based on activation magnitude, quantizing remaining weights via second-order error compensation:
  $$w_q^* = \arg\min_{w_q} (w - w_q)^\top H (w - w_q)$$
- **Structured 2:4 Sparsity & Distillation**:
  Enforces 2 zero values out of every 4 contiguous elements, coupled with Hinton soft-target temperature distillation loss:
  $$\mathcal{L}_{\text{distill}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha T^2 D_{\text{KL}}(\sigma(z_s / T) \parallel \sigma(z_t / T))$$

---
## 7. Pillar V: Complete Computer Vision Universe, 3D Vision & Robotics

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU COMPUTER VISION, 3D & ROBOTICS UNIVERSE                             |
+──────────────────────────────────────┬──────────────────────────────────────+
| [2D Vision Backbones]                | [Vision Transformers & State-Space]  |
| • ConvNeXt (V1/V2 with GRN)          | • ViT (Patch Embeddings + Pos)       |
| • ResNet (18-152, Wide, Res2Net)     | • Swin Transformer (V1/V2 Log-CPB)   |
| • MobileNet (V1-V4), EfficientNet V2 | • Vision Mamba (Vim 2D Bidirectional)|
| • DenseNet, SqueezeNet, ShuffleNet   | • VMamba (Cross-Scan CSMS)           |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Segmentation & Detection]           | [3D Vision & Neural Radiance]        |
| • UNet, UNet++, TransUNet            | • PointNet++, DGCNN (EdgeConv)       |
| • DeepLabV3+ (ASPP), SAM / SAM-2     | • Instant-NGP (Hash Grid NeRF)       |
| • YOLOv1-v11 (Dual Assignment GELAN) | • 3D Gaussian Splatting (3DGS)       |
| • RT-DETR / Deformable DETR          | • 4DGS (Dynamic Deformations)        |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 7.1 2D CNN Backbones (`chokkhu.models.vision.architectures`)
- **ConvNeXt V1/V2 with Global Response Normalization (GRN)**:
  - Macro-design: 7x7 Depthwise Conv $\to$ LayerNorm $\to$ 1x1 Conv (expand 4x) $\to$ GELU $\to$ GRN $\to$ 1x1 Conv (project down) + Residual.
  - GRN unit optimizes channel competition:
    $$gx_i = \|x_i\|_2, \quad nx_i = \frac{gx_i}{\frac{1}{C}\sum_j gx_j + \epsilon}, \quad x_i' = \gamma (x_i \odot nx_i) + \beta + x_i$$
- **ResNet, Wide-ResNet & Res2Net**:
  - Res2Net splits feature map into $s=4$ scale subsets, constructing hierarchical multi-scale receptive field connections.
- **MobileNet V4 & EfficientNet V2**: NAS-optimized Fused-MBConv blocks with Squeeze-and-Excitation channel gating $\sigma(W_2 \text{ReLU}(W_1 x))$.

### 7.2 Vision State-Space & Transformers (`chokkhu.models.vision`)
- **Vision Mamba (Vim)**:
  - Flattens 2D image patches $X \in \mathbb{R}^{H \times W \times C}$ into 1D sequences.
  - Applies 2D bidirectional selective state-space scanning: forward scan $h_t^f$ and backward scan $h_t^b$, merging outputs via gated projection.
- **VMamba (CSMS Cross-Scan Module)**:
  - Traverses 2D feature grids along 4 spatial trajectories: (1) Top-Left $\to$ Bottom-Right, (2) Bottom-Right $\to$ Top-Left, (3) Top-Right $\to$ Bottom-Left, (4) Bottom-Left $\to$ Top-Right.
- **Swin Transformer (V1/V2)**:
  - Regular and Shifted window multi-head self-attention with continuous Log-Spaced Coordinate Position Bias (Log-CPB):
    $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + B\right) V, \quad B_{ij} = \text{MLP}(\text{sign}(\Delta x)\ln(1 + |\Delta x|), \text{sign}(\Delta y)\ln(1 + |\Delta y|))$$
- **DINOv2 & MAE**: Multi-crop teacher-student self-distillation with Sinkhorn-Knopp center assignment.

### 7.3 Dense Segmentation & Real-Time Object Detection (`chokkhu.models.vision`)
- **Advanced Dense Segmentation**:
  - **DeepLabV3+**: Atrous Spatial Pyramid Pooling (ASPP) with dilation rates $r \in \{1, 6, 12, 18\}$ and low-level feature fusion.
  - **SAM & SAM-2 (Segment Anything)**: Promptable segmentation engine combining image encoder, prompt encoder (points, boxes, masks), and two-way lightweight mask decoder.
- **Real-Time Object Detection**:
  - **YOLOv11 Architecture**: GELAN (Generalized Efficient Layer Aggregation Network), dual-assignment label assigner (one-to-many for training, one-to-one for NMS-free inference), and SPPF (Spatial Pyramid Pooling Fast).
  - **RT-DETR (Real-Time DEtection TRansformer)**: Hybrid encoder with intra-scale interaction and cross-scale fusion, eliminating anchor tuning.
- **Visual Explainability (XAI)**:
  - **GradCAM++**: Second and third-order derivative weighted feature activation heatmaps:
    $$\alpha_k^c = \sum_{i,j} w_{ij}^{kc} \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{ij}^k}\right), \quad L_{\text{Grad-CAM++}}^c = \sum_k \alpha_k^c A^k$$

### 7.4 3D Computer Vision, Neural Radiance & Gaussian Splatting (`chokkhu.models.vision_3d`)
- **Point Cloud Deep Learning**:
  - **PointNet++**: Hierarchical point set feature learning with Farthest Point Sampling (FPS) and ball query radius grouping.
  - **DGCNN (Dynamic Graph CNN)**: Dynamic $k$-NN edge convolution capturing local geometric structures:
    $$e_{ijm} = \text{ReLU}(\theta_m \cdot (x_j - x_i) + \phi_m \cdot x_i)$$
  - **Point Transformer (v1-v3)**: Vector self-attention over continuous 3D Euclidean coordinates.
- **Neural Radiance Fields & 3D/4D Gaussian Splatting**:
  - **Instant-NGP (Instant Neural Graphics Primitives)**: Multi-resolution hash grid spatial encoding mapping $(x, y, z) \to \mathbb{R}^{16}$ with compact shallow MLP volume rendering.
  - **3D Gaussian Splatting (3DGS)**: Continuous differentiable 3D Gaussians characterized by position $\mu$, covariance $\Sigma = R S S^\top R^\top$, opacity $\alpha$, and spherical harmonics (SH) color coefficients:
    $$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$
    Projected onto 2D camera coordinates via Jacobian $J$: $\Sigma' = J W \Sigma W^\top J^\top$.
  - **4D Gaussian Splatting (4DGS)**: Dynamic deformation networks $d(x, t) = (\Delta \mu, \Delta r, \Delta s)$ modeling time-varying spatial deformations.
  - **DeepSDF & Marching Cubes**: Continuous Signed Distance Function $f_\theta(x) = s$ with zero-isosurface polygonal mesh extraction.

### 7.5 Embodied AI, Robotics & VLA (`chokkhu.models.robotics`)
- **Vision-Language-Action (VLA) Architecture**:
  Continuous 7-DoF robot end-effector displacements $(x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper})$ discretized into 256 vocabulary bins, allowing unified autoregressive policy generation.
- **Action Chunking Transformer (ACT) with CVAE**:
  Predicts multi-step temporal action chunks $a_{t:t+k}$ conditioned on visual tokens and joint angles, regularized via latent variable $z \sim \mathcal{N}(\mu, \Sigma)$.
- **Learned Energy MPPI (Model Predictive Path Integral)**:
  Differentiable trajectory optimization sampling control perturbations $\epsilon_k \sim \mathcal{N}(0, \Sigma)$ weighted by exponential cost surfaces.

---

## 8. Pillar VI: NLP, Frontier Large Language Models, Reasoning & Autonomous Agents

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU FRONTIER NLP & LLM ARCHITECTURE                                     |
+──────────────────────────────────────┬──────────────────────────────────────+
| [Tokenization & Embeddings]          | [State-Space & Linear Attention]     |
| • Byte-Level BPE (GPT-4 Regex)       | • Mamba (S6 Selective Scan)          |
| • SentencePiece / Unigram            | • Mamba-2 (State Space Duality SSD)  |
| • RoPE, YaRN (128k), ALiBi, CoPE     | • RWKV (v4-v7), RetNet, Jamba        |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Modern LLM Engines]                 | [Reasoning, Alignment & Agents]      |
| • LLaMA 3.3 (SwiGLU, RMSNorm, GQA)   | • MCTS Tree Search & PRM Verifiers   |
| • DeepSeek-V3 (256-Expert MoE + MLA) | • RLHF PPO, DPO, KTO, SimPO, SPIN    |
| • DeepSeek-R1 (Reasoning Distill)    | • Speculative Decoding & Medusa      |
| • BitNet 1.58b (Ternary Weights)     | • ReAct, Toolformer, Reflexion       |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 8.1 Pure Python Tokenization & Frontier Positional Encodings (`chokkhu.models.nlp.tokenizers`)
- **Byte-Level Byte Pair Encoding (BPE)**:
  - Exact implementation of GPT-4 regex pattern tokenizer:
    `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?[\p{L}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+'`
  - Min-heap priority merge queue for $O(N \log V)$ tokenization speed.
- **Frontier Positional Embeddings**:
  - **Rotary Position Embedding (RoPE)**: $R_{\Theta, m}^d x = (x_1 \cos m\theta_1 - x_2 \sin m\theta_1, \dots)$.
  - **YaRN (Yet another RoPE extensioN)**: Extends context windows up to 128k tokens via temperature-scaled NTK-aware interpolation:
    $$\theta'_i = \theta_i^{(1 - \alpha)} \cdot (s \cdot \theta_i)^\alpha, \quad \text{where } \alpha = \text{ramp}\left(\frac{2\pi}{\lambda_i}\right)$$
  - **ALiBi & CoPE (Contextual Position Encoding)**: Distance penalty attention biases and content-dependent positional queries.

### 8.2 State-Space Models & Linear Attention (`chokkhu.models.nlp.architectures`)
- **Mamba (S6 Selective Scan Algorithm)**:
  Hardware-aware time-varying discretization:
  $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t, \quad \bar{A}_t = \exp(\Delta_t A), \; \bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I) \Delta_t B$$
  where $\Delta_t = \text{softplus}(\text{Linear}(x_t))$, $B_t = \text{Linear}(x_t)$, and $C_t = \text{Linear}(x_t)$.
- **Mamba-2 (SSD: State Space Duality)**:
  Proves equivalence between structured SSMs and masked 1-semiseparable matrix multiplication, enabling fast block-diagonal matrix operations.
- **RWKV-v7 (Goose) & RetNet (Retention Network)**: Multi-scale retention with parallel, recurrent, and chunkwise representations.

### 8.3 Frontier LLM Architectures & Mixture-of-Experts (`chokkhu.models.nlp.architectures`)
- **DeepSeek Multi-Head Latent Attention (MLA)**:
  Low-rank KV compression drastically reducing KV cache size during autoregressive inference:
  $$c_t^{KV} = W_{DKV} h_t, \quad [k_t^C; k_t^R] = [W_{UK} c_t^{KV}; \text{RoPE}(W_{KR} h_t)], \quad v_t^C = W_{UV} c_t^{KV}$$
- **DeepSeek-V3 Fine-Grained MoE**:
  256 routed experts + 1 shared expert with auxiliary-loss-free load balancing using dynamic bias adjustments:
  $$y = u + \sum_{i=1}^{K} g_i \text{Expert}_i(u), \quad g_i = \text{Softmax}\left(\text{TopK}(u^\top e_j + b_j, K)\right)$$
- **BitNet 1.58b (Ternary Weights $\{-1, 0, 1\}$)**:
  Replaces matrix multiplication with addition:
  $$\tilde{W} = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right), \quad \gamma = \frac{1}{n m}\sum_{i,j} |W_{ij}|$$
- **LLaMA 3.3, Mistral/Mixtral, Gemma 2, Qwen 2.5 & Flan-T5**.

### 8.4 Test-Time Search, Reasoning & Process Reward Models (`chokkhu.models.nlp.agents`)
- **MCTS & Process Reward Models (PRMs)**:
  Monte Carlo Tree Search evaluating intermediate reasoning steps via step-level reward verifiers.
- **Alignment Algorithms**:
  - **Direct Preference Optimization (DPO)**:
    $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[ \ln \sigma\left( \beta \ln \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \ln \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
  - **KTO (Kahneman-Tversky Optimization), SimPO (Simple Preference Optimization), and SPIN**.
- **Autonomous Multi-Agent Coordination Engine**:
  - **ReAct (Reasoning + Acting)**: Structured Thought $\to$ Action $\to$ Observation execution loops.
  - **Toolformer**: Self-supervised API call generation and response integration.
  - **Reflexion**: Memory-guided verbal reinforcement learning for iterative code and reasoning correction.

---
## 9. Pillar VII: Multimodal, Audio & Generative Flow Modeling

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU MULTIMODAL, AUDIO & GENERATIVE SUITE                                |
+──────────────────────────────────────┬──────────────────────────────────────+
| [Multimodal & Cross-Modal AI]        | [Audio & Speech Processing]          |
| • CLIP & SigLIP (Sigmoid Loss)       | • STFT, Mel-Spectrogram, MFCC        |
| • ImageBind (6-Modality Unified)     | • RVQ Codecs (EnCodec, DAC)          |
| • LLaVA OneVision & BLIP-2 Q-Former  | • Conformer, Whisper, VITS, HiFi-GAN |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Generative & Diffusion Models]      | [Flow Matching & Energy Models]      |
| • VQ-GAN, StyleGAN 1-3               | • Continuous Velocity Flow Matching  |
| • Latent Diffusion (SDXL / SD3)      | • 2-Rectified Flow (Reflow ODE)      |
| • DiT (adaLN-Zero Transformer)       | • Score-Matching EBM & GFlowNets     |
| • LoRA, QLoRA, ControlNet            | • Single-Step Consistency Models     |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 9.1 Multimodal Vision-Language Models (`chokkhu.models.multimodal`)
- **SigLIP (Sigmoid Loss for Language-Image Pre-Training)**:
  Replaces cross-entropy with pairwise binary classification, enabling linear batch-size scaling:
  $$\mathcal{L}_{\text{SigLIP}} = -\sum_{i,j} \ln\left( \frac{1}{1 + e^{-y_{ij} (t \langle u_i, v_j \rangle + b)}} \right), \quad y_{ij} = \begin{cases} 1 & i = j \\ -1 & i \neq j \end{cases}$$
- **ImageBind**: Unified joint embedding space across 6 modalities (Image/Video, Text, Audio, Depth, Thermal, IMU).
- **LLaVA OneVision & BLIP-2 Q-Former**: Cross-attention perceiver resamplers bridging spatial vision tokens into autoregressive language streams.

### 9.2 Neural Audio & Speech Processing (`chokkhu.models.audio`)
- **First-Principles Spectral Analysis**: STFT, Mel-filterbank bank integration, MFCC, and SpecAugment.
- **Residual Vector Quantization (RVQ) Codecs (EnCodec / DAC)**:
  Cascaded codebook quantization capturing multi-rate audio signals:
  $$x_0 = x, \quad z_k = \text{Quantize}(x_{k-1}, \mathcal{C}_k), \quad x_k = x_{k-1} - z_k$$
- **Conformer, Whisper, VITS (Conditional VAE + GAN TTS) & HiFi-GAN Vocoder**.

### 9.3 Generative Diffusion, Flow Matching & Consistency Models (`chokkhu.models.generative`)
- **Diffusion Transformers (DiT)**:
  Replaces standard U-Net with ViT backbones conditioned via adaptive Layer Normalization (adaLN-Zero):
  $$\text{adaLN}(h, y) = \gamma(y) \odot \text{LayerNorm}(h) + \beta(y), \quad \text{Gate}(h, y) = \alpha(y) \odot h$$
- **Continuous Velocity Flow Matching (Flux.1 / SD3)**:
  Directly learns the probability flow ODE velocity field:
  $$x_t = (1 - t) x_0 + t x_1, \quad u_t(x_t) = x_1 - x_0, \quad \mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|_2^2]$$
- **Consistency Models & Reflow Straightening**: Single-step generative generation along probability flow trajectories.
- **Score-Matching Energy-Based Models & GFlowNets**: Annealed Langevin Dynamics and Trajectory Balance generative flows.

---

## 10. Pillar VIII: Geometric Deep Learning, GNNs, Topology & Non-Euclidean Manifolds

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU GEOMETRIC, GNN & TOPOLOGICAL ECOSYSTEM                              |
+──────────────────────────────────────┬──────────────────────────────────────+
| [Spatial & Spectral GNNs]            | [Equivariant & Sheaf Networks]       |
| • GCN, GATv2, GraphSAGE, GIN, PNA    | • E(n) / SE(3) EGNN (Equivariant)    |
| • Graphormer, GPS Transformer        | • Cellular Sheaf Neural Networks     |
| • GraphCL & DGI (Self-Supervised)    | • Clifford Geometric Algebra (GANN)  |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Non-Euclidean Manifolds]            | [Topological Data Analysis (TDA)]    |
| • Poincaré Ball Model (Hyperbolic)   | • Vietoris-Rips Simplicial Complexes |
| • Lorentz Hyperboloid Manifold       | • Persistent Homology Diagrams       |
| • Hyperbolic Graph CNN (HGCN)        | • Persistence Landscapes & Images    |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 10.1 Graph Neural Networks & Graph Transformers (`chokkhu.models.gnn`)
- **Spatial & Spectral Message Passing**:
  - **GATv2 (Dynamic Attention GNN)**: Dynamic attention resolving static ranking issues in standard GAT:
    $$\alpha_{ij} = \frac{\exp\left(a^\top \text{LeakyReLU}(W [h_i \parallel h_j])\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(a^\top \text{LeakyReLU}(W [h_i \parallel h_k])\right)}$$
  - **GIN (Graph Isomorphism Network)**: Maximally expressive 1-Weisfeiler-Lehman graph classification:
    $$h_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$
  - **Graphormer & GPS (General, Powerful, Scalable) Graph Transformers**: Combines local message passing with global multi-head self-attention and shortest-path spatial encodings.

### 10.2 Equivariant Networks, Sheaves & Geometric Algebras (`chokkhu.models.geometric`, `chokkhu.geometry`)
- **$E(n)$ / $SE(3)$ Equivariant Graph Neural Networks (EGNN)**:
  Coordinates $x_i \in \mathbb{R}^3$ and features $h_i \in \mathbb{R}^d$ update equivariantly under rigid 3D Euclidean motions:
  $$m_{ij} = \phi_e(h_i, h_j, \|x_i - x_j\|^2, a_{ij}), \quad x_i^{(l+1)} = x_i^{(l)} + \sum_{j \in \mathcal{N}(i)} (x_i - x_j) \phi_x(m_{ij}), \quad h_i^{(l+1)} = \phi_h\left(h_i^{(l)}, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$$
- **Cellular Sheaf Neural Networks**:
  Associates stalk vector spaces $\mathcal{F}(v)$ with restriction maps $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ and computes Sheaf Laplacian diffusion:
  $$\Delta_{\mathcal{F}} = D_{\mathcal{F}} - A_{\mathcal{F}}, \quad h^{(l+1)} = \sigma\left((I - \Delta_{\mathcal{F}}) h^{(l)} W\right)$$
  Directly eliminates graph over-smoothing and models heterophilic graphs.
- **Clifford Geometric Algebra Neural Networks (GANN)**: Multivector representations with grade-preserving geometric product $u v = u \cdot v + u \wedge v$.

### 10.3 Non-Euclidean Hyperbolic Geometry & TDA (`chokkhu.geometry`, `chokkhu.tda`)
- **Poincaré Ball & Lorentz Hyperboloid Manifolds**:
  Geodesic distance in Poincaré ball $\mathbb{D}_c^n$:
  $$d_{\mathbb{D}}(u, v) = \frac{2}{\sqrt{c}} \text{artanh}\left( \sqrt{c} \frac{\|u - v\|_2}{\sqrt{1 - 2c \langle u, v \rangle + c^2 \|u\|_2^2 \|v\|_2^2}} \right)$$
- **Topological Data Analysis (TDA) Vectorization**:
  Extracts persistent homology barcode diagrams $(b_i, d_i)$ from Vietoris-Rips complexes, converting them to vector embeddings via Persistence Landscapes $\lambda_k(t)$ and Persistence Surface Images.

---
## 11. Pillar IX: SciML, Dynamical Systems, Time Series, Causal & Survival Analysis

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU SCIENTIFIC ML, DYNAMICAL, CAUSAL & TIME SYSTEMS                     |
+──────────────────────────────────────┬──────────────────────────────────────+
| [Scientific ML & Physics-Informed]   | [Time Series & State Filtering]      |
| • PINNs (Navier-Stokes, Burgers)     | • Auto-ARIMA, SARIMAX, GARCH         |
| • Neural ODE (Runge-Kutta RK4)       | • Kalman, Extended (EKF), UKF        |
| • Fourier Neural Operators (FNO)     | • Particle Filters (SIR, RBPF)       |
| • Sparse GP & Koopman / DMD          | • PatchTST, TimesNet, TiDE, Chronos  |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Causal Inference & Discovery]       | [Survival Analysis & Reliability]    |
| • Rubin CATE (S/T/X/DR-Learner)      | • Kaplan-Meier & Nelson-Aalen        |
| • DoubleML PLR (Neyman Orthogonal)   | • Cox Proportional Hazards           |
| • NOTEARS DAG Discovery & LiNGAM     | • DeepSurv & Random Survival Forests |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 11.1 Scientific Machine Learning & Dynamical Systems (`chokkhu.models.sciml`)
- **Physics-Informed Neural Networks (PINNs)**:
  Embedded PDE residuals evaluated via Autograd derivatives in the loss function:
  $$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \|\partial_t u + u \partial_x u - \nu \partial_{xx} u\|_2^2 + \lambda_{\text{BC}} \|u - g_{\text{BC}}\|_2^2$$
- **Neural Ordinary Differential Equations (Neural ODE)**:
  Continuous depth residual networks solved via 4th-order Runge-Kutta (RK4) with Adjoint state backward gradients:
  $$\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f(z(t), t, \theta)}{\partial z}$$
- **Fourier Neural Operators (FNO 2D)**:
  Kernel integral operator in frequency space:
  $$\mathcal{K}(v)(x) = \mathcal{F}^{-1}\left( R \cdot \mathcal{F}(v) \right)(x), \quad v_{t+1}(x) = \sigma\left( W v_t(x) + \mathcal{K}(v_t)(x) \right)$$
- **Sparse GP (Titsias Variational Free Energy VFE)** & **Exact Koopman Dynamic Mode Decomposition (DMD)**.

### 11.2 Time Series Forecasting & State Filtering (`chokkhu.models.timeseries`)
- **Non-Linear State Filtering**:
  - **Unscented Kalman Filter (UKF)**: Deterministic Sigma-Point propagation through non-linear dynamics:
    $$\chi_0 = \mu, \quad \chi_i = \mu + (\sqrt{(L + \lambda)\Sigma})_i, \quad \chi_{i+L} = \mu - (\sqrt{(L + \lambda)\Sigma})_i$$
  - **Sequential Importance Resampling (SIR) & Rao-Blackwellized Particle Filter (RBPF)**:
    Recursive particle updates with Systematic, Stratified, and Residual resampling.
- **Deep Patch Forecasting**:
  - **PatchTST**: Unfolds continuous time-series into sub-series patches processed via channel-independent self-attention.
  - **TimesNet**: Transforms 1D time series into 2D temporal-variation tensors via Fast Fourier Transform.

### 11.3 Causal Inference & Causal Discovery (`chokkhu.models.causal`)
- **Double Machine Learning (DoubleML PLR)**:
  Neyman-orthogonal score function eliminating regularization bias in treatment effect estimation:
  $$\psi(W; \theta, \eta) = (Y - \ell(X)) - \theta (D - m(X)), \quad \mathbb{E}\left[\frac{\partial \psi}{\partial \eta}\right] = 0$$
- **Meta-Learners (S/T/X/DR-Learners)**: Doubly-robust heterogeneous treatment effect (CATE) estimation.
- **Continuous Optimization DAG Discovery (NOTEARS)**:
  Formulates acyclicity constraint as a smooth algebraic equality:
  $$\min_W \frac{1}{2n}\|X - X W\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad h(W) = \text{Tr}(\exp(W \odot W)) - d = 0$$

### 11.4 Survival & Reliability Modeling (`chokkhu.models.survival`)
- **Cox Proportional Hazards**: Partial likelihood maximization for censored failure times:
  $$L(\beta) = \prod_{i: \delta_i = 1} \frac{\exp(x_i^\top \beta)}{\sum_{j \in \mathcal{R}(t_i)} \exp(x_j^\top \beta)}$$
- **DeepSurv**: Non-linear deep neural network optimizing negative Cox log-partial likelihood.
- **Random Survival Forests (RSF)**: Log-rank test split maximization with Nelson-Aalen cumulative hazard estimations.

---

## 12. Pillar X: Specialized Frontiers (Quantum, HDC, Safety, Privacy, Merging & Valuation)

```
+─────────────────────────────────────────────────────────────────────────────+
| CHOKKHU SPECIALIZED FRONTIERS SUITE                                         |
+──────────────────────────────────────┬──────────────────────────────────────+
| [AI Safety & Watermarking]           | [Privacy & Machine Unlearning]       |
| • Kirchenbauer Green/Red Partition   | • DP-SGD (Rényi Differential Privacy)|
| • Directional Refusal Probes         | • SISA Exact Unlearning & SCRUB      |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Model Merging & Weight Surgery]     | [Quantum ML & HDC Genomics]          |
| • TIES-Merging, DARE, SLERP, RegMean | • Variational Quantum (VQC Shifts)   |
| • Task Vectors & Disjoint Merging    | • HDC Vector Symbolic Architecture   |
+──────────────────────────────────────┼──────────────────────────────────────+
| [Data Valuation & Meta-Learning]     | [Uncertainty & Information]          |
| • Data Shapley & Beta Shapley        | • Conformalized Quantile Reg (CQR)   |
| • MMD Kernel Drift Detectors         | • Information Bottleneck Principle   |
| • MAML, Reptile & MAP-Elites         | • Continuous Synthetic Tabular CTGAN |
+──────────────────────────────────────┴──────────────────────────────────────+
```

### 12.1 AI Safety & LLM Watermarking (`chokkhu.safety`)
- **Statistical Green/Red Token Watermarking**:
  Pseudo-randomly hashes preceding tokens to partition vocabulary into green list $G$ and red list $R$, adding logit bias $\delta$:
  $$\tilde{l}_k = \begin{cases} l_k + \delta & k \in G \\ l_k & k \in R \end{cases}, \quad z = \frac{|s|_G - \gamma |s|}{\sqrt{|s|\gamma(1 - \gamma)}} > z_{\text{threshold}}$$
- **Directional Refusal Probes**: Linear activation probes identifying safety refusal directions in residual activations.

### 12.2 Privacy, Federated Learning & Machine Unlearning (`chokkhu.privacy`)
- **Differential Privacy (DP-SGD)**:
  Per-sample gradient clipping with calibrated Gaussian noise addition:
  $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \text{clip}\left(g_i, C\right) + \mathcal{N}\left(0, \sigma^2 C^2 I\right) \right)$$
- **Federated Byzantine-Robust Optimization**: Trimmed Mean, Coordinate-wise Median, and Krum aggregation.
- **SISA Exact & Influence-Function Approximate Machine Unlearning**.

### 12.3 Model Merging & Weight Surgery (`chokkhu.models.merging`)
- **TIES-Merging & DARE**: Resolves sign disagreement and parameter interference across multi-task fine-tuned checkpoints without retraining.
- **SLERP (Spherical Linear Interpolation)** & **RegMean (Regression Mean)**.

### 12.4 Quantum ML, HDC & Genomics (`chokkhu.quantum`, `chokkhu.hdc`, `chokkhu.models.bio`)
- **Variational Quantum Classifier (VQC)**: Parameterized quantum circuits ($R_x, R_y, R_z$, CNOT) trained via the exact Parameter-Shift Rule:
  $$\frac{\partial \langle H \rangle}{\partial \theta_i} = \frac{\langle H \rangle_{\theta_i + \pi/2} - \langle H \rangle_{\theta_i - \pi/2}}{2}$$
- **Hyperdimensional Computing (HDC)**: 10,000-dimensional bipolar vectors $\{-1, 1\}^D$ with binding (XOR/Hadamard), bundling (majority rule), and permutation shifts.

### 12.5 Uncertainty Quantification, Data Valuation & Meta-Learning (`chokkhu.uncertainty`, `chokkhu.evaluation`, `chokkhu.models.meta`)
- **Conformalized Quantile Regression (CQR)**:
  Constructs prediction intervals with mathematically guaranteed coverage $1 - \alpha$:
  $$C(x) = [\hat{q}_{\alpha/2}(x) - Q_{1-\alpha}(E), \hat{q}_{1-\alpha/2}(x) + Q_{1-\alpha}(E)]$$
- **Data Shapley & Beta Shapley**: Computes data valuation scores reflecting marginal utility of training samples.
- **Maximum Mean Discrepancy (MMD)**: Non-parametric kernel two-sample test for dataset drift detection.
- **MAML / Reptile Meta-Learning** & **MAP-Elites Quality Diversity**.

---
## 13. Pillar XI: The Grand Unified Universal Pipeline Engine (`chokkhu.pipeline`)

```
+========================================================================================+
| CHOKKHU UNIVERSAL PIPELINE ENGINE (ck.pipeline, ck.Pipeline)                           |
+========================================================================================+
| [Auto-Ingestion]  ──► CSV / Parquet / Images / Audio / SafeTensors                     |
| [Auto-Cleaner]    ──► MICE Imputer / Matrix Factorization / Isolation Forest           |
| [Auto-Preprocess] ──► Yeo-Johnson / OOF Target Encoder / WoE / RFECV Selector          |
| [Auto-Transform]  ──► Deep Feature Synthesis / UMAP / SMOTE-Tomek                      |
| [Auto-Dispatcher] ──► Task Inference (Tabular, Vision, TimeSeries, Causal, Survival)  |
| [SuperLearner]    ──► Out-of-Fold Nelder-Mead Meta-Stacking Ensemble                   |
| [Report Builder]  ──► Standalone Dark-Mode Interactive HTML Diagnostic Dashboard       |
+========================================================================================+
```

### 13.1 Universal 1-Line Pipeline Architecture (`ck.pipeline(...)`)

Execute an entire leak-free, production-ready machine learning pipeline in a single line of code:

```python
import chokkhu as ck

# 1-Line Universal End-to-End Orchestration
result = ck.pipeline(
    data="train_data.parquet",
    target="label",
    task="auto",               # Automatically infers classification, regression, timeseries, causal, etc.
    clean=True,                # Autonomous MICE imputation and anomaly scrubbing
    feature_engineering=True,  # Automated Deep Feature Synthesis (DFS)
    stacking=True,             # Multi-layer SuperLearner stacking with out-of-fold optimization
    conformal_interval=0.95,   # Exact 95% conformal prediction bands
    report="diagnostic.html"   # Generates standalone interactive dark-mode HTML dashboard
)

# Predict on unseen production data with 0 data leakage
predictions = result.predict("unseen_test.parquet")
```

### 13.2 Fluent Composable Chaining API (`ChokkhuPipeline`)

For granular custom workflows, Chokkhu provides a type-safe fluent chaining API:

```python
from chokkhu.pipeline import ChokkhuPipeline
from chokkhu.cleaning import KNNImputer
from chokkhu.preprocessing import QuantileScaler, TargetEncoder
from chokkhu.transformation import PCA, SMOTE
from chokkhu.models.trees import HistGradientBoostingClassifier

pipe = (
    ChokkhuPipeline()
    .add_cleaner(KNNImputer(k=5))
    .add_preprocessor(TargetEncoder(smoothing=10.0))
    .add_preprocessor(QuantileScaler(n_quantiles=100))
    .add_transformer(SMOTE(sampling_strategy="auto"))
    .add_transformer(PCA(n_components=12))
    .add_model(HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6))
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 13.3 Multi-Task Universal Pipeline Dispatcher

The auto-dispatcher analyzes metadata, cardinality, column types, and temporal ordering to automatically configure:
1. **Tabular Classification & Regression**: Dispatches HistGradientBoosting, Model Trees, SVM, and Neural Stacks.
2. **Time Series Forecasting & State Filtering**: Dispatches Auto-ARIMA, UKF, Particle Filters, and PatchTST.
3. **Causal Treatment Inference**: Dispatches DoubleML, DR-Learners, and NOTEARS DAG Discovery.
4. **Survival & Event-Time Modeling**: Dispatches Cox-PH, DeepSurv, and Random Survival Forests.
5. **Unsupervised Anomaly Detection & Clustering**: Dispatches Isolation Forest, LOF, UMAP, and GMM.

### 13.4 Strict Zero-Leakage State Preservation

The pipeline maintains isolated state records:
- `PreprocessorState`: Encapsulates scaling quantiles, encoding maps, and variance masks.
- `TransformationState`: Encapsulates projection bases ($W_{\text{PCA}}$, simplicial sets).
- `ModelWeightsState`: Encapsulates learned tree structures and neural graph weights.

### 13.5 Standalone Dark-Mode Interactive HTML Diagnostic Dashboard Generator

At the completion of pipeline training, `reports/html_builder.py` compiles an interactive, self-contained HTML dashboard:
- Embedded CSS Grid with dark theme (`#0F172A` slate background).
- Interactive SVG charts (Loss Curves, ROC/PR Curves, Residual Plots, Feature Importances, Confusion Matrices).
- Conformal Prediction Band visualizers.
- Zero external CDN dependencies (100% offline viewable).

---

## 14. Verification, Quality Assurance & Release Protocol

### 14.1 Continuous Testing & Verification
- **Unit Testing**: 430+ strict test suites verifying every layer, function, and mathematical operator from first principles.
- **Numerical Invariance Tests**: Validates that outputs match reference mathematical formulas within double-precision tolerance ($\text{tol} \le 10^{-6}$).
- **Leakage Invariance Checks**: Asserts that shuffling test data or introducing future timestamps yields zero change in training parameters.

### 14.2 Code Standards & Build Automation
- **Type Checking**: Strict `mypy` type validation across all modules.
- **Formatting**: 100% compliant with `black` and `flake8` standards.
- **Release Protocol**: Multi-platform wheel packaging via `uv` and automated PyPI distribution.

---
### *CHOKKHU: Sovereign Intelligence, Engineered from First Principles.*
