# CHOKKHU (চক্ষু) — The Sovereign All-in-One AI & Machine Learning Pipeline Framework
## The Ultimate Ultra-Pro-Max Architectural Master Blueprint
### "Minimalistic Code. Maximum Output. Zero Heavy Dependencies. Infinite Intelligence."

---

## 1. Executive Vision & Sovereign Philosophy

### 1.1 The Mission of Chokkhu
Modern AI and Data Science ecosystems are crippled by **dependency bloat**, **fragile binary environments**, **complex boilerplate**, and **hidden data leakage**. A practitioner building an end-to-end intelligent system is forced to juggle gigabytes of external frameworks:
`torch`, `torchvision`, `tensorflow`, `keras`, `scikit-learn`, `transformers`, `huggingface_hub`, `tokenizers`, `diffusers`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `shap`, `lime`, `statsmodels`, `gym`.

**Chokkhu (চক্ষু) is built to be the single sovereign Python package that replaces them all with mathematical purity.**

Every single algorithm—from basic data imputation to classical tree ensembles, Convolutional Neural Networks (CNNs), Vision Transformers (ViT), Auto-Regressive Large Language Models (Mini-GPT), Denoising Diffusion Probabilistic Models (DDPM), and Deep Reinforcement Learning (DQN/A2C)—is engineered **100% from scratch using pure NumPy, SciPy, Pandas, OpenCV-headless, Matplotlib, and Seaborn.**

```
                                 CHOKKHU ECOSYSTEM
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 UNIFIED 1-LINE API                                     │
│  chokkhu.load() | clean() | preprocess() | transform() | split() | train() | evaluate()│
│                               chokkhu.pipeline() | explain()                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
        │                 │                  │                 │                 │
        ▼                 ▼                  ▼                 ▼                 ▼
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │ Tabular ML  │   │  Deep Tensor│   │  Computer   │   │     NLP &   │   │ Generative  │
 │ & Ensembles │   │  & Autograd │   │   Vision    │   │Transformers │   │     AI      │
 │ (RF, GBM,   │   │ (Graph, SGD,│   │ (Conv2D,    │   │(BPE, Multi- │   │ (VAE, GAN,  │
 │  SVM, KNN)  │   │  Adam, Loss)│   │  ResNet,ViT)│   │ Head, GPT)  │   │  Diffusion) │
 └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
        │                 │                  │                 │                 │
        └─────────────────┴──────────────────┼─────────────────┴─────────────────┘
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │          CORE PURE FOUNDATION (STRICT)       │
                     │  NumPy  •  SciPy  •  Pandas  •  OpenCV-Head  │
                     │  No PyTorch • No TensorFlow • No Scikit-Learn│
                     └──────────────────────────────────────────────┘
```

---

### 1.2 Core Design Principles

1. **Pure First-Principles Implementation (Zero Heavy Dependencies)**:
   - **STRICTLY ALLOWED**: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `opencv-python-headless`, `tqdm`.
   - **STRICTLY BANNED**: `torch`, `torchvision`, `tensorflow`, `keras`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `transformers`, `huggingface_hub`, `tokenizers`, `diffusers`, `statsmodels`, `shap`, `lime`, `gym`.
2. **Minimalist 1-Line Developer Experience (DX)**:
   - Every complex operation can be executed with a single intuitive function call (e.g. `ck.train("gpt", ...)`, `ck.train("resnet", ...)`, `ck.pipeline(...)`).
3. **Guaranteed Zero Data Leakage**:
   - Automated state preservation (`PreprocessorState`, `TransformationState`) ensuring strict statistical isolation between training folds and test/inference data.
4. **Featherweight Package Size (< 5 MB Wheel)**:
   - Ultra-fast download, zero C-compiler bottlenecks, instant CI/CD installation, and minimal runtime memory footprint.
5. **Research-Grade Mathematical Rigor**:
   - Exact backpropagation equations, vectorized matrix implementations (e.g., `im2col` GEMM for convolutions, scaled dot-product attention), and numerical stability safeguards ($\\epsilon$ clipping, log-sum-exp tricks).
6. **Self-Contained Interactive Reporting**:
   - Dark-mode responsive HTML dashboards with base64 embedded interactive visual analytics.

---

## 2. Complete Architectural Directory Structure

```
chokkhu/
├── __init__.py                        # Unified Top-Level Entry API
├── cli.py                             # Command Line Interface (CLI)
├── core/                              # Shared Utilities & Base Primitives
│   ├── tensor.py                      # Pure NumPy Autograd Tensor & Computation Graph
│   ├── base.py                        # Base ChokkhuModel & Transformer classes
│   ├── logger.py                      # Colored stream logger
│   ├── visualizer.py                  # Standard plotting helper
│   └── config.py                      # Global configuration & seed manager
│
├── io/                                # Phase 1: Universal Ingestion & Storage
│   ├── loader.py                      # Tabular, Image, Text, Audio, Parquet, Excel loader
│   ├── exporters.py                   # Data serialization (CSV, Parquet, Feather, JSON)
│   └── streaming.py                   # Chunked out-of-core file reader
│
├── eda/                               # Phase 2: Exploratory Data Analysis Engine
│   ├── tabular/                       # Univariate, Bivariate, Multivariate (VIF, PSI, Mahalanobis)
│   ├── image/                         # Resolution, Blur, Entropy, SNR, pHash duplicates
│   ├── text/                          # Word distributions, Zipf's Law, n-grams, perplexity
│   └── timeseries/                    # Autocorrelation (ACF/PACF), Seasonal Decomposition
│
├── cleaning/                          # Phase 3: Data Cleaning & Sanitation
│   ├── missing.py                     # Mean, Median, Mode, KNN, MICE Iterative, Spline
│   ├── outliers.py                    # Tukey IQR, Z-Score, Modified MAD, Isolation Forest
│   ├── duplicates.py                  # Exact & Approximate duplicate detection
│   └── dtype_fixer.py                 # Intelligent dtype coercion & datetime parsing
│
├── preprocessing/                     # Phase 4: Feature Preprocessing & Selection
│   ├── scaling.py                     # Standard, MinMax, Robust, Power (Yeo-Johnson), Quantile
│   ├── encoding.py                    # OneHot, Binary, Target, Frequency, Ordinal, Hash
│   └── feature_selection.py           # Variance, Correlation, Mutual Info, ANOVA, RFESelector
│
├── transformation/                    # Phase 5: Space Projections, Resampling & Augmentation
│   ├── decomposition.py               # PCA, SVD, LDA, t-SNE, UMAP-approximation
│   ├── resampling.py                  # SMOTE, ADASYN, Tomek Links, SMOTETomek, Undersampling
│   ├── feature_engineering.py         # Polynomial, Cross-interactions, Binning, Log
│   ├── augmentation_vision.py         # Flip, Rotate, Crop, Blur, Cutout, MixUp, ColorJitter
│   └── augmentation_text.py           # Synonym replacement, Random swap, Random deletion
│
├── splitting/                         # Phase 6: Leak-Free Partitioning
│   ├── engine.py                      # Train/Test, Train/Val/Test (3-way)
│   └── cross_validation.py            # KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
│
├── models/                            # Phase 7, 8, 9, 10, 11, 12, 13: Full Modeling Universe
│   ├── engine.py                      # Universal chokkhu.train() dispatcher & Auto-Tuning
│   │
│   ├── ml/                            # Phase 7: Classical ML & Ensembles
│   │   ├── linear_regression.py       # OLS, Ridge, Lasso, ElasticNet (GD & Normal Eq)
│   │   ├── logistic_regression.py     # Binary & Multinomial Logistic Regression
│   │   ├── knn.py                     # K-Nearest Neighbors (Classification & Regression)
│   │   ├── decision_tree.py           # CART Tree with Gini, Entropy, MSE, MAE criteria
│   │   ├── random_forest.py           # Bagging Ensemble with OOB scoring & feature subsets
│   │   ├── gradient_boosting.py       # Gradient Tree Boosting with Log-Odds & Deviance
│   │   ├── svm.py                     # Support Vector Machine (Linear, RBF, Poly kernels)
│   │   ├── naive_bayes.py             # Gaussian, Multinomial, Bernoulli Naive Bayes
│   │   ├── kmeans.py                  # K-Means++ with Inertia & Elbow evaluation
│   │   ├── dbscan.py                  # Density-Based Spatial Clustering
│   │   └── hierarchical.py            # Agglomerative Clustering (Single, Complete, Ward)
│   │
│   ├── dl/                            # Phase 8: Autograd Deep Learning Engine
│   │   ├── autograd.py                # Computational Graph, Tensor, Backpropagation
│   │   ├── layers.py                  # Dense, Dropout, BatchNorm1D, LayerNorm, Flatten
│   │   ├── activations.py             # ReLU, GELU, SiLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   │   ├── losses.py                  # MSE, MAE, CrossEntropy, BinaryCrossEntropy, Huber, Focal
│   │   ├── optimizers.py              # SGD (Momentum, Nesterov), Adam, AdamW, RMSProp
│   │   ├── schedulers.py              # StepLR, CosineAnnealingLR, ExponentialLR
│   │   ├── sequential.py              # Modular Sequential neural network builder
│   │   └── callbacks.py               # EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
│   │
│   ├── vision/                        # Phase 9: Computer Vision & CNN Architectures
│   │   ├── conv_layers.py             # Conv2D (im2col), ConvTranspose2D, MaxPool2D, AvgPool2D
│   │   ├── batchnorm2d.py             # 2D Spatial Batch Normalization
│   │   ├── architectures/             # Built-in Vision Models from Scratch
│   │   │   ├── lenet.py               # LeNet-5 Architecture
│   │   │   ├── alexnet.py             # AlexNet Architecture
│   │   │   ├── vgg.py                 # VGG-11, VGG-16 Architectures
│   │   │   ├── resnet.py              # ResNet-18, ResNet-34, ResNet-50 (Residual Skip Blocks)
│   │   │   ├── mobilenet.py           # MobileNetV1 / V2 (Depthwise Separable Convolutions)
│   │   │   ├── unet.py                # U-Net for Image Segmentation (Encoder-Decoder Skips)
│   │   │   └── vit.py                 # Vision Transformer (Patch Extraction + MultiHead Attention)
│   │   ├── detection.py               # Anchor Boxes, IoU, Non-Maximum Suppression (NMS), YOLO head
│   │   └── xai_vision.py              # Grad-CAM, Grad-CAM++, Saliency Maps, Integrated Gradients
│   │
│   ├── nlp/                           # Phase 10: NLP, Sequence Models & Transformers
│   │   ├── tokenizers/                # Pure NumPy Tokenization Engines
│   │   │   ├── bpe.py                 # Byte-Pair Encoding (BPE) Tokenizer from Scratch
│   │   │   ├── wordpiece.py           # WordPiece Tokenizer from Scratch
│   │   │   └── character.py           # Char-level & Word-level fallbacks
│   │   ├── embeddings.py              # Token Embeddings + Sinusoidal / Learned / RoPE Embeddings
│   │   ├── rnn.py                     # Vanilla Recurrent Neural Network (RNN)
│   │   ├── lstm.py                    # Long Short-Term Memory (LSTM) with Full Gate Calculus
│   │   ├── gru.py                     # Gated Recurrent Unit (GRU)
│   │   ├── attention.py               # Scaled Dot-Product & Multi-Head Self/Cross Attention
│   │   ├── transformer_blocks.py      # Transformer Encoder & Decoder Blocks (Pre/Post-LN)
│   │   └── architectures/             # Large Language Models & Encoders from Scratch
│   │       ├── mini_gpt.py            # Causal Decoder-Only LLM (GPT-style, KV-Cache, Top-p/k)
│   │       ├── mini_bert.py           # Masked Encoder-Only Transformer (BERT-style MLM)
│   │       └── seq2seq.py             # Encoder-Decoder Transformer for Translation & Summary
│   │
│   ├── generative/                    # Phase 11: Generative AI & Latent Space Models
│   │   ├── autoencoder.py             # Vanilla, Denoising & Sparse Autoencoders
│   │   ├── vae.py                     # Variational Autoencoder (Reparameterization Trick, KL)
│   │   ├── gan.py                     # Generative Adversarial Networks (Vanilla & DCGAN)
│   │   └── diffusion.py               # Denoising Diffusion Probabilistic Models (DDPM from Scratch)
│   │
│   ├── timeseries/                    # Phase 12: Time Series Modeling & Forecasting
│   │   ├── arima.py                   # AR, MA, ARMA, ARIMA, SARIMA from Scratch
│   │   ├── exponential_smoothing.py   # Single, Double (Holt), Triple (Holt-Winters) Smoothing
│   │   ├── tcn.py                     # Temporal Convolutional Network (Dilated Causal Convolutions)
│   │   └── time_transformer.py        # Time Series Attention Transformer
│   │
│   └── rl/                            # Phase 13: Reinforcement Learning
│       ├── environments.py            # Lightweight Built-in Environments (GridWorld, CartPole-lite)
│       ├── q_learning.py              # Tabular Q-Learning & SARSA
│       ├── dqn.py                     # Deep Q-Network (DQN) with Replay Buffer & Target Network
│       ├── policy_gradient.py         # REINFORCE with Baseline
│       └── actor_critic.py            # Advantage Actor-Critic (A2C)
│
├── evaluation/                        # Phase 14: Comprehensive Metric & Diagnostic Engine
│   ├── metrics.py                     # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log-Loss,
│   │                                  # MSE, RMSE, MAE, R2, Silhouette, Davies-Bouldin, BLEU, mAP
│   └── engine.py                      # Universal evaluate() with EvaluationResult object
│
├── explainability/                    # Phase 15: Unified Explainable AI (XAI)
│   ├── permutation.py                 # Permutation Feature Importance
│   ├── shap_engine.py                 # KernelSHAP & Additive Feature Attributions
│   ├── lime_engine.py                 # LIME Local Interpretable Surrogate Model
│   ├── pdp_engine.py                  # Partial Dependence Plots (PDP) & ICE Curves
│   └── engine.py                      # Universal chokkhu.explain() interface
│
├── pipeline/                          # Phase 16: Zero-Leakage End-to-End Orchestrator
│   └── engine.py                      # chokkhu.pipeline() with PipelineResult & Serialization
│
└── reports/                           # Phase 17: Interactive HTML Dashboard Generator
    ├── html_builder.py                # Self-contained Responsive Dark-Mode Dashboard
    └── templates/                     # CSS grids, SVG charts, and interactive tabs
```

---

## 3. Detailed Phase-by-Phase Technical Specifications

```
==========================================================================================
                              CHOKKHU 17-PHASE MATRIX
==========================================================================================
 Phase 1: Universal I/O Loader & Streamer      Phase 10: NLP, LLMs & Transformers
 Phase 2: Statistical & Visual EDA Engine     Phase 11: Generative AI (VAE, GAN, Diffusion)
 Phase 3: Autonomous Data Cleaning            Phase 12: Time Series & Forecasting
 Phase 4: Stateful Preprocessing & Selection  Phase 13: Advanced Reinforcement Learning
 Phase 5: Dimensionality & Resampling         Phase 14: Mathematical Evaluation Engine
 Phase 6: Leak-Free Splitting Engine          Phase 15: Explainable AI (XAI) Universe
 Phase 7: Classical ML & Tree Ensembles       Phase 16: Zero-Leakage Auto-Pipeline Engine
 Phase 8: Autograd Tensor & DL Engine         Phase 17: Interactive Master HTML Dashboard
 Phase 9: Computer Vision & CNN Architectures
==========================================================================================
```

---

### PHASE 1: Universal I/O Loader & Streaming Engine (`chokkhu.load`)

#### Purpose
Universal data ingestion module that automatically detects file signatures, parses formats, streams large datasets without out-of-memory errors, and accepts tabular data, image directories, text corpora, and audio signals.

#### API Signature
```python
chokkhu.load(
    path: str | list[str],             # File path, directory, or list of files
    format: str = "auto",              # "auto", "csv", "tsv", "json", "parquet", "excel", "feather", "image", "text"
    type: str = "tabular",             # "tabular", "image", "text", "timeseries"
    chunk_size: int | None = None,     # For out-of-core chunked streaming
    img_size: tuple[int, int] = (224, 224), # Image resizing target (H, W)
    color_mode: str = "rgb",           # "rgb", "grayscale", "bgr"
    normalize: bool = False,           # Normalize image pixels to [0, 1]
    verbose: bool = True
) -> pd.DataFrame | dict[str, np.ndarray]
```

---

### PHASE 2: Exploratory Data Analysis (EDA) Engine (`chokkhu.eda`)

#### Tabular EDA (`chokkhu.eda.tabular`)
- **Univariate Analysis**: Automatic type inference (Continuous, Discrete, Nominal, Ordinal), Empirical Cumulative Distribution Function (ECDF), Parametric Distribution Fitting (Gaussian, Gamma, Beta, Log-Normal via maximum likelihood), Skewness ($G_1$), Kurtosis ($G_2$), Tukey IQR and Z-Score outlier counts.
- **Bivariate Analysis**: Pearson ($r$), Spearman ($\\rho$), Point-Biserial, Cramér's $V$, ANOVA $F$-statistic, Chi-Squared ($\\chi^2$) test of independence, top-10 ranked interaction plots.
- **Multivariate Analysis**: Variance Inflation Factor ($VIF = \\frac{1}{1 - R_j^2}$), Robust Mahalanobis Distance ($D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}$) with memory-safe $O(N \\times D)$ projection, Population Stability Index ($PSI = \\sum (P_i - Q_i) \\ln(P_i / Q_i)$) dataset drift quantification.

#### Image EDA (`chokkhu.eda.image`)
- **Spatial & Quality Analysis**: Dimension distributions, Aspect ratios, Laplacian variance blur detection ($\\sigma_{\\Delta}^2$), Pure NumPy Shannon 2D Entropy ($H = -\\sum p_i \\log_2 p_i$), Signal-to-Noise Ratio (SNR), Gray-Level Co-occurrence Matrix (GLCM) Contrast and Homogeneity.
- **Perceptual Duplicate Detection**: 64-bit Discrete Cosine Transform (DCT) Perceptual Hash (`pHash`) clustering to detect identical or near-duplicate images.

---

### PHASE 3: Autonomous Data Cleaning Engine (`chokkhu.clean`)

#### Imputation Algorithms (Pure NumPy / SciPy)
1. **Mean / Median / Mode**: Fast vector imputation with category-safe mapping.
2. **KNN Imputer (From Scratch)**: NaN-aware Euclidean pairwise distance computation and distance-weighted neighbor averaging:
   $$\\hat{x}_{i, j} = \\frac{\\sum_{k \\in N_K(i)} \\frac{1}{d(i, k)} x_{k, j}}{\\sum_{k \\in N_K(i)} \\frac{1}{d(i, k)}}$$
3. **MICE / Iterative Imputer (From Scratch)**: Iterative chained linear ridge regressions predicting missing values across correlated feature dimensions until convergence ($||X^{(t)} - X^{(t-1)}|| < \\epsilon$).
4. **Time-Series Interpolation**: Spline, Linear, Polynomial, Forward-Fill, and Backward-Fill.

#### Outlier Detection & Treatment
- **Tukey IQR**: Boundaries at $[Q_1 - 1.5 \\cdot IQR, Q_3 + 1.5 \\cdot IQR]$.
- **Modified Z-Score**: Using Median Absolute Deviation ($MAD = \\text{median}(|x_i - \\tilde{x}|)$):
  $$M_i = \\frac{0.6745 \\cdot (x_i - \\tilde{x})}{MAD}$$
- **Isolation Forest (From Scratch)**: Pure NumPy randomized axis-aligned partitioning trees with path length normalization on subsample size $\\psi$:
  $$c(\\psi) = 2 \\ln(\\psi - 1) + 0.5772156649 - \\frac{2(\\psi - 1)}{\\psi}$$
  $$s(x, \\psi) = 2^{-\\frac{E(h(x))}{c(\\psi)}}$$
- **Outlier Actions**: `"remove"` (drop rows), `"cap"` (Winsorize percentiles), `"nan"` (convert to NaN for imputer).

---

### PHASE 4: Stateful Preprocessing & Feature Selection (`chokkhu.preprocess`)

#### Feature Scalers (`PreprocessorState`)
- **StandardScaler**: $z = (x - \\mu) / \\sigma$
- **MinMaxScaler**: $z = (x - x_{\\min}) / (x_{\\max} - x_{\\min})$
- **RobustScaler**: $z = (x - Q_2) / (Q_3 - Q_1)$
- **PowerScaler (Yeo-Johnson from Scratch)**: Maximum likelihood estimation of transformation parameter $\\lambda$:
  $$\\psi(\\lambda, y) = \\begin{cases} ((y + 1)^\\lambda - 1) / \\lambda & \\text{if } \\lambda \\neq 0, y \\ge 0 \\\\ \\ln(y + 1) & \\text{if } \\lambda = 0, y \\ge 0 \\\\ -((-y + 1)^{2 - \\lambda} - 1) / (2 - \\lambda) & \\text{if } \\lambda \\neq 2, y < 0 \\\\ -\\ln(-y + 1) & \\text{if } \\lambda = 2, y < 0 \\end{cases}$$
- **QuantileScaler**: Non-linear rank-based uniform/normal quantile mapping.

#### Categorical Encoders
- **OneHotEncoder**: Vectorized indicator matrix creation with unseen category handling.
- **BinaryEncoder**: Logarithmic integer-to-binary bitmask feature expansion.
- **TargetEncoder (Smoothed)**: Out-of-fold empirical Bayes shrinkage with smoothing parameter $m$:
  $$S_c = \\frac{n_c \\cdot \\bar{y}_c + m \\cdot \\bar{y}_{\\text{global}}}{n_c + m}$$
- **Frequency / Ordinal / Label Encoders**.

#### Feature Selectors
- **VarianceThreshold**, **CorrelationFilter**, **MutualInformationSelector**, **ANOVA F-Selector**.
- **RFESelector (Recursive Feature Elimination from Scratch)**: Iteratively fits base estimators, computes feature importances/weights, and eliminates lowest-ranking features.

---

### PHASE 5: Space Projections, Resampling & Augmentation (`chokkhu.transform`)

#### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Covariance eigen-decomposition and SVD $X = U \\Sigma V^T$ with explained variance ratio tracking.
- **Linear Discriminant Analysis (LDA)**: Maximizing Fisher criterion $J(w) = \\frac{w^T S_B w}{w^T S_W w}$ (Between-class vs Within-class scatter).
- **t-SNE (From Scratch)**: Symmetric SNE with Student-t distribution heavy-tailed kernel and Barnes-Hut / Exact gradient descent optimization.

#### Class Imbalance Resampling (Batched $O(N \\times K)$)
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Interpolating synthetic minority instances $x_{\\text{new}} = x_i + \\lambda (x_{zi} - x_i), \\lambda \\sim U(0, 1)$.
- **ADASYN (Adaptive Synthetic)**: Density-weighted adaptive synthesis focusing on difficult boundary samples.
- **Tomek Links & SMOTETomek**: Identifying and pruning borderline ambiguity pairs where nearest neighbor of class $A$ is class $B$.

#### Computer Vision Augmentations (Pure NumPy / OpenCV)
- Horizontal/Vertical Flip, Random Rotation ($[-30^\\circ, +30^\\circ]$), Affine Shear, Random Cropping & Resizing, Gaussian Blur, Random Noise Injection, CutOut (erasing rectangular patches), MixUp ($x = \\lambda x_1 + (1-\\lambda)x_2, y = \\lambda y_1 + (1-\\lambda)y_2$), Color Jitter (Brightness, Contrast, Saturation).

---

### PHASE 6: Leak-Free Data Partitioning (`chokkhu.split`)

- **Train/Test & Train/Val/Test (3-Way)**: Strict single-split partitioning.
- **Stratified Splitting**: Bin-balanced label preserving splits for severe class imbalances.
- **K-Fold & Stratified K-Fold Cross-Validation Generators**.
- **TimeSeriesSplit**: Expanding-window temporal walk-forward evaluation (no lookahead bias).
- **GroupKFold**: Preserving entity / patient / customer grouping across folds.

---

### PHASE 7: Classical ML & Tree Ensembles (`chokkhu.train`)

1. **Linear Regression**: Ordinary Least Squares (OLS via Moore-Penrose pseudo-inverse), Ridge ($L_2$), Lasso ($L_1$ coordinate descent), ElasticNet.
2. **Logistic Regression**: Binary sigmoid and Multinomial softmax gradient descent with $L_1/L_2$ penalties.
3. **K-Nearest Neighbors (KNN)**: Vectorized KD-tree / Ball-tree / Matrix Euclidean distance neighbor voting and distance weighting.
4. **CART Decision Trees**: Gini impurity, Shannon Entropy, MSE, and MAE splitting with pre-pruning (`max_depth`, `min_samples_split`, `min_samples_leaf`).
5. **Random Forest**: Bootstrap aggregating (Bagging) ensemble with sub-feature sampling (`max_features="sqrt"|"log2"`), Out-of-Bag (OOB) error estimation, and `predict_proba`.
6. **Gradient Boosting Machine (GBM)**: First-principles gradient tree boosting with initial log-odds $F_0 = \\ln(\\bar{y} / (1 - \\bar{y}))$, pseudo-residuals negative gradient fitting, shrinkage step size (learning rate), and tree leaf value optimization.
7. **Support Vector Machines (SVM)**: Sequential Minimal Optimization (SMO) algorithm with Linear, Polynomial, and Radial Basis Function (RBF) kernels:
   $$K(x, z) = \\exp(-\\gamma ||x - z||^2)$$
8. **Naive Bayes**: Gaussian Naive Bayes (log-likelihood densities), Multinomial Naive Bayes (Laplace smoothing), and Bernoulli Naive Bayes.
9. **Unsupervised Clustering**: K-Means++ (probabilistic centroid initialization and Voronoi iteration), DBSCAN (density reachability core/border/noise classification), Agglomerative Hierarchical (Ward, Single, Complete linkage).
10. **Reinforcement Learning (Tabular)**: Q-Learning with Bellman Optimality Equation:
    $$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$

---

### PHASE 8: Pure NumPy Autograd & Deep Learning Core

#### The Computational Graph & Tensor Engine (`chokkhu.core.tensor.Tensor`)
```
                      ┌──────────────────────────────────────────────┐
                      │              CHOKKHU AUTOGRAD TENSOR         │
                      │  data: np.ndarray, grad: np.ndarray          │
                      │  creator: Op, requires_grad: bool            │
                      └──────────────────────────────────────────────┘
                                      │              ▲
                       forward(args)  │              │  backward(grad)
                                      ▼              │
                      ┌──────────────────────────────────────────────┐
                      │               OPERATION NODE (Op)            │
                      │  Add, Mul, MatMul, Conv2D, Relu, Softmax     │
                      └──────────────────────────────────────────────┘
```

#### Neural Network Layers (`chokkhu.models.dl.layers`)
- **Dense / Linear**: $Y = X W + b$, $\\frac{\\partial L}{\\partial W} = X^T \\frac{\\partial L}{\\partial Y}$, $\\frac{\\partial L}{\\partial X} = \\frac{\\partial L}{\\partial Y} W^T$.
- **Dropout**: Inverted dropout with Bernoulli mask $M \\sim \\text{Bernoulli}(1 - p) / (1 - p)$.
- **BatchNorm1D & LayerNorm**: Tracking running mean $\\mu$ and variance $\\sigma^2$, with learnable scale $\\gamma$ and shift $\\beta$.

#### Activations & Mathematical Derivatives
- **ReLU**: $\\max(0, x)$, **GELU**: $x \\cdot \\Phi(x) \\approx 0.5 x (1 + \\tanh(\\sqrt{2/\\pi}(x + 0.044715 x^3)))$, **SiLU / Swish**: $x \\cdot \\sigma(x)$, **LeakyReLU**, **Sigmoid**, **Tanh**, **Softmax** (numerically stabilized with $\\max(x)$ subtraction).

#### Loss Functions & Optimizers
- **Losses**: Mean Squared Error (MSE), Cross-Entropy Loss, Binary Cross-Entropy (with log-sum-exp stabilization), Huber Loss, Focal Loss.
- **Optimizers**:
  - **SGD with Momentum & Nesterov**: $v_t = \\beta v_{t-1} + \\alpha \\nabla L(\\theta)$.
  - **Adam & AdamW**: Adaptive moments with decoupled weight decay:
    $$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$
    $$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$
    $$\\theta_t = \\theta_{t-1} - \\alpha \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_{t-1} \\right)$$
  - **RMSProp & AdaGrad**.

---

### PHASE 9: Computer Vision & Convolutional Architectures (From Scratch)

#### 1. Vectorized Convolution Engine: `im2col` + GEMM
Convolutions in Chokkhu are NOT slow nested 4D Python loops. They are implemented via vectorized **`im2col` (Image-to-Column)** transformations that convert sliding receptive fields into 2D matrices, executing the convolution as a single fast C-level General Matrix Multiply (GEMM via NumPy BLAS):

```
 Input Image Tensor (N, C, H, W)
             │
             ▼  im2col transformation
 ┌─────────────────────────────────────────────────────────┐
 │ 2D Matrix: (N * H_out * W_out, C * K_h * K_w)           │
 └─────────────────────────────────────────────────────────┘
             │
             ▼  GEMM: np.dot(X_col, W_row.T)
 ┌─────────────────────────────────────────────────────────┐
 │ Output Matrix: (N * H_out * W_out, Out_Channels)        │
 └─────────────────────────────────────────────────────────┘
             │
             ▼  col2im reshape
 Output Tensor (N, Out_Channels, H_out, W_out)
```

#### 2. Computer Vision Layers
- **`Conv2D`**: Stride, Padding (`"valid"`, `"same"`, integer), Dilation.
- **`ConvTranspose2D`**: Fractionally strided convolution for upsampling in generators and decoders.
- **`DepthwiseSeparableConv2D`**: Spatial depthwise convolution followed by $1 \\times 1$ pointwise convolution ($9\\times$ parameter reduction).
- **`MaxPool2D` & `AvgPool2D` & `GlobalAvgPool2D`**.
- **`BatchNorm2D`**: Spatial batch normalization across $(N, H, W)$ for each feature map channel.

#### 3. Sovereign Vision Architectures from Scratch
1. **LeNet-5**: Classic 2-stage Conv-Pool-Dense baseline.
2. **AlexNet**: 5 Convolutional layers, MaxPool, ReLU, Dropout.
3. **VGG-11 / VGG-16**: Homogeneous stacks of $3 \\times 3$ convolutions and $2 \\times 2$ MaxPool.
4. **ResNet-18 / ResNet-34 / ResNet-50**:
   - Residual Skip Connections solving vanishing gradients:
     $$F(x) + x \\quad \\text{or} \\quad F(x) + W_s x$$
   - BasicBlock ($3 \\times 3 \\rightarrow 3 \\times 3$) and Bottleneck ($1 \\times 1 \\rightarrow 3 \\times 3 \\rightarrow 1 \\times 1$).
5. **MobileNetV1 / MobileNetV2**: Inverted residual blocks with linear bottlenecks for extreme lightweight edge execution.
6. **U-Net**: Symmetric contracting encoder and expanding decoder with concatenate skip connections for pixel-level semantic segmentation.
7. **Vision Transformer (ViT from Scratch)**:
   - Patch Extraction ($16 \\times 16$ non-overlapping patches flattened to linear projection tokens).
   - Class Token (`[CLS]`) concatenation + 1D learnable Positional Embedding.
   - Stacked Transformer Encoder blocks with Multi-Head Self-Attention.

#### 4. Object Detection Primitives
- Anchor Box generation across multi-scale grids.
- Intersection over Union (IoU) and Generalized IoU (GIoU).
- Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes.
- YOLO-style Multi-Part Loss: Bounding Box Coordinates (CIoU/MSE) + Objectness Confidence (BCE) + Classification (Cross-Entropy).

#### 5. Vision Explainability (XAI)
- **Grad-CAM & Grad-CAM++**: Computing gradients of class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:
  $$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial y^c}{\\partial A_{i, j}^k}, \\quad L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left( \\sum_k \\alpha_k^c A^k \\right)$$
- **Saliency Maps & Integrated Gradients** on image pixels.

---

### PHASE 10: NLP, Sequence Models & Transformers (From Scratch)

#### 1. Pure NumPy Tokenization Engines
1. **Byte-Pair Encoding (BPE from Scratch)**:
   - Starts with character vocabulary.
   - Iteratively counts most frequent symbol pairs and merges them into new tokens until target vocabulary size $V$ is reached.
   - Encodes raw string text into integer token IDs and decodes back with byte fallback.
2. **WordPiece Tokenizer**: Maximizing likelihood of training corpus with `##` subword prefixes.
3. **Character & Whitespace Tokenizers**.

#### 2. Embedding & Positional Encodings
- **Token Embedding Table**: $E \\in \\mathbb{R}^{V \\times d_{\\text{model}}}$.
- **Sinusoidal Positional Encoding**:
  $$PE_{(pos, 2i)} = \\sin\\left( \\frac{pos}{10000^{2i / d_{\\text{model}}}} \\right), \\quad PE_{(pos, 2i+1)} = \\cos\\left( \\frac{pos}{10000^{2i / d_{\\text{model}}}} \\right)$$
- **Rotary Position Embedding (RoPE)**: Rotating query and key vectors in complex 2D planes for superior length extrapolation:
  $$R_{\\Theta, m}^d x_m = \\text{diag}(R_{m \\theta_1}, \\dots, R_{m \\theta_{d/2}}) x_m$$

#### 3. Classic Sequence Models (From Scratch)
1. **Recurrent Neural Network (RNN)**: $h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
2. **Long Short-Term Memory (LSTM)**: Exact computation of Forget Gate ($f_t$), Input Gate ($i_t$), Candidate Cell ($\\tilde{C}_t$), Cell State ($C_t$), Output Gate ($o_t$), and Hidden State ($h_t$):
   $$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$
   $$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i), \\quad \\tilde{C}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$$
   $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
   $$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o), \\quad h_t = o_t \\odot \\tanh(C_t)$$
3. **Gated Recurrent Unit (GRU)**: Reset Gate ($r_t$) and Update Gate ($z_t$).

#### 4. Scaled Dot-Product & Multi-Head Self-Attention
```
 Query (Q), Key (K), Value (V)
       │         │
       ▼         ▼
  MatMul: Q @ K.T
       │
       ▼  Scale by 1 / sqrt(d_k)
 ┌─────────────────────────────────────────────────────────┐
 │ Scaled Dot-Product: S = (Q @ K.T) / sqrt(d_k)           │
 └─────────────────────────────────────────────────────────┘
       │
       ▼  Apply Mask (if Causal: S[upper_tri] = -inf)
 ┌─────────────────────────────────────────────────────────┐
 │ Causal / Padding Masking                                │
 └─────────────────────────────────────────────────────────┘
       │
       ▼  Softmax: P = softmax(S)
 ┌─────────────────────────────────────────────────────────┐
 │ Attention Weights Matrix                                │
 └─────────────────────────────────────────────────────────┘
       │
       ▼  MatMul: P @ V
 Output Context Matrix: Attention(Q,K,V)
```
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M \\right) V$$
$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O$$

#### 5. Transformer Architectures from Scratch
1. **Mini-GPT (Decoder-Only Causal Language Model)**:
   - Stacked Transformer Decoder blocks with Pre-LayerNorm (`LayerNorm` before Attention & MLP).
   - Causal Triangular Lower-Triangular Mask (prevents attending to future tokens).
   - Multi-Layer Perceptron: $\\text{GELU}(X W_1 + b_1) W_2 + b_2$.
   - **Key-Value Caching (KV-Cache)**: Caches past $K$ and $V$ projections during auto-regressive generation for $O(1)$ token step complexity instead of $O(T^2)$.
   - **Sampling Engine**: Greedy, Temperature scaling, Top-$k$ filtering, Top-$p$ (Nucleus) cumulative probability sampling, Repetition penalty.
2. **Mini-BERT (Encoder-Only Transformer)**:
   - Bidirectional Multi-Head Self-Attention (no causal mask).
   - Masked Language Modeling (MLM) head (predicts $15\\%$ masked tokens `[MASK]`).
   - Sequence Classification & Sentence Embedding pooled output head.
3. **Seq2Seq Transformer**: Cross-Attention encoder-decoder for translation and text summarization.

---

### PHASE 11: Generative AI & Latent Space Models (From Scratch)

#### 1. Autoencoders & Variational Autoencoders (VAE)
- **Encoder**: Maps input $x \\rightarrow$ Latent mean $\\mu$ and log-variance $\\log \\sigma^2$.
- **Reparameterization Trick**: Enables backpropagation through stochastic sampling:
  $$z = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)$$
- **Decoder**: Reconstructs $\\hat{x} \\leftarrow z$.
- **Loss Function**: Reconstruction Loss + Kullback-Leibler (KL) Divergence regularization:
  $$\\mathcal{L}_{\\text{VAE}} = \\text{BCE}(x, \\hat{x}) - \\frac{1}{2} \\sum \\left( 1 + \\log \\sigma^2 - \\mu^2 - \\sigma^2 \\right)$$

#### 2. Generative Adversarial Networks (GAN & DCGAN)
- **Generator ($G$)**: Transposed convolutions mapping noise vector $z \\sim \\mathcal{N}(0, I) \\rightarrow$ Synthetic image $\\hat{x}$.
- **Discriminator ($D$)**: Strided convolutions classifying real vs synthetic images $D(x) \\in [0, 1]$.
- **Minimax Objective**:
  $$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]$$

#### 3. Denoising Diffusion Probabilistic Models (DDPM From Scratch)
- **Forward Noise Process ($q(x_t | x_0)$)**: Adding Gaussian noise across timesteps $t \\in \\{1, \\dots, T\\}$ according to linear or cosine variance schedule $\\beta_1, \\dots, \\beta_T$:
  $$x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I), \\quad \\alpha_t = 1 - \\beta_t, \\quad \\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$$
- **Reverse Denoising Process ($p_\\theta(x_{t-1} | x_t)$)**: A neural network (U-Net or MLP with sinusoidal time embeddings) trained to predict the added noise:
  $$\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ ||\\epsilon - \\epsilon_\\theta(x_t, t)||^2 \\right]$$
- **Sampling Algorithm**: Iteratively removes predicted noise step-by-step from pure Gaussian noise $x_T \\sim \\mathcal{N}(0, I)$ down to clean sample $x_0$:
  $$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{1 - \\alpha_t}{\\sqrt{1 - \\bar{\\alpha}_t}} \\epsilon_\\theta(x_t, t) \\right) + \\sigma_t z, \\quad z \\sim \\mathcal{N}(0, I)$$

---

### PHASE 12: Time Series Modeling & Forecasting (From Scratch)

1. **ARIMA & SARIMA**:
   - AutoRegressive (AR($p$)), Moving Average (MA($q$)), Differencing ($d$), Seasonal ($P, D, Q, s$) estimation via maximum likelihood and Yule-Walker equations.
2. **Exponential Smoothing**: Holt-Winters additive and multiplicative trend and seasonality forecasting.
3. **Temporal Convolutional Network (TCN / WaveNet)**:
   - Dilated Causal 1D Convolutions ($d = 1, 2, 4, 8, 16$) providing exponential receptive field without recurrent bottlenecks.
4. **Time Series Attention Transformer**: Multi-horizon quantile forecasting with temporal self-attention.

---

### PHASE 13: Advanced Reinforcement Learning (From Scratch)

1. **Tabular Q-Learning & SARSA**: Epsilon-greedy exploration, discount factor $\\gamma$, learning rate $\\alpha$.
2. **Deep Q-Network (DQN)**:
   - Deep neural network approximating $Q(s, a; \\theta)$.
   - Experience Replay Buffer (breaks temporal correlation between transitions $(s, a, r, s', d)$).
   - Target Network $Q(s, a; \\theta^-)$ updated periodically for stable convergence.
3. **Policy Gradient (REINFORCE with Baseline)**: Direct policy optimization $\\nabla_\\theta J(\\theta) = \\mathbb{E} [ \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) (G_t - b(s_t)) ]$.
4. **Advantage Actor-Critic (A2C)**: Synchronous Actor (policy $\\pi_\\theta$) and Critic (state-value baseline $V_\\phi(s)$) with TD Advantage $A(s, a) = r + \\gamma V(s') - V(s)$.

---

### PHASE 14: Mathematical Evaluation Engine (`chokkhu.evaluate`)

- **Classification**: Accuracy, Precision, Recall, F1-Score (Macro, Micro, Weighted), Specificity, Matthews Correlation Coefficient (MCC), Cohen's Kappa, Log-Loss, ROC-AUC (Trapezoidal integration), PR-AUC, Confusion Matrix, Brier Score.
- **Regression**: MSE, RMSE, MAE, Mean Absolute Percentage Error (MAPE), $R^2$ Score, Adjusted $R^2$, Maximum Error, Median Absolute Error.
- **Clustering**: Silhouette Coefficient (Mean intra-cluster distance $a$ vs nearest-cluster distance $b$), Davies-Bouldin Index, Calinski-Harabasz Dispersion Ratio, Inertia.
- **NLP / Generation**: Perplexity ($PPL = \\exp(L)$), BLEU-1/2/3/4 n-gram precision, ROUGE-1/2/L longest common subsequence.
- **Vision / Detection**: Mean Average Precision (mAP@0.5, mAP@0.5:0.95), Intersection over Union (IoU), Dice Similarity Coefficient.

---

### PHASE 15: Unified Explainable AI (XAI) (`chokkhu.explain`)

- **Permutation Feature Importance**: Shuffling feature columns and measuring score delta $\\Delta S$.
- **KernelSHAP**: Solving weighted linear regression over combinatorial feature coalitions to compute Shapley values:
  $$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|! (|F| - |S| - 1)!}{|F|!} \\left( f(S \\cup \\{i\\}) - f(S) \\right)$$
- **LIME (Local Interpretable Model-agnostic Explanations from Scratch)**: Generating local Gaussian perturbations, weighting samples by exponential kernel distance, and fitting a sparse weighted Lasso surrogate model.
- **Partial Dependence Plots (PDP) & ICE Curves**: Marginalizing over feature grid distributions.
- **Grad-CAM & Saliency Maps** for Vision models.
- **Attention Heatmaps** for Transformer & NLP models.

---

### PHASE 16: Zero-Leakage End-to-End Orchestrator (`chokkhu.pipeline`)

#### The Zero-Leakage Architectural Guarantee
```
 Raw Dataset (File Path or DataFrame)
             │
             ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Load & Baseline Sanitization (Dtypes & Duplicates)   │
 └─────────────────────────────────────────────────────────┘
             │
             ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Strict Split BEFORE Processing (Train / Val / Test)  │
 └─────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────┐
             │ (Train Stream)                              │ (Test / Inference Stream)
             ▼                                             ▼
 ┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
 │ 3. Preprocess (Fit & Transform)       │   │ PreprocessorState.transform()             │
 │    - Scalers, Encoders, Selectors     │   │ (Uses parameters fitted ONLY on Train)    │
 └───────────────────────────────────────┘   └───────────────────────────────────────────┘
             │                                             │
             ▼                                             ▼
 ┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
 │ 4. Feature Transforms (Fit & Trans)   │   │ TransformationState.transform()           │
 │    - PCA, SVD, Polynomial             │   │ (Uses projection matrix fitted on Train)  │
 └───────────────────────────────────────┘   └───────────────────────────────────────────┘
             │                                             │
             ▼                                             │ (NEVER resample Test / Inference!)
 ┌───────────────────────────────────────┐                 │
 │ 5. Resample (SMOTE, ADASYN)           │                 │
 │    (Applied STRICTLY on Train only!)  │                 │
 └───────────────────────────────────────┘                 │
             │                                             │
             ▼                                             │
 ┌───────────────────────────────────────┐                 │
 │ 6. Model Training & Auto-Tuning       │                 │
 │    (Classical ML, Deep Learning, etc.)│                 │
 └───────────────────────────────────────┘                 │
             │                                             │
             ▼                                             ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 7. Evaluation & Metrics on Untouched Test Set                                         │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

#### `PipelineResult` Interface
- `result.predict(new_data)`: Auto-applies fitted preprocessing and transformation states on unseen raw data with zero leakage.
- `result.predict_proba(new_data)`: Returns class probabilities.
- `result.explain(method="shap")`: Explains model predictions.
- `result.save("pipeline.pkl")` and `PipelineResult.load("pipeline.pkl")`.
- `result.summary()`: Detailed formatted report.

---

### PHASE 17: Interactive Master HTML Dashboard (`chokkhu.reports`)

- **Single Self-Contained HTML File**: Zero external CDN dependencies, fully offline capable, base64-encoded SVG/PNG plots.
- **Modern Dark-Mode UI**: Responsive CSS Grid with collapsible cards and interactive navigation tabs:
  1. Executive Summary & Metadata.
  2. Exploratory Data Analysis & Quality Heatmaps.
  3. Cleaning & Preprocessing Audit Trail.
  4. Model Architecture & Hyperparameter Configuration.
  5. Learning Curves (Loss & Metric progression across epochs/iterations).
  6. Evaluation Metrics & Confusion Matrix / ROC Curves.
  7. Explainability (SHAP, Permutation, PDP, Grad-CAM).

---

## 4. Unified Minimalist 1-Line Developer API Matrix

Every single workflow in Chokkhu is triggered with 1 clean line of Python:

```python
import chokkhu as ck

# 1. DATA LOADING
df = ck.load("dataset.parquet")
images = ck.load("image_folder/", type="image", img_size=(224, 224))

# 2. EXPLORATORY DATA ANALYSIS
ck.eda.tabular("data.csv", target_col="target", save_reports=True)
ck.eda.image("image_folder/", save_reports=True)

# 3. DATA CLEANING
df_clean = ck.clean(df, missing="knn", outliers="isolation", duplicates=True)

# 4. PREPROCESSING
df_proc, state = ck.preprocess(df_clean, target="target", scale="robust", encode="onehot", select_features="rfe", select_k=15)

# 5. TRANSFORMATION
df_trans = ck.transform(df_proc, target="target", pca=5, resample="smote")

# 6. DATA SPLITTING
X_train, X_test, y_train, y_test = ck.split(df_trans, target="target", test_size=0.2, stratify=True)

# 7. CLASSICAL ML WITH AUTO-TUNING
model_rf = ck.train(model="random_forest", X_train=X_train, y_train=y_train, tune=True)

# 8. COMPUTER VISION CNN (RESNET FROM SCRATCH)
model_cv = ck.train(model="resnet", X_train=X_images, y_train=y_labels, epochs=50, learning_rate=0.001)

# 9. LARGE LANGUAGE MODEL (MINI-GPT FROM SCRATCH)
model_gpt = ck.train(model="gpt", text="corpus.txt", vocab_size=5000, n_layers=4, n_heads=4, d_model=128, epochs=20)
generated_text = model_gpt.generate("Artificial Intelligence is", max_tokens=100, temperature=0.8, top_p=0.9)

# 10. GENERATIVE AI (DIFFUSION MODEL DDPM FROM SCRATCH)
model_diff = ck.train(model="diffusion", X_train=X_images, timesteps=500, epochs=100)
synthetic_images = model_diff.sample(n_samples=16)

# 11. REINFORCEMENT LEARNING (DQN FROM SCRATCH)
model_rl = ck.train(model="dqn", env="CartPole", episodes=500, gamma=0.99)

# 12. EVALUATION & METRICS
eval_res = ck.evaluate(model_rf, X_test, y_test, save_reports=True)

# 13. EXPLAINABLE AI
explanation = ck.explain(model_rf, X_test, y_test, method="shap")

# 14. THE ULTIMATE ZERO-LEAKAGE PIPELINE (ONE FUNCTION CALL DOES EVERYTHING)
res = ck.pipeline(data="dataset.csv", target="price", model="auto", clean_missing="knn", scale="standard", save_reports=True)
preds = res.predict("new_unseen_data.csv")
```

---

## 5. PyPI Packaging & Featherweight Footprint Optimization Guide

### 5.1 Package Size Target: < 5 MB Wheel
Because Chokkhu relies solely on pure Python algorithms and lightweight foundational dependencies (`numpy`, `scipy`, `pandas`, `opencv-python-headless`, `matplotlib`, `seaborn`, `tqdm`), the entire source and wheel distribution remains **below 5 Megabytes**—compared to tens of gigabytes for PyTorch/TensorFlow/Transformers setups.

### 5.2 Build & Distribution Configuration (`pyproject.toml` / `setup.py`)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chokkhu"
version = "0.2.0"
description = "An End-to-End, Production-Grade Sovereign ML, DL, CV, NLP & Generative Pipeline Framework."
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.8"
authors = [{ name = "Tamim Hossain" }]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "opencv-python-headless>=4.5.0",
    "tqdm>=4.60.0",
]

[project.scripts]
chokkhu = "chokkhu.cli:main"
```

---

## 6. Phased Implementation Roadmap & Milestones

| Milestone | Target Domain | Key Deliverables | Status |
|---|---|---|:---:|
| **M1: Tabular Foundation** | Core ML & Tabular Pipeline | Phases 1, 2, 3, 4, 5, 6, 7, 9 (Tabular), 10 (Permutation/SHAP/PDP), 11, 16, CLI | **100% COMPLETE** (70 Tests Passing) |
| **M2: Deep Autograd Core** | Tensor & Neural Net Engine | Phase 8 (`Tensor` autograd, computation graph, backward pass, AdamW, Loss layers) | Ready for expansion |
| **M3: Computer Vision Engine** | Convolutional & Vision Models | Phase 9 (`im2col` Conv2D, ResNet, MobileNet, U-Net, ViT, Grad-CAM, YOLO primitives) | Ready for expansion |
| **M4: NLP & Transformers** | Sequence Models & LLMs | Phase 10 (BPE Tokenizer from scratch, LSTM/GRU, Multi-Head Attention, Mini-GPT, Mini-BERT) | Ready for expansion |
| **M5: Generative AI & Latent** | Creative AI & Diffusion | Phase 11 (VAE reparameterization, DCGAN, DDPM Denoising Diffusion from scratch) | Ready for expansion |
| **M6: Advanced RL & TimeSeries**| Sequential Decision & Forecasting| Phase 12 (ARIMA, WaveNet TCN, Time Transformer) & Phase 13 (DQN, A2C Actor-Critic) | Ready for expansion |
| **M7: Sovereign Release** | PyPI Deployment & Master Report| Phase 17 (Interactive Master Dark-Mode Dashboard), Packaging, Docs & PyPI Release | Ready for Release |

---

## 7. Summary & Sovereign Declaration

Chokkhu is engineered to stand as a monumental testament to first-principles computer science:
1. **Every Model from Scratch**: Pure linear algebra, calculus, and matrix mathematics.
2. **Minimalist 1-Line API**: Complete pipelines executed in one clean command.
3. **Guaranteed Zero Data Leakage**: Scientific validation integrity.
4. **Featherweight Footprint**: Lowest possible package size without sacrificing capability.
5. **Unified AI Universe**: Tabular, Vision, NLP, Generative, Time-Series, and Reinforcement Learning under one sovereign roof.

*"Give it your data. Get back everything." — That is Chokkhu.*
