# CHOKKHU (চক্ষু) — The Sovereign All-in-One AI & Machine Learning Pipeline Framework
## The Ultimate Ultra-Pro-Max Architectural Master Blueprint
### "Minimalistic Code. Maximum Output. Zero Heavy Dependencies. Infinite Intelligence."

---

## 1. Executive Vision & Sovereign Philosophy

### 1.1 The Mission of Chokkhu
Modern AI and Data Science ecosystems are crippled by **dependency bloat**, **fragile binary environments**, **complex boilerplate**, and **hidden data leakage**. A practitioner building an end-to-end intelligent system is forced to juggle gigabytes of external frameworks:
`torch`, `torchvision`, `tensorflow`, `keras`, `scikit-learn`, `transformers`, `huggingface_hub`, `tokenizers`, `diffusers`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `shap`, `lime`, `statsmodels`, `gym`.

**Chokkhu (চক্ষু) is built to be the single sovereign Python package that replaces them all with mathematical purity.**

Every single algorithm—from basic data imputation to classical tree ensembles, Convolutional Neural Networks (CNNs), Vision Transformers (ViT), Auto-Regressive Large Language Models (Mini-GPT / LLaMA-style), Denoising Diffusion Probabilistic Models (DDPM), and Deep Reinforcement Learning—is engineered **100% from scratch using pure NumPy, SciPy, Pandas, OpenCV-headless, Matplotlib, and Seaborn.**

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

### 1.2 Core Development & Operational Directives

1. **Zero Heavy Dependencies (Strictly Enforced)**:
   - **STRICTLY ALLOWED**: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `opencv-python-headless`, `tqdm`.
   - **STRICTLY BANNED**: `torch`, `torchvision`, `tensorflow`, `keras`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `transformers`, `huggingface_hub`, `tokenizers`, `diffusers`, `statsmodels`, `shap`, `lime`, `gym`.
2. **Minimalist 1-Line Developer Experience (DX)**:
   - Every model and pipeline stage can be executed with a single intuitive function call (e.g. `ck.train("resnet50", ...)`, `ck.train("gpt", ...)`, `ck.train("random_forest", ...)`, `ck.pipeline(...)`).
3. **Guaranteed Zero Data Leakage**:
   - Automated state preservation (`PreprocessorState`, `TransformationState`) ensuring strict statistical isolation between training folds and test/inference data.
4. **Featherweight Package Size (< 5 MB Wheel)**:
   - Ultra-fast download, zero C-compiler bottlenecks, instant CI/CD installation, and minimal runtime memory footprint.
5. **Research-Grade Mathematical Rigor**:
   - Exact backpropagation calculus, vectorized matrix operations (e.g., `im2col` GEMM for convolutions, scaled dot-product attention), and numerical stability safeguards ($\epsilon$ clipping, log-sum-exp tricks).
6. **Humanized Git Commit Messages**:
   - All Git commit messages must be written in a natural, humanized style like a real junior developer (e.g. `add vision models and tests`, `fix import issue in lenet`, `implement nlp tokenizer and transformer blocks`, `fix ci test errors on python 3.12`), completely avoiding robotic AI-generated tags.
7. **Robust Multi-OS & Multi-Python Compatibility Matrix**:
   - All code, linting, and test suites must run cleanly with 100% pass rate across Ubuntu, Windows, and macOS on Python 3.9, 3.10, 3.11, 3.12, and 3.13.

---

## 2. Complete End-to-End Architectural Directory Structure

```
chokkhu/
├── .github/
│   └── workflows/
│       ├── ci.yml                         # Multi-OS & Multi-Python Matrix CI (3.9-3.13)
│       └── publish.yml                    # Automated PyPI Release Pipeline
│
├── src/
│   └── chokkhu/
│       ├── __init__.py                    # Unified 1-Line API entry point
│       ├── py.typed                       # PEP 561 type-hinting compliance marker
│       ├── cli.py                         # Command-line interface
│       │
│       ├── core/                          # Foundation Infrastructure
│       │   ├── __init__.py
│       │   ├── exceptions.py              # Custom exceptions & validation errors
│       │   ├── logger.py                  # Beautiful rich-formatted logging
│       │   ├── config.py                  # Global settings, thread limits, seed control
│       │   └── tensor.py                  # Dynamic Autograd Tensor Engine
│       │
│       ├── io/                            # Phase 2: Multi-Modal Data Ingestion
│       │   ├── __init__.py
│       │   ├── tabular.py                 # CSV, TSV, JSON, Parquet, Feather, Excel loaders
│       │   ├── image.py                   # Multi-threaded image directory & format loaders
│       │   ├── text.py                    # Raw text, corpus, and line-by-line loaders
│       │   ├── audio.py                   # WAV, MP3 byte decoding & signal loader
│       │   └── persistence.py             # Atomic model & pipeline serialization
│       │
│       ├── cleaning/                      # Phase 3: Deep Data Cleaning & Quality Control
│       │   ├── __init__.py
│       │   ├── missing_imputer.py         # Mean, Median, Mode, KNN, Iterative, Constant
│       │   ├── outlier_detector.py        # IQR, Z-score, Isolation Forest, Elliptic Envelope
│       │   ├── duplicate_handler.py       # Exact, subset, and fuzzy duplicate handling
│       │   └── dtype_fixer.py             # Intelligent dtype coercion & datetime parsing
│       │
│       ├── preprocessing/                 # Phase 4: Feature Preprocessing & Scaling
│       │   ├── __init__.py
│       │   ├── scaling.py                 # StandardScaler, MinMaxScaler, RobustScaler, PowerScaler, QuantileScaler
│       │   ├── encoding.py                # OneHotEncoder, OrdinalEncoder, BinaryEncoder, TargetEncoder, FrequencyEncoder, HashEncoder
│       │   └── feature_selection.py       # VarianceThreshold, CorrelationFilter, MutualInfo, ANOVA, RFESelector
│       │
│       ├── transformation/                # Phase 5: Space Projections, Resampling & Augmentation
│       │   ├── __init__.py
│       │   ├── decomposition.py           # PCA, SVD, LDA, t-SNE, KernelPCA
│       │   ├── resampling.py              # SMOTE, ADASYN, Tomek Links, SMOTETomek, RandomUnderSampler
│       │   ├── feature_engineering.py     # PolynomialFeatures, Cross-interactions, Binning, Log
│       │   ├── augmentation_vision.py     # Flip, Rotate, Crop, Blur, Cutout, MixUp, ColorJitter
│       │   ├── augmentation_text.py       # Synonym replacement, Random swap, Random deletion
│       │   └── augmentation_audio.py      # Time stretch, Pitch shift, Noise injection, SpecAugment
│       │
│       ├── splitting/                     # Phase 6: Leakage-Free Data Splitting
│       │   ├── __init__.py
│       │   ├── engine.py                  # Train/Test, Train/Val/Test (3-way split)
│       │   └── cross_validation.py        # KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
│       │
│       ├── models/                        # Phase 7 to 13: The Complete Modeling Universe
│       │   ├── __init__.py
│       │   ├── base.py                    # Base ChokkhuModel abstraction
│       │   ├── engine.py                  # Universal chokkhu.train() dispatcher & Auto-Tuning
│       │   │
│       │   ├── ml/                        # Phase 7: Classical ML & Ensembles
│       │   │   ├── __init__.py
│       │   │   ├── linear_regression.py   # OLS, Ridge, Lasso, ElasticNet (GD & Normal Eq)
│       │   │   ├── logistic_regression.py # Binary & Multinomial Logistic Regression
│       │   │   ├── knn.py                 # K-Nearest Neighbors (Classification & Regression)
│       │   │   ├── decision_tree.py       # CART Tree with Gini, Entropy, MSE, MAE criteria
│       │   │   ├── random_forest.py       # Bagging Ensemble with OOB scoring & feature subsets
│       │   │   ├── extra_trees.py         # Extremely Randomized Trees
│       │   │   ├── gradient_boosting.py   # Gradient Tree Boosting with Log-Odds & Deviance
│       │   │   ├── adaboost.py            # Adaptive Boosting with decision stumps
│       │   │   ├── svm.py                 # Support Vector Machine (Linear, RBF, Poly, Sigmoid)
│       │   │   ├── naive_bayes.py         # Gaussian, Multinomial, Bernoulli, Complement NB
│       │   │   ├── kmeans.py              # K-Means++ with Inertia & Elbow evaluation
│       │   │   ├── kmedoids.py            # K-Medoids with PAM algorithm
│       │   │   ├── dbscan.py              # Density-Based Spatial Clustering
│       │   │   ├── optics.py              # Ordering Points To Identify Clustering Structure
│       │   │   ├── hierarchical.py        # Agglomerative Clustering (Single, Complete, Ward)
│       │   │   ├── gmm.py                 # Gaussian Mixture Models with EM Algorithm
│       │   │   └── anomaly.py             # Isolation Forest, LOF, One-Class SVM
│       │   │
│       │   ├── dl/                        # Phase 8: Sovereign Deep Learning & Autograd
│       │   │   ├── __init__.py
│       │   │   ├── autograd.py            # Computational Graph, Tensor, Reverse Autodiff
│       │   │   ├── layers.py              # Linear, Dropout, BatchNorm1D, LayerNorm, RMSNorm, GroupNorm, Flatten, Embedding
│       │   │   ├── activations.py         # ReLU, GELU, SiLU, LeakyReLU, ELU, SELU, Mish, Sigmoid, Tanh, Softmax
│       │   │   ├── losses.py              # MSE, MAE, CrossEntropy, BinaryCrossEntropy, Huber, Focal, Dice, Triplet
│       │   │   ├── optimizers.py          # SGD, Momentum, Nesterov, Adam, AdamW, RMSProp, Adagrad, Lion
│       │   │   ├── schedulers.py          # StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
│       │   │   ├── callbacks.py           # EarlyStopping, ModelCheckpoint, LRLogger
│       │   │   └── sequential.py          # Sequential Model Builder (.fit, .predict, .predict_proba)
│       │   │
│       │   ├── vision/                    # Phase 9: Complete Computer Vision Universe
│       │   │   ├── __init__.py
│       │   │   ├── conv_layers.py         # Conv2D (im2col), ConvTranspose2D, DepthwiseSeparableConv2D, GroupedConv2D, ChannelShuffle, MaxPool2D, AvgPool2D, GlobalAvgPool2D
│       │   │   ├── batchnorm2d.py         # Spatial 2D Batch Normalization
│       │   │   ├── attention_blocks.py    # SEBlock (Squeeze-and-Excitation), CBAM
│       │   │   ├── detection.py           # Anchor Boxes, IoU, Non-Maximum Suppression (NMS), YOLO-Head
│       │   │   ├── xai_vision.py          # GradCAM, GradCAM++, Saliency Maps
│       │   │   └── architectures/         # Comprehensive CNN & Vision Architectures
│       │   │       ├── __init__.py
│       │   │       ├── lenet.py           # LeNet-5 (1998)
│       │   │       ├── alexnet.py         # AlexNet (2012), ZFNet (2013)
│       │   │       ├── vgg.py             # VGG-11, VGG-13, VGG-16, VGG-19 (2014)
│       │   │       ├── inception.py       # GoogLeNet / InceptionV1, InceptionV3 (2014-2015)
│       │   │       ├── resnet.py          # ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 (2015)
│       │   │       ├── resnext.py         # ResNeXt-50, ResNeXt-101 (2017)
│       │   │       ├── densenet.py        # DenseNet-121, DenseNet-169, DenseNet-201 (2017)
│       │   │       ├── squeezenet.py      # SqueezeNet with Fire Modules (2016)
│       │   │       ├── mobilenet.py       # MobileNetV1, MobileNetV2 (Inverted Residuals), MobileNetV3 (2017-2019)
│       │   │       ├── shufflenet.py      # ShuffleNetV1, ShuffleNetV2 (Channel Shuffle) (2018)
│       │   │       ├── efficientnet.py    # EfficientNet-B0 to B4 (MBConv + Compound Scaling) (2019)
│       │   │       ├── convnext.py        # ConvNeXt-Tiny, ConvNeXt-Small (Modern Pure CNN) (2022)
│       │   │       ├── unet.py            # U-Net Image Segmentation (2015)
│       │   │       ├── fcn.py             # FCN-8s, FCN-32s Semantic Segmentation (2015)
│       │   │       ├── segnet.py          # SegNet with Max-Unpooling (2017)
│       │   │       ├── fpn.py             # Feature Pyramid Network (2017)
│       │   │       ├── vit.py             # Vision Transformer (ViT-Tiny, ViT-Base) (2020)
│       │   │       └── swin.py            # Swin Transformer (Shifted Window Attention) (2021)
│       │   │
│       │   ├── nlp/                       # Phase 10: NLP, Sequence Models & Sovereign Transformers
│       │   │   ├── __init__.py
│       │   │   ├── tokenizers/            # Pure NumPy Tokenization Subsystem
│       │   │   │   ├── __init__.py
│       │   │   │   ├── bpe.py             # Byte-Pair Encoding (BPE) Tokenizer
│       │   │   │   ├── wordpiece.py       # WordPiece Tokenizer (BERT-style)
│       │   │   │   ├── sentencepiece.py   # Unigram / SentencePiece Tokenizer
│       │   │   │   └── character.py       # Character & Word-level fallbacks
│       │   │   ├── embeddings.py          # TokenEmbedding, Sinusoidal, Learned, RoPE (Rotary), ALiBi
│       │   │   ├── rnn.py                 # Vanilla Recurrent Neural Network (RNN)
│       │   │   ├── lstm.py                # Long Short-Term Memory (LSTM) with full gate calculus
│       │   │   ├── gru.py                 # Gated Recurrent Unit (GRU)
│       │   │   ├── attention.py           # Scaled Dot-Product, Multi-Head, Multi-Query (MQA), Grouped-Query (GQA)
│       │   │   ├── transformer_blocks.py  # Pre-LN / Post-LN Transformer Encoder & Decoder Blocks
│       │   │   ├── generation.py          # Greedy, Temperature, Top-K, Top-P Nucleus, Beam Search
│       │   │   └── architectures/         # Sovereign Language Model Architectures
│       │   │       ├── __init__.py
│       │   │       ├── bert.py            # BERT / RoBERTa Bidirectional Transformer Encoder
│       │   │       ├── mini_gpt.py        # GPT-1/2 Autoregressive Causal LLM with KV-Caching
│       │   │       ├── llama.py           # LLaMA / Mistral Modern LLM (RMSNorm, SwiGLU, RoPE, GQA)
│       │   │       └── seq2seq.py         # T5 / Seq2Seq Encoder-Decoder Translation Transformer
│       │   │
│       │   ├── audio/                     # Phase 11: Audio, Speech & Signal Processing
│       │   │   ├── __init__.py
│       │   │   ├── stft.py                # Short-Time Fourier Transform & Inverse STFT
│       │   │   ├── mel_filters.py         # Mel-Scale Filterbanks & Mel-Spectrogram
│       │   │   ├── mfcc.py                # Mel-Frequency Cepstral Coefficients
│       │   │   ├── conv1d.py              # Temporal 1D Convolution & Dilated Convolutions
│       │   │   └── wavenet.py             # WaveNet / SoundNet Audio Generation & Classification
│       │   │
│       │   ├── generative/                # Phase 12: Generative Models & Diffusion
│       │   │   ├── __init__.py
│       │   │   ├── autoencoder.py         # Vanilla Autoencoder & Denoising Autoencoder
│       │   │   ├── vae.py                 # Variational Autoencoder (VAE) with Reparameterization
│       │   │   ├── gan.py                 # Vanilla GAN & Deep Convolutional GAN (DCGAN)
│       │   │   └── diffusion.py           # Denoising Diffusion Probabilistic Model (DDPM) from scratch
│       │   │
│       │   ├── gnn/                       # Phase 13: Graph Neural Networks
│       │   │   ├── __init__.py
│       │   │   ├── graph_ops.py           # Adjacency, Degree, Normalized Graph Laplacian
│       │   │   ├── gcn.py                 # Graph Convolutional Network (GCN) Layer
│       │   │   └── gat.py                 # Graph Attention Network (GAT) Layer
│       │   │
│       │   ├── timeseries/                # Phase 11B: Time Series & Forecasting
│       │   │   ├── __init__.py
│       │   │   ├── decomposition.py       # Seasonal-Trend Decomposition (Classical & Robust)
│       │   │   ├── arima.py               # AR, MA, ARMA, ARIMA, SARIMA Solver
│       │   │   ├── exponential_smoothing.py# Simple, Holt's Linear, Holt-Winters Seasonal
│       │   │   └── dlinear.py             # DLinear & N-BEATS Deep Time Series Forecaster
│       │   │
│       │   └── rl/                        # Phase 7B: Reinforcement Learning
│       │       ├── __init__.py
│       │       ├── q_learning.py          # Tabular Q-Learning with Epsilon-Greedy
│       │       ├── dqn.py                 # Deep Q-Network with Replay Buffer & Target Network
│       │       └── policy_gradient.py     # REINFORCE & Actor-Critic (A2C)
│       │
│       ├── evaluation/                    # Phase 14: Comprehensive Metrics & Diagnostics
│       │   ├── __init__.py
│       │   ├── metrics_classification.py  # Accuracy, Precision, Recall, F1, F-beta, ROC-AUC, PR-AUC, LogLoss, MCC, Kappa, ConfusionMatrix
│       │   ├── metrics_regression.py      # MSE, RMSE, MAE, R2, Adjusted R2, MAPE, SMAPE, Explained Variance
│       │   ├── metrics_clustering.py      # Silhouette, Davies-Bouldin, Calinski-Harabasz
│       │   ├── engine.py                  # Universal chokkhu.evaluate() dispatcher
│       │   └── drift_detector.py          # Population Stability Index (PSI), Wasserstein Distance, KS-Test
│       │
│       ├── explainability/                # Phase 15: Sovereign Explainable AI (XAI)
│       │   ├── __init__.py
│       │   ├── feature_importance.py      # Model-agnostic Permutation Feature Importance
│       │   ├── shap_engine.py             # Sovereign Kernel SHAP & Shapley Value Estimator
│       │   ├── lime_engine.py             # Sovereign LIME Local Surrogate Explainer
│       │   ├── partial_dependence.py      # Partial Dependence Plots (PDP) & ICE Curves
│       │   ├── integrated_gradients.py    # Axiomatic Attribution for Neural Networks
│       │   └── engine.py                  # Universal chokkhu.explain() dispatcher
│       │
│       ├── eda/                           # Exploratory Data Analysis & Dashboards
│       │   ├── __init__.py
│       │   ├── tabular.py                 # Automated Tabular Profiling
│       │   ├── image.py                   # Automated Image Dataset Profiling
│       │   └── reports.py                 # Standalone Dark-Mode HTML Dashboard Generator
│       │
│       └── pipeline/                      # Phase 16: Universal Autonomous AutoML Engine
│           ├── __init__.py
│           ├── core.py                    # ChokkhuPipeline Chaining & Serialization
│           ├── state.py                   # TransformationState & PreprocessorState
│           ├── automl.py                  # Autonomous Task Detection, Model Search & Blending
│           └── tuner.py                   # Grid Search, Random Search, Bayesian-style K-Fold Tuning
│
├── tests/                                 # Comprehensive Test Suite
│   ├── integration/
│   │   ├── test_end_to_end_complete.py    # Full lifecycle tests
│   │   └── test_pipeline_integration.py   # Multi-modal pipeline tests
│   └── unit/
│       ├── test_autograd_tensor.py        # Autograd tensor unit tests
│       ├── test_conv_layers.py            # Convolutional layers unit tests
│       ├── test_vision_architectures.py   # Basic vision architectures
│       ├── test_all_vision_universe.py    # Complete 25+ vision model suite
│       ├── test_nlp_tokenizers.py         # BPE, WordPiece, SentencePiece tests
│       ├── test_nlp_transformers.py       # MultiHeadAttention, GPT, BERT, LLaMA tests
│       ├── test_generative_models.py      # VAE, GAN, DDPM diffusion tests
│       ├── test_audio_and_timeseries.py   # Audio STFT, MFCC, ARIMA, DLinear tests
│       ├── test_gnn_and_rl.py             # GCN, GAT, DQN tests
│       ├── test_cleaning.py               # Data cleaning & imputers
│       ├── test_preprocessing.py          # Scalers & encoders
│       ├── test_transformation.py         # Space projections & resampling
│       ├── test_models.py                 # Classical ML models
│       ├── test_evaluation.py             # Evaluation metrics
│       └── test_explainability.py         # SHAP, LIME, PDP explainers
│
├── pyproject.toml                         # Modern PEP 518/621 Build System Configuration
├── setup.py                               # Setuptools Packaging Script
├── requirements.txt                       # Core Minimal Runtime Dependencies
├── requirements_dev.txt                   # Developer & CI Dependencies
└── README.md                              # Professional Documentation with ASCII Art & Badges
```

---

## 3. Phase-by-Phase Detailed Implementation Roadmap

### Phase 1: Foundation, Build System & Multi-OS CI/CD
- **Modern Packaging**: Zero build-time compilation, PEP 518/621 compliance (`pyproject.toml`, `setup.py`), `< 5 MB` wheel target.
- **CI/CD Matrix**: Automated GitHub Actions testing across Ubuntu and Windows for Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- **Code Standards**: 100% Black formatting, 0 Flake8 errors, 100% Mypy type-checked.
- **Commit Protocol**: Natural, humanized commit messages matching real junior developer habits.

### Phase 2: Multi-Modal Data Ingestion (`io/`)
- Pure NumPy, Pandas, Pillow, OpenCV-headless ingestion for Tabular, Image folders, Raw text, and Audio WAV files.
- Thread-safe, memory-mapped batch streaming.

### Phase 3: Deep Data Cleaning & Imputation (`cleaning/`)
- Imputers: Mean, Median, Mode, KNN Imputer, Iterative MICE Imputer, Constant.
- Outliers: IQR Filter, Z-Score Filter, Isolation Forest, Elliptic Envelope.
- Duplicates & Dtypes: Exact & fuzzy duplicate pruning, intelligent datetime & numerical coercion.

### Phase 4: Feature Preprocessing & Scaling (`preprocessing/`)
- Scalers: StandardScaler, MinMaxScaler, RobustScaler, PowerScaler (Yeo-Johnson), QuantileScaler.
- Encoders: OneHotEncoder, OrdinalEncoder, BinaryEncoder, TargetEncoder, FrequencyEncoder, HashEncoder.
- Feature Selectors: VarianceThreshold, CorrelationFilter, MutualInfoSelector, ANOVA F-Selector, RFESelector.

### Phase 5: Space Projections, Resampling & Augmentations (`transformation/`)
- Space Projections: PCA, SVD, LDA, t-SNE, KernelPCA.
- Resampling: SMOTE, ADASYN, Tomek Links, SMOTETomek, RandomUnderSampler.
- Augmentation Suites:
  - Vision: Flip, Rotate, Crop, Blur, Cutout, MixUp, ColorJitter.
  - Text: Synonym Replacement, Random Swap, Random Deletion.
  - Audio: Time Stretch, Pitch Shift, Additive Gaussian Noise, SpecAugment.

### Phase 6: Leakage-Free Splitting & Stratification (`splitting/`)
- Train/Test Split, 3-Way Train/Val/Test Split.
- KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold.

### Phase 7: Classical Machine Learning & Ensembles (`models/ml/`)
- Linear & Logistic: OLS, Ridge, Lasso, ElasticNet, Binary & Multinomial Logistic Regression.
- Trees & Forests: CART Decision Tree (Gini/Entropy/MSE/MAE), Random Forest, Extra Trees.
- Boosting: Gradient Tree Boosting, AdaBoost.
- Support Vector Machines: Linear, Polynomial, RBF, Sigmoid Kernels with SMO/QP solvers.
- Neighbors & Naive Bayes: KNN (Classification & Regression), Gaussian, Multinomial, Bernoulli, Complement NB.
- Clustering: K-Means++, K-Medoids (PAM), DBSCAN, OPTICS, Agglomerative Hierarchical (Ward/Single/Complete/Average), GMM.
- Anomaly Detection: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.

### Phase 8: Sovereign Autograd Deep Learning Framework (`models/dl/`)
- **Autograd Engine**: Reverse-mode automatic differentiation with dynamic DAG generation and topological backpropagation.
- **Layers**: Linear, Dropout, SpatialDropout, BatchNorm1d, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, Flatten, Embedding.
- **Activations**: ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU/Swish, Mish, Softmax, LogSoftmax, Sigmoid, HardSigmoid, Tanh.
- **Losses**: MSE, MAE, Huber, SmoothL1, CrossEntropy (log-sum-exp stabilized), BinaryCrossEntropy, FocalLoss, DiceLoss, TripletMarginLoss.
- **Optimizers**: SGD (Momentum, Nesterov, L2 decay), Adam, AdamW (decoupled weight decay), RMSProp, Adagrad, Lion.
- **Schedulers & Callbacks**: StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint.
- **Sequential Container**: `.add()`, `.fit()`, `.predict()`, `.predict_proba()`, `.save()`, `.load()`.

### Phase 9: Complete Computer Vision Architecture Universe (`models/vision/`)
- **Convolutions**: Vectorized `im2col`/`col2im` GEMM Conv2D, ConvTranspose2D, DepthwiseSeparableConv2D, GroupedConv2D, ChannelShuffle, MaxPool2D, AvgPool2D, GlobalAvgPool2D, BatchNorm2D.
- **Attention Modules**: Squeeze-and-Excitation (`SEBlock`), Convolutional Block Attention Module (`CBAM`).
- **Architectures**:
  - Pioneers: `LeNet5` (1998), `AlexNet` (2012), `ZFNet` (2013), `VGG11`, `VGG13`, `VGG16`, `VGG19` (2014).
  - Multi-Branch: `GoogLeNet` / `InceptionV1` (2014), `InceptionV3` (2015).
  - Residual & Dense: `ResNet18`, `ResNet34`, `ResNet50`, `ResNet101`, `ResNet152` (2015), `ResNeXt50`, `ResNeXt101` (2017), `DenseNet121`, `DenseNet169`, `DenseNet201` (2017), `SqueezeNet` (2016).
  - Edge & Mobile: `MobileNetV1` (2017), `MobileNetV2` (2018), `MobileNetV3` (2019), `ShuffleNetV1`, `ShuffleNetV2` (2018).
  - Modern Pure CNNs: `EfficientNetB0` (2019), `ConvNeXtTiny`, `ConvNeXt` (2022).
  - Segmentation: `UNet` (2015), `FCN8s`, `FCN32s` (2015), `SegNet` (2017), `FPN` (2017).
  - Vision Transformers: `VisionTransformer` (`ViT`), `SwinTransformer` (2020-2021).
- **Detection & XAI**: Anchor Box generation, IoU, Non-Maximum Suppression (`NMS`), YOLO Grid Head, `GradCAM`, `GradCAM++`.

### Phase 10: NLP, Sequence Models & Sovereign Transformers (`models/nlp/`)
- **Tokenizers**: Byte-Pair Encoding (`BPE`), `WordPiece`, `SentencePiece`, Character-level.
- **Embeddings**: TokenEmbedding, Sinusoidal Positional Encoding, Learned 1D Positional Embedding, Rotary Positional Embedding (`RoPE`), `ALiBi`.
- **Recurrent Models**: Vanilla `RNN`, `LSTM` (with Forget, Input, Output, Cell state calculus), `BiLSTM`, `GRU` (Reset, Update gates).
- **Attention Subsystem**: Scaled Dot-Product Attention, Multi-Head Attention (`MHA`), Multi-Query Attention (`MQA`), Grouped-Query Attention (`GQA`), Tiled Attention.
- **Transformer Blocks**: Pre-LN & Post-LN `TransformerEncoderBlock`, `TransformerDecoderBlock` (with cross-attention & causal masking).
- **Architectures**:
  - `BERT` / `RoBERTa`: Bidirectional Transformer with MLM & Sequence Classification heads.
  - `MiniGPT` / `GPT-2`: Autoregressive Causal Decoder with KV-Caching, Temperature, Top-K, Top-P Nucleus, and Beam Search.
  - `LLaMA` / `Mistral`: Modern LLM architecture featuring RMSNorm, SwiGLU activation, RoPE positional embeddings, and GQA.
  - `Seq2SeqTransformer` / `T5`: Encoder-Decoder Translation and Summarization model.

### Phase 11: Audio, Speech & Time Series Signal Processing (`models/audio/`, `models/timeseries/`)
- Audio: Short-Time Fourier Transform (`STFT`), Mel-Filterbanks, Mel-Spectrogram, MFCC extraction, 1D Convolutions, `WaveNet` from scratch.
- Time Series: Seasonal-Trend Decomposition, `ARIMA` / `SARIMA` pure NumPy solver, Holt-Winters Exponential Smoothing, `DLinear`, `N-BEATS` deep forecaster.

### Phase 12: Generative Models & Deep Generative AI (`models/generative/`)
- Autoencoders: Vanilla `Autoencoder`, `DenoisingAutoencoder`.
- Variational Autoencoders: `VAE` with Reparameterization trick & KL Divergence loss.
- GANs: Vanilla `GAN`, Deep Convolutional GAN (`DCGAN`).
- Diffusion Models: Denoising Diffusion Probabilistic Models (`DDPM`) from scratch (Forward Gaussian noise scheduler $eta_t$, Reverse sampling denoising loop).

### Phase 13: Graph Neural Networks (`models/gnn/`)
- Graph representation, Normalized Graph Laplacian $	ilde{D}^{-1/2} 	ilde{A} 	ilde{D}^{-1/2}$.
- Graph Convolutional Network (`GCN`) Layer: $H^{(l+1)} = \sigma(	ilde{D}^{-1/2} 	ilde{A} 	ilde{D}^{-1/2} H^{(l)} W^{(l)})$.
- Graph Attention Network (`GAT`) Layer: Self-attention over graph adjacency neighborhoods.

### Phase 14: Model Evaluation & Diagnostics (`evaluation/`)
- Classification: Accuracy, Precision, Recall, F1, F-beta, ROC-AUC, PR-AUC, LogLoss, MCC, Cohen's Kappa, Confusion Matrix.
- Regression: MSE, RMSE, MAE, R2, Adjusted R2, MAPE, SMAPE, Explained Variance.
- Clustering: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
- Data Drift: Population Stability Index (`PSI`), Kolmogorov-Smirnov (`KS-Test`), Wasserstein Distance.

### Phase 15: Sovereign Explainable AI (`explainability/`)
- Model-agnostic Permutation Feature Importance.
- Sovereign `KernelSHAP` with coalition sampling & Shapley values.
- Sovereign `LIME` local linear surrogate model with exponential kernel distance weighting.
- Partial Dependence Plots (`PDP`) & Individual Conditional Expectation (`ICE`).
- `IntegratedGradients` axiomatic path integral attributions.

### Phase 16: Universal Pipeline Engine & Autonomous AutoML (`pipeline/`)
- ChokkhuPipeline: Leakage-free sequential chaining of Cleaning -> Preprocessing -> Transformation -> Modeling.
- State Serialization: PreprocessorState & TransformationState for zero-leakage production inference.
- Autonomous AutoML: 1-line automated task inference, dataset profiling, multi-model candidate tournament, hyperparameter optimization, and ensemble stacking/blending.
- Interactive Reporting: Self-contained, responsive dark-mode HTML dashboard generation with embedded charts.

### Phase 17: Production Packaging, Wheels & Multi-OS Verification
- Wheel build verification (`python -m build`).
- Target wheel size: `< 5 MB`.
- Zero heavy dependencies runtime check.
- Multi-OS CI/CD verification on Ubuntu, Windows, macOS across Python 3.9, 3.10, 3.11, 3.12, 3.13 with 100% test pass rate.
