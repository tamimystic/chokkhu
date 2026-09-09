# CHOKKHU (চক্ষু) — The Sovereign All-in-One AI & Machine Learning Pipeline Framework
## The Ultimate Ultra-Pro-Max Architectural Master Blueprint
### "Minimalistic Code. Maximum Output. Zero Heavy Dependencies. Infinite Intelligence."

---

## 1. Executive Vision & Sovereign Philosophy

### 1.1 The Mission of Chokkhu
Modern AI and Data Science ecosystems are crippled by **dependency bloat**, **fragile binary environments**, **complex boilerplate**, and **hidden data leakage**. A practitioner building an end-to-end intelligent system is forced to juggle gigabytes of external frameworks:
`torch`, `torchvision`, `tensorflow`, `keras`, `scikit-learn`, `transformers`, `huggingface_hub`, `tokenizers`, `diffusers`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `shap`, `lime`, `statsmodels`, `gym`.

**Chokkhu (চক্ষু) is built to be the single sovereign Python package that replaces them all with mathematical purity.**

Every single algorithm—from basic data imputation to classical tree ensembles, Convolutional Neural Networks (CNNs), Vision Transformers (ViT), Auto-Regressive Large Language Models (Mini-GPT / LLaMA / Mistral / Gemma / DeepSeek-style), Denoising Diffusion Probabilistic Models (DDPM / Flow Matching), Audio Transformers (Conformer / Whisper / AST / Wav2Vec2), Graph Neural Networks (GCN / GAT / Graphormer), Deep Time Series Forecasters (N-BEATS / PatchTST / TimesNet), Explainable AI (SHAP / LIME / Integrated Gradients / DLA), and Deep Reinforcement Learning (DQN / PPO / SAC)—is engineered **100% from scratch using pure NumPy, SciPy, Pandas, OpenCV-headless, Matplotlib, and Seaborn.**

```
                                 CHOKKHU ECOSYSTEM
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 UNIFIED 1-LINE API                                     │
│  chokkhu.load() | clean() | preprocess() | transform() | split() | train() | evaluate()│
│                        chokkhu.pipeline() | explain() | auto_train()                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
        │                 │                  │                 │                 │
        ▼                 ▼                  ▼                 ▼                 ▼
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │ Tabular ML  │   │  Deep Tensor│   │  Computer   │   │     NLP &   │   │ Generative  │
 │ & Ensembles │   │  & Autograd │   │   Vision    │   │Transformers │   │     AI      │
 │ (RF, GBM,   │   │ (Graph, SGD,│   │ (Conv2D,    │   │ (BPE, RoPE, │   │ (VAE, DDPM, │
 │  SVM, KNN)  │   │  Adam, Loss)│   │  ResNet,ViT)│   │  GPT, Mamba)│   │  Flux, GAN) │
 └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
        │                 │                  │                 │                 │
        ├─────────────────┴──────────────────┼─────────────────┴─────────────────┤
        ▼                                    ▼                                   ▼
 ┌─────────────┐                      ┌─────────────┐                     ┌─────────────┐
 │ Audio/Speech│                      │ Graph Neural│                     │ Time Series │
 │(STFT, Confo-│                      │  Networks   │                     │ & Deep SOTA │
 │ rmer,Whisper│                      │ (GCN, GAT,  │                     │ (N-BEATS,   │
 │   Codec)    │                      │ Graphormer) │                     │ PatchTST)   │
 └─────────────┘                      └─────────────┘                     └─────────────┘
        │                                    │                                   │
        └─────────────────┬──────────────────┴─────────────────┬─────────────────┘
                          │                                    │
                          ▼                                    ▼
               ┌───────────────────────┐            ┌───────────────────────┐
               │    Sovereign XAI &    │            │   Autonomous AutoML   │
               │   Mechanistic Interp  │            │  & Stacking Ensembles │
               │ (SHAP, DLA, Gradient) │            │ (Bayesian, BOHB, DFS) │
               └───────────────────────┘            └───────────────────────┘
                                         │
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
   - Every model and pipeline stage can be executed with a single intuitive function call (e.g. `ck.train("resnet50", ...)`, `ck.train("gpt", ...)`, `ck.train("random_forest", ...)`, `ck.pipeline(...)`, `ck.auto_train(...)`).
3. **Guaranteed Zero Data Leakage**:
   - Automated state preservation (`PreprocessorState`, `TransformationState`) ensuring strict statistical isolation between training folds and test/inference data.
4. **Featherweight Package Size (< 5 MB Wheel)**:
   - Ultra-fast download, zero C-compiler bottlenecks, instant CI/CD installation, and minimal runtime memory footprint.
5. **Research-Grade Mathematical Rigor**:
   - Exact backpropagation calculus, vectorized matrix operations (e.g., `im2col` GEMM for convolutions, scaled dot-product attention, selective state-space scan), and numerical stability safeguards ($\epsilon$ clipping, log-sum-exp tricks).
6. **Robust Multi-OS & Multi-Python Compatibility Matrix**:
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
│       ├── models/                        # Phase 7 to 13 + RL: The Complete Modeling Universe
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
│       │   │   ├── detection.py           # Anchor Boxes, IoU, Non-Maximum Suppression (NMS), YOLO-Head, SSD, RetinaNet
│       │   │   ├── fpn.py                 # Feature Pyramid Networks (FPN), PANet
│       │   │   ├── losses.py              # FocalLoss, DiceLoss, TverskyLoss, GIoU, DIoU, CIoU
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
│       │   │       ├── vit.py             # Vision Transformer (ViT-Tiny, ViT-Base) (2020)
│       │   │       ├── swin.py            # Swin Transformer (Shifted Window Attention) (2021)
│       │   │       └── deit.py            # DeiT (Distillation Token Vision Transformer) (2021)
│       │   │
│       │   ├── nlp/                       # Phase 10: NLP, Sequence Models & Ultra-Modern LLMs
│       │   │   ├── __init__.py
│       │   │   ├── tokenizers/            # Pure NumPy Tokenization Subsystem
│       │   │   │   ├── __init__.py
│       │   │   │   ├── bpe.py             # Byte-Pair Encoding & Byte-Level Regex (Tiktoken style)
│       │   │   │   ├── wordpiece.py       # WordPiece Tokenizer (BERT-style)
│       │   │   │   ├── sentencepiece.py   # Unigram / SentencePiece Tokenizer with EM Viterbi
│       │   │   │   └── character.py       # Character & Word-level fallbacks
│       │   │   ├── embeddings.py          # TokenEmbedding, Sinusoidal, Learned, RoPE, YaRN, ALiBi
│       │   │   ├── recurrent.py           # Vanilla RNN, LSTM (full gate calculus), BiLSTM, GRU
│       │   │   ├── attention.py           # Scaled Dot-Product, MHA, GQA, SWA, DeepSeek MLA
│       │   │   ├── linear_attention.py    # RWKV (Receptance-Weighted Key-Value), RetNet (Multi-Scale Retention)
│       │   │   ├── state_space.py         # Mamba / S6 Selective State Space Scan Model
│       │   │   ├── transformer_blocks.py  # Pre-LN / Post-LN Blocks, SwiGLU, RMSNorm
│       │   │   ├── moe.py                 # Mixture of Experts (TopKRouter, Auxiliary Load Balance)
│       │   │   ├── generation.py          # Greedy, Temperature, Top-K, Top-P, Speculative Decoding
│       │   │   ├── alignment.py           # Direct Preference Optimization (DPO), KTO, ORPO
│       │   │   └── architectures/         # Sovereign Language Model Architectures
│       │   │       ├── __init__.py
│       │   │       ├── bert.py            # BERT (Masked LM & Sequence Classification)
│       │   │       ├── roberta.py         # RoBERTa Optimized Pretraining Architecture
│       │   │       ├── deberta.py         # DeBERTa (3-term Disentangled Self-Attention)
│       │   │       ├── mini_gpt.py        # GPT-1/2 Autoregressive Causal LLM with KV-Caching
│       │   │       ├── llama.py           # LLaMA-2/3 (RMSNorm, SwiGLU, RoPE, GQA)
│       │   │       ├── mistral.py         # Mistral (Sliding Window Attention + GQA)
│       │   │       ├── gemma.py           # Gemma (GeGLU + RMSNorm)
│       │   │       ├── deepseek.py        # DeepSeek-V3 / R1 (Fine-grained MoE + Multi-Token Prediction)
│       │   │       ├── qwen.py            # Qwen 2.5 (Dual-chunk Attention + SwiGLU scaling)
│       │   │       └── seq2seq.py         # T5 / Seq2Seq Encoder-Decoder Translation Transformer
│       │   │
│       │   ├── audio/                     # Phase 11: Audio, Speech & Multi-Modal Signal Processing
│       │   │   ├── __init__.py
│       │   │   ├── transforms.py          # STFT, ISTFT, Mel-Spectrogram, MFCC, Deltas, CQT, Chroma
│       │   │   ├── conv1d.py              # Temporal 1D Convolution & DepthwiseConv1D
│       │   │   ├── codecs.py              # Residual Vector Quantization (RVQ) Neural Audio Codec
│       │   │   └── architectures/         # Sovereign Speech & Audio Architectures
│       │   │       ├── __init__.py
│       │   │       ├── conformer.py       # Conformer (Macaron FFN + MHA + ConvModule + GLU)
│       │   │       ├── ast.py             # Audio Spectrogram Transformer (AST)
│       │   │       ├── wav2vec2.py        # Wav2Vec2 Self-Supervised Speech Representation
│       │   │       ├── whisper.py         # Whisper Multi-Task Speech Recognition & Translation
│       │   │       ├── vits.py            # VITS Conditional VAE + GAN Text-to-Speech
│       │   │       └── hubert.py          # HuBERT Self-Supervised Acoustic Modeling
│       │   │
│       │   ├── generative/                # Phase 12: Generative Models, Latent Diffusion & Flow Matching
│       │   │   ├── __init__.py
│       │   │   ├── autoencoders.py        # VAE (Reparameterization Trick + KL), VQ-VAE (Straight-Through)
│       │   │   ├── gan.py                 # DCGAN, WGAN-GP (Wasserstein with Gradient Penalty), StyleGAN2
│       │   │   ├── diffusion.py           # DDPM (Sinusoidal time embeddings, Reverse Sampling), DDIM fast sampler
│       │   │   ├── flow_matching.py       # Rectified Flow Matching (Flux / SD3 continuous ODE solver)
│       │   │   ├── latent_diffusion.py    # Latent Diffusion Model (LDM / Stable Diffusion style with Cross-Attention)
│       │   │   ├── normalizing_flows.py   # RealNVP & GLOW Invertible Affine Coupling Flows
│       │   │   └── adapters.py            # LoRA (Low-Rank Adaptation), ControlNet spatial conditioning
│       │   │
│       │   ├── gnn/                       # Phase 13: Graph Neural Networks & Geometric Deep Learning
│       │   │   ├── __init__.py
│       │   │   ├── layers.py              # GCNLayer, GATLayer (Multi-Head), GraphSAGELayer, GINLayer
│       │   │   ├── utils.py               # Normalized Laplacian, Adjacency conversions, Global Pooling
│       │   │   ├── graph_transformer.py   # Graphormer & Graph Transformer with Laplacian Eigenvector positional bias
│       │   │   ├── relational.py          # Relational GCN (R-GCN for heterogeneous multi-relational graphs)
│       │   │   ├── equivariant.py         # EGNN / SchNet Equivariant GNNs for 3D coordinates & molecules
│       │   │   ├── hypergraph.py          # Hypergraph Neural Networks (HGNN for N-ary relationships)
│       │   │   └── architectures.py       # GCN, GAT, GraphSAGE, GIN, DiffPool Hierarchical Classifiers
│       │   │
│       │   ├── timeseries/                # Phase 14: Time Series, Statistical & Deep Forecasting
│       │   │   ├── __init__.py
│       │   │   ├── statistical.py         # ARIMA, SARIMAX, Holt-Winters Exponential Smoothing, Auto-ARIMA
│       │   │   ├── transforms.py          # Seasonal-Trend Decomposition, Lag Matrices, Differencing, DTW
│       │   │   └── neural.py              # N-BEATS, N-HiTS, PatchTST, TimesNet, TiDE, Informer, DeepAR
│       │   │
│       │   └── rl/                        # Bonus Phase: Sovereign Reinforcement Learning Universe
│       │       ├── __init__.py
│       │       ├── environments.py        # Custom GridWorld, CartPole, Multi-Armed Bandit Envs
│       │       ├── q_learning.py          # Tabular Q-Learning with Epsilon-Greedy
│       │       ├── dqn.py                 # DQN, Double DQN, Dueling DQN, Prioritized Experience Replay (PER)
│       │       ├── policy_gradient.py     # REINFORCE with Baseline, Advantage Actor-Critic (A2C)
│       │       ├── ppo.py                 # Proximal Policy Optimization (PPO with clipped surrogate objective)
│       │       ├── sac.py                 # Soft Actor-Critic (SAC with maximum entropy objective)
│       │       └── bandits.py             # UCB1, Thompson Sampling (Beta/Gaussian), LinUCB Contextual Bandits
│       │
│       ├── evaluation/                    # Phase 15: Metrics, Diagnostics, Calibration & Drift
│       │   ├── __init__.py
│       │   ├── metrics.py                 # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, LogLoss, MCC, Kappa, RMSE, MAE, R2, sMAPE, MASE, NDCG@K, MAP@K
│       │   ├── calibration.py             # Expected Calibration Error (ECE), Reliability Curves, Platt Scaling, Isotonic Regression
│       │   ├── drift_detector.py          # Population Stability Index (PSI), Kolmogorov-Smirnov (KS), MMD, ADWIN
│       │   └── engine.py                  # Universal chokkhu.evaluate() dispatcher & report generator
│       │
│       ├── explainability/                # Phase 16: Sovereign Explainable AI (XAI) & Mechanistic Interpretability
│       │   ├── __init__.py
│       │   ├── importance.py              # Permutation Feature Importance
│       │   ├── shap.py                    # Sovereign KernelSHAP & Exact TreeSHAP
│       │   ├── lime.py                    # Sovereign LIME (Local Interpretable Surrogate with Exponential Weighting)
│       │   ├── pdp.py                     # Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE)
│       │   ├── counterfactuals.py         # Wachter Counterfactual Explanation Optimizer
│       │   ├── neural.py                  # Integrated Gradients, SmoothGrad, DeepLIFT
│       │   ├── mechanistic.py             # Transformer Attention Rollout, Direct Logit Attribution (DLA), Causal Tracing
│       │   └── engine.py                  # Universal chokkhu.explain() dispatcher
│       │
│       ├── eda/                           # Exploratory Data Analysis & Automated Profiling
│       │   ├── __init__.py
│       │   ├── tabular/                   # Univariate, Bivariate, Multivariate Stats & Plotters
│       │   ├── image/                     # Metadata, Color, Texture, Duplicates, Quality Engines
│       │   └── engine.py                  # Standalone Dark-Mode HTML Dashboard Generator
│       │
│       └── pipeline/                      # Phase 17: Autonomous AutoML, Stacking & Universal Pipelines
│           ├── __init__.py
│           ├── engine.py                  # Universal ck.pipeline() leak-free orchestrator
│           ├── state.py                   # PreprocessorState & TransformationState serializers
│           ├── stacking.py                # Multi-Layer Super Learner Stacking & Nelder-Mead Weight Blending
│           ├── feature_synthesis.py       # Deep Feature Synthesis (DFS automated cross-aggregations & ratios)
│           └── automl/                    # Autonomous Tournament, Bayesian Optimization, Hyperband, BOHB
│               ├── __init__.py
│               ├── autotrainer.py         # Autonomous Dataset Profiling & Candidate Tournament
│               ├── bayesian.py            # Gaussian Process Surrogate with EI and UCB Acquisitions
│               ├── hyperband.py           # Successive Halving Bandit Tuner
│               └── bohb.py                # Combined Bayesian Optimization + Hyperband (BOHB)
│
├── tests/                                 # Comprehensive Multi-Platform Test Suite (184+ Tests)
│   ├── integration/
│   │   ├── test_end_to_end_complete.py    # Full lifecycle tests
│   │   └── test_int.py                    # Pipeline integration tests
│   └── unit/
│       ├── test_algorithmic_fixes.py
│       ├── test_all_vision_universe.py
│       ├── test_api_and_tuning.py
│       ├── test_audio.py
│       ├── test_autograd_tensor.py
│       ├── test_automl.py
│       ├── test_cleaning.py
│       ├── test_cli.py
│       ├── test_conv_layers.py
│       ├── test_evaluation.py
│       ├── test_explainability.py
│       ├── test_generative.py
│       ├── test_gnn.py
│       ├── test_io.py
│       ├── test_metrics_extended.py
│       ├── test_models.py
│       ├── test_neural_network.py
│       ├── test_neural_xai.py
│       ├── test_nlp_classical.py
│       ├── test_nlp_modern_llms.py
│       ├── test_nlp_moe_mla.py
│       ├── test_nlp_transformers.py
│       ├── test_pipeline.py
│       ├── test_preprocessing.py
│       ├── test_rl.py
│       ├── test_scalers_selectors.py
│       ├── test_splitting.py
│       ├── test_timeseries.py
│       ├── test_transformation.py
│       ├── test_unit.py
│       ├── test_vision_architectures.py
│       ├── test_vision_detection_models.py
│       ├── test_vision_detection_xai.py
│       ├── test_vision_fpn_conv.py
│       ├── test_vision_losses.py
│       └── test_vision_swin_deit.py
│
├── pyproject.toml                         # Modern PEP 518/621 Build System Configuration
├── setup.py                               # Setuptools Packaging Script
├── setup.cfg                              # Linter & Type-Checking Configuration
├── requirements.txt                       # Core Minimal Runtime Dependencies
├── requirements_dev.txt                   # Developer & CI Dependencies
└── README.md                              # Professional Documentation with ASCII Art & Badges
```

---

## 3. Phase-by-Phase Detailed Implementation Roadmap

### Phase 1: Foundation, Build System & Multi-OS CI/CD
- **Modern Packaging**: Zero build-time compilation, PEP 518/621 compliance (`pyproject.toml`, `setup.py`), `< 5 MB` wheel target.
- **CI/CD Matrix**: Automated GitHub Actions testing across Ubuntu, Windows, and macOS for Python 3.9, 3.10, 3.11, 3.12, and 3.13 (17 active jobs, 100% Green).
- **Code Standards**: 100% Black formatting, 0 Flake8 errors, 100% Mypy type-checked.

### Phase 2: Multi-Modal Data Ingestion (`io/`)
- Pure NumPy, Pandas, Pillow, OpenCV-headless ingestion for Tabular (CSV, TSV, JSON), Image folders, Raw text, and Audio WAV files.
- Thread-safe, memory-mapped batch streaming and atomic pipeline serialization.

### Phase 3: Deep Data Cleaning & Imputation (`cleaning/`)
- Imputers: Mean, Median, Mode, KNN Imputer, Iterative MICE Imputer, Constant.
- Outliers: IQR Filter, Z-Score Filter, Isolation Forest, Elliptic Envelope.
- Duplicates & Dtypes: Exact & subset duplicate pruning, intelligent datetime & numerical coercion.

### Phase 4: Feature Preprocessing & Scaling (`preprocessing/`)
- Scalers: StandardScaler, MinMaxScaler, RobustScaler, PowerScaler (Yeo-Johnson/Box-Cox), QuantileScaler (Uniform/Normal).
- Encoders: OneHotEncoder, OrdinalEncoder, BinaryEncoder, TargetEncoder, FrequencyEncoder, HashEncoder.
- Feature Selectors: VarianceThreshold, CorrelationFilter, MutualInfoSelector, ANOVA F-Selector, RFESelector.

### Phase 5: Space Projections, Resampling & Augmentations (`transformation/`)
- Space Projections: PCA, SVD, LDA, t-SNE, KernelPCA.
- Resampling: SMOTE, ADASYN, Tomek Links, SMOTETomek, RandomOverSampler, RandomUnderSampler.
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
- **Layers**: Linear, Dropout, SpatialDropout, BatchNorm1D, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, Flatten, Embedding.
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
  - Segmentation: `UNet` (2015), `FCN8s`, `FCN32s` (2015).
  - Vision Transformers: `VisionTransformer` (`ViT`), `SwinTransformer` (2021), `DeiT` (2021).
- **Detection & XAI**: Anchor Box generation, IoU, Non-Maximum Suppression (`NMS`), YOLO Grid Head, SSD Head, RetinaNet Head, Feature Pyramid Network (`FPN`), `PANet`, `GradCAM`.

---

### Phase 10: NLP, Sequence Models & Ultra-Modern Frontier LLMs (`models/nlp/`)
- **Tokenizers**:
  - `BPETokenizer`: Byte-Pair Encoding with Byte-level regex splitting (GPT-4 / Tiktoken style regex).
  - `WordPieceTokenizer`: WordPiece tokenization with `##` subword continuation prefixes.
  - `SentencePieceTokenizer`: Unigram Language Model tokenizer with EM parameter estimation & Viterbi optimal segmentation.
  - `CharacterTokenizer` & `WordTokenizer`: Lightweight fallbacks.
- **Classical NLP & Information Retrieval**:
  - `PorterStemmer`: Complete 5-step algorithmic suffix-stripping.
  - `TfidfVectorizer`: Sublinear TF scaling, custom N-gram ranges, L2 normalization.
  - `BM25Retriever`: Okapi BM25+ ranking function with document length normalization ($k_1=1.5, b=0.75, \delta=1.0$).
- **Embeddings & Long-Context Encodings**:
  - `TokenEmbedding`, `SinusoidalPositionalEncoding`, `LearnedPositionalEmbedding`.
  - `RoPE` (Rotary Positional Embeddings): Complex space 2D block rotations $R_{\Theta, m}^d$.
  - `YaRN` (Yet another RoPE extensioN) & `Dynamic NTK-aware RoPE`: Dynamic context expansion up to 128k tokens.
  - `ALiBi` (Attention with Linear Biases): Distance penalty slope bias $m \cdot (i - j)$.
- **Recurrent & Linear Attention**:
  - `RNN`, `LSTM` (full gate equations), `BiLSTM`, `GRU`.
  - `RWKV` (Receptance Weighted Key Value): Linear attention RNN with Transformer-level parallelism.
  - `RetNet` (Retentive Network): Multi-Scale Retention mechanism.
  - `Mamba / S6 Selective State Space Model`: Discrete-time parameter-dependent scan $h_t = ar{\mathbf{A}}_t h_{t-1} + ar{\mathbf{B}}_t x_t$, $y_t = \mathbf{C}_t h_t$.
- **Attention Modules**:
  - Scaled Dot-Product Attention, Multi-Head Attention (`MHA`), Multi-Query Attention (`MQA`), Grouped-Query Attention (`GQA`).
  - `SlidingWindowAttention` (SWA): Local attention window with rolling KV-cache.
  - `DeepSeekMLA` (Multi-Head Latent Attention): Low-rank KV compression & decoupled RoPE queries.
- **Modern Transformer LLM Architectures**:
  - `BERT` & `RoBERTa`: Bidirectional Transformer with MLM & Sequence Classification heads.
  - `DeBERTa`: 3-term Disentangled Self-Attention (Content-to-Content, Content-to-Position, Position-to-Content).
  - `MiniGPT` / `GPT-2`: Autoregressive Causal Decoder with KV-Caching.
  - `LLaMA-2/3`: RMSNorm, SwiGLU, RoPE, Grouped-Query Attention.
  - `Mistral`: Sliding Window Attention + GQA.
  - `Gemma`: GeGLU activation + RMSNorm scaling.
  - `DeepSeek-V3 / DeepSeek-R1`: Fine-Grained MoE with Auxiliary-Loss-Free load balancing + Multi-Token Prediction (MTP).
  - `Qwen 2.5`: Dual-chunk attention and scaled SwiGLU intermediate dimensions.
  - `Seq2SeqTransformer` / `T5`: Encoder-Decoder Translation and Summarization model.
- **Generation & LLM Alignment**:
  - `TextGenerator`: Greedy, Temperature scaling, Top-K, Top-P Nucleus sampling, Beam Search with N-gram repetition penalty.
  - `Speculative Decoding`: Small draft model speculative token generation with parallel main-model verification (2-3x speedup).
  - `Direct Preference Optimization` (DPO): Direct policy parameter alignment without auxiliary reward models.
  - `ORPO` (Odds Ratio Preference Optimization) & `KTO` (Kahneman-Tversky Optimization).

---

### Phase 11: Audio, Speech & Multi-Modal Signal Processing (`models/audio/`)
- **Signal Transformations & Spectral Analysis**:
  - `STFT` & `ISTFT`: Short-Time Fourier Transform and overlap-add reconstruction with perfect window normalization.
  - `Mel-Spectrogram` & `Mel-Filterbanks`: Triangular Mel-scale filterbanks with Slaney/HTK formulas.
  - `MFCC`: Mel-Frequency Cepstral Coefficients with Type-II DCT and delta/delta-delta derivatives.
  - `Chroma STFT`: 12-semitone pitch class chroma feature extraction.
  - Spectral Descriptors: Spectral Centroid, Spectral Rolloff, Spectral Contrast, Zero Crossing Rate.
  - `Constant-Q Transform (CQT)` & `Gammatone Filterbanks`.
- **Temporal Convolutions & Augmentations**:
  - `Conv1D`, `DepthwiseConv1D`, `BatchNorm1D`.
  - `SpecAugment`: Frequency masking, time masking, and time warping.
  - Room Impulse Response (RIR) acoustic convolution and background noise mixing.
- **Neural Audio Codecs & Compression**:
  - `Residual Vector Quantizer (RVQ)`: Multi-stage hierarchical codebook quantization (EnCodec / SoundStream style).
- **Sovereign Speech & Audio Architectures**:
  - `Conformer`: Macaron-style Feed-Forward Modules, Multi-Head Attention, Depthwise ConvModule, and Gated Linear Units.
  - `Audio Spectrogram Transformer (AST)`: Pure patch-based 2D Vision Transformer for audio spectrograms.
  - `Wav2Vec2`: Self-supervised raw audio convolutional encoder + Transformer contextualizer with Gumbel-Softmax quantization.
  - `Whisper (OpenAI)`: Multilingual multi-task speech recognition & translation encoder-decoder with log-mel spectrogram input and timestamp token decoding.
  - `VITS`: Conditional Variational Autoencoder with Adversarial Learning for end-to-end Text-to-Speech synthesis.
  - `HuBERT`: Hidden-Unit BERT with offline k-means cluster targets.
  - `Conv-TasNet` / `SepFormer`: Time-domain speech separation and noise reduction.

---

### Phase 12: Generative Models, Latent Diffusion & Flow Matching (`models/generative/`)
- **Autoencoders**:
  - `Autoencoder` & `DenoisingAutoencoder`.
  - `VAE` (Variational Autoencoder): Gaussian latent space with Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$) and analytical KL Divergence loss.
  - `VQ-VAE`: Vector-Quantized Variational Autoencoder with Straight-Through Estimator codebook dictionary learning.
- **Generative Adversarial Networks (GANs)**:
  - `DCGAN`: Deep Convolutional GAN with transpose convolutions.
  - `WGAN-GP`: Wasserstein GAN with exact Gradient Penalty constraint ($\mathbb{E}[(\|
abla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$).
  - `StyleGAN2 / StyleGAN-XL`: Mapping Network $f: \mathcal{Z} ightarrow \mathcal{W}$, Style Modulation/Demodulation, Noise injection, and Path Length Regularization.
  - `Pix2Pix` & `CycleGAN`: Paired and Unpaired Image-to-Image translation with PatchGAN discriminator and cycle consistency loss.
- **Diffusion Models & Deterministic Samplers**:
  - `DDPM` (Denoising Diffusion Probabilistic Models): Linear/Cosine $eta_t$ variance schedulers, sinusoidal time embeddings, reverse Gaussian denoising.
  - `DDIM` (Denoising Diffusion Implicit Models): Deterministic fast sampling loop achieving high fidelity in 15-20 steps.
  - `Classifier-Free Guidance (CFG)`: Conditional vs unconditional score interpolation $v_{	ext{guided}} = v_{	ext{uncond}} + s \cdot (v_{	ext{cond}} - v_{	ext{uncond}})$.
  - `DPM-Solver` & `Euler A` high-order ODE samplers.
- **Latent Diffusion & Flow Matching**:
  - `Latent Diffusion Models (LDM / Stable Diffusion style)`: VAE latent space diffusion with Cross-Attention text/context conditioning.
  - `Rectified Flow Matching (Flux / SD3)`: Optimal transport straight-line velocity field ODE solver ($rac{dx_t}{dt} = v_t(x_t)$).
  - `Normalizing Flows (RealNVP, GLOW)`: Invertible affine coupling layers with exact log-determinant Jacobian computation.
- **Adapters & Fine-Tuning**:
  - `LoRA (Low-Rank Adaptation)`: Parameter-efficient rank-$r$ adaptation $\Delta W = rac{lpha}{r} B \cdot A$ for all linear/convolutional layers.
  - `ControlNet`: Spatial condition injection via zero-initialized convolution layers.

---

### Phase 13: Graph Neural Networks & Geometric Deep Learning (`models/gnn/`)
- **Graph Foundations & Spectral Operations**:
  - Graph Adjacency, Degree Matrix, Symmetric Normalized Laplacian ($	ilde{D}^{-1/2} 	ilde{A} 	ilde{D}^{-1/2}$), Random-Walk Normalization ($	ilde{D}^{-1} 	ilde{A}$).
  - Dense $\leftrightarrow$ Sparse COO Edge Index conversion.
  - Global Graph Readout Pooling: Mean, Max, Sum, Attention-based readout.
- **Core GNN Layers**:
  - `GCNLayer` (Kipf & Welling): First-order localized spectral graph convolution.
  - `GATLayer` (Veličković et al.): Multi-Head Attention over graph neighborhoods with LeakyReLU self-attention coefficients.
  - `GraphSAGELayer` (Hamilton et al.): Neighborhood sampling with Mean, Max-Pooling, and Sum aggregators.
  - `GINLayer` (Graph Isomorphism Network): Maximally powerful Weisfeiler-Lehman graph isomorphism test layer with learnable $\epsilon$.
- **Advanced GNNs & Geometric Architectures**:
  - `Graph Transformer & Graphormer`: Laplacian Eigenvector positional encodings, shortest-path distance spatial bias, and degree centralities.
  - `Relational GCN (R-GCN)`: Multi-relational graph convolution for heterogeneous graphs and knowledge bases.
  - `Principal Neighbourhood Aggregation (PNA)`: Multi-aggregator (Mean, Max, Min, Std) and degree scaler combination.
  - `Equivariant GNNs (EGNN / SchNet)`: Rotation, translation, and reflection $E(n)$-equivariant networks for 3D coordinate and molecular modeling.
  - `Hypergraph Neural Networks (HGNN)`: Hyperedge incidence matrix operations for complex multi-node interactions.
- **Graph Tasks & Hierarchical Pooling**:
  - Node Classification & Graph Classification.
  - Link Prediction: Dot-product and MLP scoring with negative edge sampling.
  - `DiffPool`: Hierarchical differentiable graph clustering and pooling.
  - `Node2Vec` & `NeighborLoader` mini-batch sub-graph sampling.

---

### Phase 14: Time Series & Deep Forecasting Universe (`models/timeseries/`)
- **Statistical & Classical Econometric Models**:
  - `ARIMA(p, d, q)`: Pure NumPy exact least-squares and gradient-based solver.
  - `SARIMAX(p, d, q)x(P, D, Q)_s`: Seasonal ARIMA with exogenous covariate support.
  - `Auto-ARIMA`: AIC / BIC information criterion grid search for optimal $(p, d, q)$ orders.
  - `ExponentialSmoothing`: Simple Exponential Smoothing, Holt's Linear Trend, and Holt-Winters Additive/Multiplicative Seasonality.
  - `Vector Autoregression (VAR)`: Multivariate time series inter-dependency modeling.
  - `Seasonal-Trend Decomposition`: Classical additive/multiplicative moving average decomposition.
- **Time Series Feature Transformations & Metrics**:
  - Lag Matrices, Rolling Window Statistics, Differencing, Dynamic Time Warping (DTW) distance.
- **State-of-the-Art Deep Forecasters**:
  - `N-BEATS`: Doubly Residual generic and trend/seasonality polynomial expansion architecture.
  - `N-HiTS`: Hierarchical multi-rate sampling and multi-scale interpolation.
  - `PatchTST`: Subseries patching + Channel-Independent Transformer for long-horizon forecasting.
  - `TimesNet`: 2D-Variation representation transforming 1D time series into 2D tensors via multi-periodicity FFT.
  - `TiDE (Time-series Dense Encoder)`: High-speed linear dense encoder-decoder handling static and dynamic past/future covariates.
  - `Informer & Autoformer`: $O(L \log L)$ ProbSparse Attention and Auto-Correlation decomposition blocks.
  - `DeepAR`: Autoregressive recurrent probabilistic forecaster outputting Gaussian / Negative Binomial distribution parameters.
  - `Matrix Profile (STAMP / STOMP)`: Time series anomaly discord discovery and repeating motif extraction.

---

### Phase 15: Model Evaluation, Diagnostics, Uncertainty & Calibration (`evaluation/`)
- **Classification Metrics**:
  - Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, F-$eta$ Score.
  - ROC-AUC (Trapezoidal numerical integration), PR-AUC (Average Precision score).
  - Log-Loss / Binary Cross-Entropy, Confusion Matrix.
  - Matthew's Correlation Coefficient (MCC), Cohen's Kappa, Brier Score.
- **Regression & Forecasting Metrics**:
  - Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
  - $R^2$ Score, Adjusted $R^2$, Explained Variance Score.
  - Mean Absolute Percentage Error (MAPE), Symmetric MAPE (sMAPE), Mean Absolute Scaled Error (MASE).
  - Quantile / Pinball Loss ($q \in [0.1, 0.5, 0.9]$), Continuous Ranked Probability Score (CRPS).
- **Ranking & Recommendation Metrics**:
  - Normalized Discounted Cumulative Gain (NDCG@K), Mean Average Precision (MAP@K), Mean Reciprocal Rank (MRR), Hit Rate@K.
- **Model Calibration & Reliability**:
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
  - Reliability Diagrams (Calibration curves with binning).
  - Post-Hoc Calibrators: Platt Scaling (Sigmoid logistic calibration), Isotonic Regression calibrator, Temperature Scaling.
- **Data Drift, Concept Drift & Diagnostics**:
  - Population Stability Index (`PSI` with automated binning).
  - Kolmogorov-Smirnov (`KS-2Sample` test), Maximum Mean Discrepancy (`MMD` with RBF kernel), Wasserstein Distance.
  - Streaming Concept Drift Detectors: `ADWIN` (Adaptive Windowing) and `Page-Hinkley` test.

---

### Phase 16: Sovereign Explainable AI (XAI) & Mechanistic Interpretability (`explainability/`)
- **Model-Agnostic Explainers**:
  - `PermutationFeatureImportance`: Shuffle-based validation metric drop evaluation.
  - `KernelSHAP`: Weighted linear local surrogate estimating exact Shapley values via coalition sampling.
  - `Sovereign LIME`: Local Interpretable Model-agnostic Explanations with exponential distance kernel perturbation weighting and Ridge regression.
  - `PartialDependencePlots (PDP)` & `IndividualConditionalExpectation (ICE)` curves.
  - `Counterfactual Explanations`: Wachter gradient-based optimization finding minimal feature perturbation to flip predicted outcome.
  - `Anchors`: High-precision rule-based local explanations.
- **Tree-Specific Explainers**:
  - `TreeSHAP`: Exact $O(TLD^2)$ polynomial-time Shapley value computation for Decision Trees, Random Forests, and Gradient Boosting.
- **Neural & Visual Explainers**:
  - `IntegratedGradients`: Axiomatic path integral attribution satisfying Completeness and Implementation Invariance ($\int_0^1 rac{\partial F(x' + lpha(x - x'))}{\partial x_i} dlpha$).
  - `SmoothGrad`: Gaussian noise perturbation smoothing for sharp gradient saliency.
  - `DeepLIFT`: Difference-from-reference conservation attribution.
  - `GradCAM`, `GradCAM++`, `Score-CAM`, `LayerCAM` for convolutional feature maps.
- **Mechanistic & Transformer Interpretability**:
  - `Attention Rollout` & `Attention Flow`: Inter-layer attention propagation tracking.
  - `Direct Logit Attribution (DLA)`: Direct projection of attention head and MLP outputs onto vocabulary logit space.
  - `Activation Patching / Causal Tracing`: Representation intervention verifying factual recall paths.

---

### Phase 17: Autonomous AutoML, NAS & Universal Pipeline Engine (`pipeline/`, `automl/`)
- **Universal Leak-Free Pipeline Engine (`ck.pipeline`)**:
  - Sequential chaining: Loading $ightarrow$ Cleaning $ightarrow$ Preprocessing $ightarrow$ Transformations $ightarrow$ Modeling $ightarrow$ Evaluation.
  - Strict Statistical Isolation: Fitting scalers, encoders, PCA, and selectors strictly on `X_train` and applying frozen `PreprocessorState` / `TransformationState` to `X_test` / `X_val` / production inference data.
  - Resampling Isolation: SMOTE and oversampling applied strictly to training data, leaving validation and test splits untouched.
  - Production Serialization: `.save()` and `PipelineResult.load()` for single-call `.predict()` on raw unseen inputs.
- **Advanced Stacking & Ensembling**:
  - `SuperLearner / Multi-Layer Stacking`: Out-of-fold (OOF) cross-validated predictions generating meta-features for Level-2 meta-learners.
  - `Weighted Ensemble Optimizer`: Nelder-Mead / SLSQP optimization finding continuous weights that maximize validation metric.
  - `Dynamic Ensemble Selection (KNORA-Eliminate / KNORA-Union)`: Test-sample k-nearest neighbor local competence evaluation.
- **Automated Feature Engineering**:
  - `Deep Feature Synthesis (DFS)`: Automated feature crosses, numerical ratios, polynomial expansions, rolling aggregates, and date-part decomposition.
  - Target-guided discretization and automated frequency encodings.
- **Autonomous AutoML & Hyperparameter Optimization**:
  - `AutoTrainer`: 1-line automated task inference, data profiling, multi-candidate tournament, and champion model selection.
  - `BayesianOptimization`: Gaussian Process surrogate with Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions.
  - `Hyperband`: Successive Halving multi-armed bandit resource allocator.
  - `BOHB`: Integrated Bayesian Optimization + Hyperband combining fast multi-fidelity bandits with TPE/GP surrogate exploration.
  - `Genetic Pipeline Search (TPOT-Style)`: Evolutionary algorithm mutating and crossing over entire preprocessing + model graph pipelines.
- **Interactive Dark-Mode Reporting**:
  - Fully self-contained, responsive HTML dashboard generator with interactive embedded charts for EDA, training curves, confusion matrices, ROC curves, and SHAP plots.

---

### Bonus Phase: Sovereign Deep Reinforcement Learning (RL) (`models/rl/`)
- **Tabular RL**:
  - `QLearning`: Epsilon-Greedy exploration, Bellman optimality update $Q(s, a) \leftarrow Q(s, a) + lpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
  - Custom GridWorld, Bandit, and CartPole simulator environments.
- **Deep Q-Networks (DQN)**:
  - `DQN`: Neural Q-function approximator with Experience Replay Buffer and periodically updated Target Network.
  - `Double DQN`: Decoupled action selection and action evaluation mitigating overestimation bias.
  - `Dueling DQN`: Separate State Value $V(s)$ and Advantage $A(s, a)$ stream decomposition ($Q(s, a) = V(s) + (A(s, a) - rac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'))$).
  - `Prioritized Experience Replay (PER)`: TD-error proportional transition sampling.
- **Policy Gradients & Actor-Critic**:
  - `REINFORCE`: Monte Carlo policy gradient with state-value baseline subtraction.
  - `Advantage Actor-Critic (A2C)`: Synchronous Actor-Critic estimating advantage $A(s, a) = Q(s, a) - V(s)$.
  - `Proximal Policy Optimization (PPO)`: Clipped surrogate objective $L^{	ext{CLIP}}(	heta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(	heta)\hat{A}_t, 	ext{clip}(r_t(	heta), 1-\epsilon, 1+\epsilon)\hat{A}_tight) ight]$.
  - `Soft Actor-Critic (SAC)`: Off-policy maximum entropy Actor-Critic maximizing return + policy entropy $\mathcal{H}(\pi(\cdot|s))$.
- **Multi-Armed Bandits**:
  - `UCB1`: Upper Confidence Bound exploration.
  - `Thompson Sampling`: Bayesian posterior sampling with Beta-Bernoulli and Gaussian conjugate priors.
  - `LinUCB`: Contextual multi-armed bandits with linear payoff models.

---

## 4. Verification & Quality Assurance Protocol
1. **Zero Heavy Dependencies Constraint Check**: Automated import audit confirming absence of banned libraries (`torch`, `tensorflow`, `sklearn`, etc.).
2. **Deterministic Reproducibility**: Unified global seeding (`ck.core.config.set_seed(42)`).
3. **Cross-Platform Matrix Test Suite**: 100% test pass rate across Ubuntu, Windows, and macOS on Python 3.9, 3.10, 3.11, 3.12, and 3.13.
4. **Featherweight Wheel Distribution**: `python -m build` generating `< 5 MB` clean wheels verified by `twine check`.
