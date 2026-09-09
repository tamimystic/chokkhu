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
│       ├── io/                            # Phase 2: Multi-Modal Data Ingestion & Serialization
│       │   ├── __init__.py
│       │   ├── tabular.py                 # CSV, TSV, JSON, Parquet, Feather, Excel, SQLite loaders
│       │   ├── image.py                   # Multi-threaded image directory & DICOM format loaders
│       │   ├── text.py                    # Raw text, corpus, JSONL, and line-by-line loaders
│       │   ├── audio.py                   # WAV, FLAC, MP3 byte decoding & signal loader
│       │   ├── video.py                   # Frame-by-frame pure OpenCV/NumPy video streamer
│       │   └── persistence.py             # Atomic model, Safetensors & pipeline serialization
│       │
│       ├── cleaning/                      # Phase 3: Deep Data Cleaning & Quality Control
│       │   ├── __init__.py
│       │   ├── missing_imputer.py         # Mean, Median, Mode, KNN, Iterative MICE, Spline, Matrix Factorization
│       │   ├── outlier_detector.py        # IQR, Z-score, MAD, Mahalanobis, Isolation Forest, LOF, Elliptic Envelope
│       │   ├── duplicate_handler.py       # Exact, subset, and fuzzy MinHash/Levenshtein duplicate handling
│       │   └── dtype_fixer.py             # Intelligent dtype coercion, rare category grouping & datetime parsing
│       │
│       ├── preprocessing/                 # Phase 4: Feature Preprocessing, Scaling & Encoding
│       │   ├── __init__.py
│       │   ├── scaling.py                 # StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerScaler, QuantileScaler, L1/L2 Normalizer
│       │   ├── encoding.py                # OneHot, Ordinal, Binary, Target (OOF smoothed), WoE/IV, Frequency, Hash, Helmert Encoders
│       │   └── feature_selection.py       # VarianceThreshold, CorrelationFilter, MutualInfo, ANOVA, RFE, RFECV, SFFS, LASSO Sparsity
│       │
│       ├── transformation/                # Phase 5: Space Projections, Manifold Learning, Resampling & Augmentation
│       │   ├── __init__.py
│       │   ├── decomposition.py           # PCA, IncrementalPCA, SVD, LDA, FactorAnalysis, FastICA
│       │   ├── manifold.py                # t-SNE (Barnes-Hut), KernelPCA, Isomap, MDS, UMAP (pure NumPy)
│       │   ├── resampling.py              # SMOTE, Borderline-SMOTE, ADASYN, SMOTE-NC, Tomek Links, ENN, SMOTETomek, RandomOver/UnderSampler
│       │   ├── feature_engineering.py     # PolynomialFeatures, SplineTransformer, Cross-interactions, Binning, Cyclical Sin/Cos
│       │   ├── augmentation_vision.py     # Flip, Rotate, Crop, Blur, Cutout, MixUp, CutMix, ColorJitter, AutoAugment
│       │   ├── augmentation_text.py       # Synonym replacement, Random swap, Random deletion, Typo injection
│       │   └── augmentation_audio.py      # Time stretch (Phase Vocoder), Pitch shift, Noise injection, SpecAugment, RIR Reverb
│       │
│       ├── splitting/                     # Phase 6: Leakage-Free Data Splitting & Cross-Validation
│       │   ├── __init__.py
│       │   ├── engine.py                  # Train/Test, Train/Val/Test (3-way split)
│       │   └── cross_validation.py        # KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, PurgedTimeSeriesSplit, RepeatedKFold, LOO
│       │
│       ├── models/                        # Phase 7 to 13 + RL: The Complete Modeling Universe
│       │   ├── __init__.py
│       │   ├── base.py                    # Base ChokkhuModel abstraction
│       │   ├── engine.py                  # Universal chokkhu.train() dispatcher & Auto-Tuning
│       │   │
│       │   ├── ml/                        # Phase 7: Classical ML & Ensembles
│       │   │   ├── __init__.py
│       │   │   ├── linear_regression.py   # OLS, Ridge, Lasso, ElasticNet, Huber, Passive-Aggressive, GLMs
│       │   │   ├── logistic_regression.py # Binary & Multinomial Logistic Regression (L-BFGS / SGD solvers)
│       │   │   ├── knn.py                 # K-Nearest Neighbors & Radius Neighbors with KD-Tree and Ball-Tree
│       │   │   ├── decision_tree.py       # CART Tree with Gini, Entropy, MSE, MAE, Cost-Complexity Pruning
│       │   │   ├── random_forest.py       # Bagging Ensemble with OOB scoring, feature subsets & quantile trees
│       │   │   ├── extra_trees.py         # Extremely Randomized Trees
│       │   │   ├── gradient_boosting.py   # Gradient Tree Boosting with Log-Odds, Deviance & HistGradientBoosting
│       │   │   ├── adaboost.py            # Adaptive Boosting (SAMME / SAMME.R)
│       │   │   ├── svm.py                 # Support Vector Machine (Linear, RBF, Poly, Sigmoid, Mahalanobis) with SMO solver
│       │   │   ├── naive_bayes.py         # Gaussian, Multinomial, Bernoulli, Complement NB, QDA
│       │   │   ├── kmeans.py              # K-Means++ and Mini-Batch K-Means with Inertia & Silhouette
│       │   │   ├── kmedoids.py            # K-Medoids with PAM algorithm
│       │   │   ├── dbscan.py              # Density-Based Spatial Clustering
│       │   │   ├── optics.py              # Ordering Points To Identify Clustering Structure
│       │   │   ├── hierarchical.py        # Agglomerative Clustering (Ward, Single, Complete, Average linkages)
│       │   │   ├── gmm.py                 # Gaussian Mixture Models with Expectation-Maximization (EM)
│       │   │   ├── spectral_clustering.py # Spectral Clustering via Graph Laplacian Eigen-decomposition
│       │   │   └── anomaly.py             # Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
│       │   │
│       │   ├── dl/                        # Phase 8: Sovereign Deep Learning & Autograd Framework
│       │   │   ├── __init__.py
│       │   │   ├── autograd.py            # Computational Graph, Tensor, Tape-based Reverse Autodiff
│       │   │   ├── layers.py              # Linear, Dropout, SpatialDropout2D, BatchNorm1D, BatchNorm2D, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, Embedding
│       │   │   ├── activations.py         # ReLU, GELU, SiLU, LeakyReLU, PReLU, ELU, SELU, Mish, Softmax, LogSoftmax, Sigmoid, Tanh
│       │   │   ├── losses.py              # MSE, MAE, Huber, CrossEntropy (log-sum-exp), BCEWithLogits, Focal, Dice, Tversky, Triplet, Contrastive, CTC
│       │   │   ├── optimizers.py          # SGD (Momentum, Nesterov), Adam, AdamW, RMSProp, Adagrad, Adadelta, Lion
│       │   │   ├── schedulers.py          # StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
│       │   │   ├── callbacks.py           # EarlyStopping, ModelCheckpoint, LRLogger, TensorBoard-style ASCII visualizer
│       │   │   └── sequential.py          # Sequential & Functional Model Builder (.fit, .predict, .predict_proba)
│       │   │
│       │   ├── vision/                    # Phase 9: Complete Computer Vision Architecture Universe
│       │   │   ├── __init__.py
│       │   │   ├── conv_layers.py         # Conv2D (im2col), ConvTranspose2D, DepthwiseSeparableConv2D, GroupedConv2D, ChannelShuffle, DilatedConv2D, DeformableConv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, SPP
│       │   │   ├── batchnorm2d.py         # Spatial 2D Batch Normalization
│       │   │   ├── attention_blocks.py    # SEBlock (Squeeze-and-Excitation), CBAM, CoordAttention, Non-Local Blocks
│       │   │   ├── detection.py           # Anchor Boxes, IoU, GIoU, DIoU, CIoU, NMS, Soft-NMS, YOLO-Head, SSD, RetinaNet
│       │   │   ├── fpn.py                 # Feature Pyramid Networks (FPN), PANet, BiFPN
│       │   │   ├── losses.py              # FocalLoss, DiceLoss, TverskyLoss, GIoU, DIoU, CIoU
│       │   │   ├── xai_vision.py          # GradCAM, GradCAM++, Score-CAM, LayerCAM, Guided Backprop, Saliency Maps
│       │   │   └── architectures/         # Comprehensive CNN & Vision Architectures
│       │   │       ├── __init__.py
│       │   │       ├── lenet.py           # LeNet-5 (1998)
│       │   │       ├── alexnet.py         # AlexNet (2012), ZFNet (2013)
│       │   │       ├── vgg.py             # VGG-11, VGG-13, VGG-16, VGG-19 (2014)
│       │   │       ├── inception.py       # GoogLeNet / InceptionV1, InceptionV3, InceptionV4 (2014-2016)
│       │   │       ├── resnet.py          # ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, WideResNet (2015)
│       │   │       ├── resnext.py         # ResNeXt-50, ResNeXt-101 (2017)
│       │   │       ├── densenet.py        # DenseNet-121, DenseNet-169, DenseNet-201 (2017)
│       │   │       ├── squeezenet.py      # SqueezeNet with Fire Modules (2016)
│       │   │       ├── mobilenet.py       # MobileNetV1, MobileNetV2 (Inverted Residuals), MobileNetV3 (2017-2019)
│       │   │       ├── shufflenet.py      # ShuffleNetV1, ShuffleNetV2 (Channel Shuffle) (2018)
│       │   │       ├── efficientnet.py    # EfficientNet-B0 to B7 (MBConv + Compound Scaling) (2019)
│       │   │       ├── convnext.py        # ConvNeXt-Tiny, ConvNeXt-Small, ConvNeXt-Base (Modern Pure CNN) (2022)
│       │   │       ├── unet.py            # U-Net Image Segmentation (2015)
│       │   │       ├── fcn.py             # FCN-8s, FCN-16s, FCN-32s Semantic Segmentation (2015)
│       │   │       ├── deeplab.py         # DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP)
│       │   │       ├── vit.py             # Vision Transformer (ViT-Tiny, ViT-Base, ViT-Large) (2020)
│       │   │       ├── swin.py            # Swin Transformer (Shifted Window Attention) (2021)
│       │   │       └── deit.py            # DeiT (Distillation Token Vision Transformer) (2021)
│       │   │
│       │   ├── nlp/                       # Phase 10: NLP, Sequence Models & Ultra-Modern Frontier LLMs
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

### Phase 1: Foundation, Infrastructure, Dynamic Config & Multi-OS CI/CD
- **Global Deterministic Control & Hardware Optimization**:
  - Unified seed management (`ck.core.config.set_seed(42)`) controlling random generators in NumPy, Python `random`, and internal weight initializers.
  - Multi-threaded CPU BLAS optimization (`set_num_threads(n)` for OpenBLAS / MKL / Accelerate).
  - Vectorized SIMD-friendly array operations ensuring zero C-compiler dependency while maximizing throughput.
- **Modern Packaging & PEP Compliance**:
  - Zero build-time compilation, PEP 518/621 compliance (`pyproject.toml`, `setup.py`), `< 5 MB` wheel target.
  - Multi-OS GitHub Actions testing across Ubuntu, Windows, and macOS for Python 3.9, 3.10, 3.11, 3.12, and 3.13 (17 active jobs, 100% Green).
- **Core Exceptions & Logging**:
  - Typed exceptions: `DataLeakageError`, `DimensionMismatchError`, `ConvergenceWarning`, `NumericalInstabilityError`.
  - Rich ANSI/Unicode terminal logger with configurable verbosity levels, execution timer wrappers, and ASCII progress tracking.

---

### Phase 2: Multi-Modal Data Ingestion, Streaming & Serialization (`io/`)
- **Tabular Ingestion**:
  - Memory-mapped streaming reader for ultra-large CSV, TSV, JSON, and JSON Lines (`.jsonl`).
  - Native Parquet and Feather format parser integration using raw buffer streams without heavy PyArrow dependency.
  - SQLite and SQL database connection interface executing direct table queries into NumPy/Pandas containers.
  - Native Excel (`.xlsx`) XML stream loader for office spreadsheets.
- **Vision, Audio & Video Multi-Modal Ingestion**:
  - Multi-threaded asynchronous image folder reader with class directory mapping, batch prefetching, and Exif metadata parsing.
  - Medical image DICOM format raw byte parser.
  - Native WAV, FLAC, OGG, MP3 PCM audio decoder reading sample rate, channel count, and floating-point audio buffers.
  - Frame-by-frame pure OpenCV/NumPy video stream generator for action recognition and video analysis.
- **Persistence & Serialization**:
  - `SafeTensors` format serializer: Zero-copy, header-indexed, pure NumPy model weight saving/loading preventing arbitrary code execution vulnerability of standard pickle.
  - Atomic pipeline serializer bundling fitted preprocessors, transformation states, models, and metadata into compressed `.chokkhu` archives.

---

### Phase 3: Deep Data Cleaning, Quality Control & Automated Imputation (`cleaning/`)
- **Advanced Imputation Engines**:
  - `Mean`, `Median`, `Mode`, and `Constant` statistical imputers with automated dtype mapping.
  - `KNNImputer`: Distance-weighted k-nearest neighbors imputation over Euclidean/Manhattan metrics.
  - `IterativeImputer (MICE)`: Multivariate Imputation by Chained Equations modeling each missing feature as a function of all other features via Bayesian ridge / decision tree regressors.
  - `Spline / Linear Interpolation`: Forward-fill and backward-fill time-series imputers.
  - `MatrixFactorizationImputer`: Low-rank SVD reconstruction for high-dimensional sparse tabular data.
- **Multivariate Outlier & Anomaly Detection**:
  - Interquartile Range (`IQR 1.5x / 3x` Tukey fences) with selective winsorization / clipping.
  - `Z-Score` and `Modified Z-Score (MAD)` (Median Absolute Deviation robust to extreme clusters).
  - `Mahalanobis Distance`: Covariance-based multivariate outlier detection accounting for inter-feature correlations ($D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$).
  - `Isolation Forest`: Randomized isolation trees isolating anomalies at shallow tree depths.
  - `Local Outlier Factor (LOF)`: Local reachability density comparison detecting outliers relative to surrounding density.
  - `Elliptic Envelope`: Minimum Covariance Determinant (MCD) estimating Gaussian support envelope.
- **Data Sanitation & Hygiene**:
  - Exact, subset, and fuzzy duplicate handling using MinHash / Levenshtein N-gram distance.
  - Intelligent dtype coercion (converting string numbers with currency symbols, percentages, and locale separators).
  - Categorical typo clustering and rare category grouping ($\text{frequency} < \tau \rightarrow \text{"Other"}$).
  - ISO 8601, RFC 2822, and Unix timestamp parsing into standard datetime objects.

---

### Phase 4: Feature Preprocessing, Scaling & Advanced Encodings (`preprocessing/`)
- **Scaling & Normalization Suite**:
  - `StandardScaler`: Zero-mean unit-variance transformation:
    $$z = \frac{x - \mu}{\sigma}$$
  - `MinMaxScaler`: Bounded interval transformation $z = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a$.
  - `MaxAbsScaler`: Scaled by maximum absolute value $z = \frac{x}{|x_{\max}|}$ preserving sparsity.
  - `RobustScaler`: Median and interquartile range scaling $z = \frac{x - \text{median}}{\text{IQR}}$ immune to outliers.
  - `PowerScaler`: Yeo-Johnson (supporting negative values) and Box-Cox (positive values) power transforms with Maximum Likelihood parameter $\lambda$ estimation for variance stabilization.
  - `QuantileScaler`: Rank-based non-linear mapping transforming arbitrary distributions into Uniform or Gaussian Normal distributions.
  - `L1 / L2 Normalizer`: Sample-wise vector unit norm projection ($x / \|x\|_p$).
- **Categorical Encoding Suite**:
  - `OneHotEncoder`: Sparse/dense binary indicator mapping with `handle_unknown='ignore'`, `min_frequency` thresholding, and drop-first dummy encoding.
  - `OrdinalEncoder` / `LabelEncoder`: Deterministic integer mapping with unknown token handling.
  - `BinaryEncoder`: Integer encoding projected into binary bit columns for high-cardinality features.
  - `TargetEncoder`: M-estimate smoothed mean target encoding with K-Fold Out-Of-Fold (OOF) cross-validation preventing target leakage:
    $$\hat{x}_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$
  - `WeightOfEvidenceEncoder (WoE)` & `InformationValue (IV)`: Credit risk and binary classification log-odds encoding ($\ln(\frac{\% \text{Goods}}{\% \text{Bads}})$).
  - `FrequencyEncoder`: Replacing category labels with their normalized training set occurrence frequencies.
  - `HashEncoder`: MurmurHash3 feature hashing projecting unbounded vocabularies into fixed $N$-dimensional spaces.
  - `Helmert / Sum / Backward Difference Contrast Encoders`.
- **Feature Selection Suite**:
  - `VarianceThreshold`: Eliminating zero-variance constants and near-zero-variance quasi-constants.
  - `CorrelationFilter`: Pairwise Pearson, Spearman, and Kendall rank multi-collinearity removal with automatic drop heuristics.
  - `MutualInfoSelector`: Non-linear dependency scoring via k-nearest neighbor entropy estimation for discrete and continuous targets:
    $$\mathcal{I}(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x)p(y)} dx dy$$
  - `ANOVA F-Selector`: Analysis of Variance F-test testing linear relationship between individual numerical features and categorical classes ($F = \frac{\text{MSB}}{\text{MSW}}$).
  - `RFESelector` & `RFECV`: Recursive Feature Elimination with step pruning and cross-validated optimal subset scoring.
  - `Sequential Feature Selector`: Sequential Forward Selection (SFS) and Sequential Backward Selection (SBS).
  - `L1 Sparsity Selector`: LASSO / ElasticNet L1 penalty pruning non-zero coefficients.

---

### Phase 5: Space Projections, Manifold Learning, Resampling & Augmentations (`transformation/`)
- **Linear Space Projections**:
  - `PCA`: Exact Full SVD, Randomized SVD, and Incremental Mini-Batch Principal Component Analysis with explained variance ratio and whitening.
  - `TruncatedSVD`: Latent Semantic Analysis (LSA) for sparse term-document matrices.
  - `LinearDiscriminantAnalysis (LDA)`: Supervised dimensionality reduction maximizing Fisher's class separability criterion ($\frac{w^T S_B w}{w^T S_W w}$).
  - `FactorAnalysis`: Probabilistic model discovering unobserved latent factors with orthogonal rotation (Varimax).
  - `FastICA`: Independent Component Analysis decomposing non-Gaussian signals via negentropy optimization.
- **Non-Linear Manifold Learning**:
  - `t-SNE`: t-Distributed Stochastic Neighbor Embedding with exact $O(N^2)$ and Barnes-Hut quadtree $O(N \log N)$ approximations.
  - `KernelPCA`: Non-linear projection using RBF (Gaussian), Polynomial, Sigmoid, and Cosine Mercer kernels.
  - `Isomap`: Isometric feature mapping computing geodesic shortest paths via Dijkstra algorithm followed by MDS.
  - `Multidimensional Scaling (MDS)`: Metric distance preservation in low-dimensional coordinates.
  - `UMAP`: Uniform Manifold Approximation and Projection using fuzzy simplicial sets and stochastic gradient descent in pure NumPy.
- **Resampling & Imbalanced Learning**:
  - `SMOTE`: Synthetic Minority Over-sampling Technique generating convex combinations of $k$-nearest minority neighbors:
    $$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
  - `Borderline-SMOTE`: Focusing synthetic oversampling strictly on minority samples near the decision boundary (DANGER zone).
  - `ADASYN`: Adaptive Synthetic sampling generating more synthetic instances for minority samples harder to learn.
  - `SMOTE-NC`: Generalized SMOTE handling mixed nominal and continuous features.
  - `RandomOverSampler` & `RandomUnderSampler`.
  - `Tomek Links`: Identifying and removing ambiguous nearest neighbor pairs of opposing classes across the boundary.
  - `EditedNearestNeighbors (ENN)`: Under-sampling majority instances whose class differs from majority of their $k$-nearest neighbors.
  - `SMOTE-Tomek` & `SMOTE-ENN`: Combined over-sampling and cleaning pipelines.
- **Feature Engineering & Mathematical Transforms**:
  - `PolynomialFeatures`: Generating interaction terms and powers up to degree $d$ with `interaction_only` and `include_bias` options.
  - `SplineTransformer`: Periodic and B-spline piecewise polynomials capturing non-linear feature curves.
  - `CyclicalEncodings`: Converting periodic time features (Hour, DayOfWeek, Month) into smooth continuous orthogonal coordinates ($\sin(\frac{2\pi t}{T}), \cos(\frac{2\pi t}{T})$).
  - `Binning / Discretization`: Equal-width, Equal-frequency (Quantile), and K-Means cluster discretization.
- **Multi-Modal Augmentation Suites**:
  - **Vision**: Random Horizontal/Vertical Flip, Random Rotate ($[-180^\circ, 180^\circ]$ bilinear), Random Crop & Resize, Color Jitter (Brightness, Contrast, Saturation, Hue), Gaussian Blur, Random Erasing / Cutout, MixUp ($x = \lambda x_1 + (1-\lambda) x_2$), CutMix (bounding box patch replacement with area ratio label interpolation), AutoAugment policy engine.
  - **Text**: Synonym Replacement (WordNet-style semantic replacement), Random Insertion, Random Swap, Random Deletion (EDA - Easy Data Augmentation), Back-Translation simulation, Character-level typo injection.
  - **Audio**: Time Stretch (Phase Vocoder), Pitch Shift, Additive Gaussian Noise, SpecAugment (Time & Frequency masking), Room Impulse Response (RIR) convolution reverb.

---

### Phase 6: Leakage-Free Splitting & Stratification (`splitting/`)
- **Splitting Strategies**:
  - `train_test_split`: Deterministic random splitting with shuffle and seed control.
  - `train_val_test_split`: 3-way statistical split ensuring completely independent evaluation.
- **Cross-Validation Engines**:
  - `KFold`: Standard $k$-fold cross-validation.
  - `StratifiedKFold`: Preserving exact class label proportions across every fold.
  - `GroupKFold`: Ensuring distinct groups (e.g., individual patients, users, devices) never overlap between train and validation splits.
  - `TimeSeriesSplit`: Expanding window and rolling origin cross-validation preserving strict temporal causality.
  - `PurgedGroupTimeSeriesSplit`: Combinatorial purged cross-validation with embargo periods eliminating serial correlation leakage in financial/time-series data.
  - `RepeatedKFold` & `LeaveOneOut (LOO)`.

---

### Phase 7: Classical Machine Learning & Ensembles (`models/ml/`)
- **Linear & Generalized Linear Models (GLMs)**:
  - `LinearRegression`: Ordinary Least Squares (OLS) via Normal Equations $(X^T X)^{-1} X^T y$ and SVD.
  - `RidgeRegression`: L2 regularized regression solving $(X^T X + \alpha I)^{-1} X^T y$.
  - `LassoRegression`: L1 regularized regression via Coordinate Descent with soft-thresholding operator $\mathcal{S}(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, 0)$.
  - `ElasticNet`: Convex combination of L1 and L2 penalties ($\alpha \rho \|w\|_1 + \frac{\alpha (1-\rho)}{2} \|w\|_2^2$).
  - `HuberRegressor`: Linear regression robust to outliers minimizing Huber loss.
  - `LogisticRegression`: Binary and Multinomial (Softmax) Logistic Regression with L1, L2, ElasticNet penalties and vectorized L-BFGS / SGD solvers.
  - `PassiveAggressive`: Online margin-based classifier and regressor for streaming data.
- **Instance-Based Models**:
  - `KNNClassifier` & `KNNRegressor`: K-Nearest Neighbors supporting Uniform and Distance-weighted voting, accelerated by pure NumPy KD-Tree and Ball-Tree spatial indices.
  - `RadiusNeighbors`: Fixed distance radius ball voting.
  - `NearestCentroid`: Prototype-based classification.
- **Decision Trees & Tree Ensembles**:
  - `DecisionTreeClassifier` & `DecisionTreeRegressor`: CART trees supporting Gini Impurity, Shannon Entropy, Mean Squared Error (MSE), Mean Absolute Error (MAE) criteria, minimum sample leaf constraints, maximum depth, and Minimal Cost-Complexity Pruning with parameter $\alpha$.
  - `RandomForestClassifier` & `RandomForestRegressor`: Bootstrap aggregation (Bagging) ensemble with random feature subspace sampling at each split ($\sqrt{p}$ or $\log_2 p$), out-of-bag (OOB) scoring, and quantile prediction intervals.
  - `ExtraTreesClassifier` & `ExtraTreesRegressor`: Extremely Randomized Trees sampling random split thresholds per feature for maximum variance reduction.
  - `IsolationForest`: Unsupervised anomaly isolation trees computing anomaly scores from average path lengths $c(n)$.
- **Boosting Algorithms**:
  - `GradientBoostingClassifier` & `GradientBoostingRegressor`: Gradient Tree Boosting with shrinkage learning rate $\eta$, stochastic subsampling, deviance / log-odds / MSE loss functions, and tree leaf value optimization.
  - `AdaBoostClassifier` & `AdaBoostRegressor`: Adaptive Boosting (SAMME and SAMME.R algorithms) adjusting sample weights $w_i \leftarrow w_i \exp(\alpha_m \mathbb{I}(y_i \neq G_m(x_i)))$.
  - `HistGradientBoosting`: Fast histogram binning (256 integer bins) enabling $10\times$ faster split evaluations on massive datasets.
- **Support Vector Machines (SVM)**:
  - `SVC` & `SVR`: Support Vector Classification and Regression implementing Platt's Sequential Minimal Optimization (SMO) algorithm.
  - Support for Linear, Polynomial ($(\gamma \langle x, x' \rangle + r)^d$), RBF ($\exp(-\gamma \|x - x'\|^2)$), Sigmoid ($\tanh(\gamma \langle x, x' \rangle + r)$), and Mahalanobis kernels.
  - `LinearSVC`: High-speed dual coordinate descent linear SVM.
- **Probabilistic & Bayesian Classifiers**:
  - `GaussianNB`: Continuous feature classification via Gaussian probability density function $\mathcal{N}(\mu_{ck}, \sigma_{ck}^2)$.
  - `MultinomialNB`: Word count / frequency classification with Laplace / Lidstone smoothing $\alpha$.
  - `BernoulliNB`: Binary feature naive Bayes with multivariate Bernoulli models.
  - `ComplementNB`: Specialized Naive Bayes designed for severely imbalanced text corpora.
  - `LinearDiscriminantAnalysis (LDA)` & `QuadraticDiscriminantAnalysis (QDA)` classifiers.
- **Clustering Universe**:
  - `KMeans`: K-Means with K-Means++ seeded initialization, Lloyd's optimization, and Mini-Batch K-Means for streaming datasets.
  - `KMedoids`: Partitioning Around Medoids (PAM) robust to extreme outliers using arbitrary distance metrics.
  - `DBSCAN`: Density-Based Spatial Clustering identifying core points, border points, and arbitrary-shaped clusters with noise isolation ($\epsilon$, $\text{MinPts}$).
  - `OPTICS`: Reachability distance ordering graph clustering.
  - `AgglomerativeClustering`: Hierarchical clustering with Ward's variance minimization, Complete (maximum), Average, and Single linkage distance updates.
  - `GaussianMixture (GMM)`: Soft probabilistic clustering via Expectation-Maximization (EM) estimating Gaussian components $(\pi_k, \mu_k, \Sigma_k)$ with full, tied, diag, and spherical covariance constraints.
  - `SpectralClustering`: Graph Laplacian eigen-decomposition followed by K-Means in spectral embedding space.
- **Anomaly Detection**:
  - `IsolationForest`, `LocalOutlierFactor (LOF)`, `OneClassSVM`, `EllipticEnvelope` (Minimum Covariance Determinant).

---

### Phase 8: Sovereign Deep Learning, Dynamic Autograd & Computational Graph (`models/dl/`, `core/tensor.py`)
- **Autograd Engine**:
  - Dynamic DAG (Directed Acyclic Graph) construction with reverse-mode automatic differentiation, tape-based execution, broadcasting backward rules, in-place operation tracking, and memory-retaining computation graphs.
- **Layers**:
  - `Linear` / `Dense`, `Dropout`, `SpatialDropout2D`, `AlphaDropout`, `BatchNorm1D`, `BatchNorm2D`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm`, `Embedding`, `Flatten`, `Reshape`, `Permute`.
- **Activation Functions**:
  - `ReLU`, `LeakyReLU`, `PReLU` (learnable slope), `ELU`, `SELU` (self-normalizing scale/alpha), `GELU` (Gaussian Error Linear Unit with exact erf and tanh approximations), `SiLU / Swish` ($x \cdot \sigma(\beta x)$), `Mish` ($x \tanh(\ln(1 + e^x))$), `Hardswish`, `Hardsigmoid`, `Softmax`, `LogSoftmax`, `Softplus`, `Tanh`, `Sigmoid`.
- **Loss Functions**:
  - `MSE`, `MAE`, `Huber Loss`, `Smooth L1 Loss`, `CrossEntropyLoss` (log-sum-exp numerically stabilized with label smoothing), `BCEWithLogits`, `FocalLoss` ($\alpha (1-p_t)^\gamma \log(p_t)$), `DiceLoss`, `TverskyLoss`, `TripletMarginLoss`, `ContrastiveLoss`, `CosineSimilarityLoss`, `CTCLoss` (Connectionist Temporal Classification for speech/OCR).
- **Optimizers**:
  - `SGD` (with Polyak Momentum, Nesterov Accelerated Gradient, Weight Decay), `Adam` (Adaptive Moment Estimation with bias correction), `AdamW` (Decoupled Weight Decay), `RMSProp` (Exponential moving average of squared gradients), `Adagrad`, `Adadelta`, `Adamax`, `NAdam`, `Lion` (EvoLved Sign Momentum).
- **Learning Rate Schedulers**:
  - `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `ReduceLROnPlateau`, `OneCycleLR` (Smith 1-cycle policy), `LinearWarmupLR`.
- **Training Callbacks & Containers**:
  - `EarlyStopping` (patience, min_delta, restore_best_weights), `ModelCheckpoint` (saving top-k checkpoints), `LRLogger`, `TensorBoard-style ASCII loss visualizer`, `Sequential` & `Functional` Model Containers with `.compile()`, `.fit()`, `.evaluate()`, `.predict()`, `.predict_proba()`.

---

### Phase 9: Complete Computer Vision Architecture Universe (`models/vision/`)
- **Convolutional Mechanics**:
  - Vectorized `im2col` & `col2im` GEMM `Conv2D`, `ConvTranspose2D` (Deconvolution / Fractional strided), `DepthwiseSeparableConv2D`, `GroupedConv2D`, `ChannelShuffle`, `Dilated / Atrous Conv2D` (rate $r$), `Deformable Convolution` (learnable offset sampling), `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`, `GlobalMaxPool2D`, `Spatial Pyramid Pooling (SPP)`.
- **Visual Attention Modules**:
  - Squeeze-and-Excitation (`SEBlock` with channel-wise squeeze and excitation), Convolutional Block Attention Module (`CBAM` with spatial and channel attention), Coordinate Attention (`CoordAttention`), `Non-Local Neural Networks` (Self-Attention in CV).
- **Architectures**:
  - **Pioneers**: `LeNet5` (1998), `AlexNet` (2012), `ZFNet` (2013), `VGG11`, `VGG13`, `VGG16`, `VGG19` (2014).
  - **Multi-Branch & Inception**: `GoogLeNet` / `InceptionV1` (2014), `InceptionV3` (Factorized convolutions 2015), `InceptionV4` / `Inception-ResNet`.
  - **Residual & Dense Networks**: `ResNet18`, `ResNet34`, `ResNet50`, `ResNet101`, `ResNet152` (He et al. 2015), `ResNeXt50`, `ResNeXt101` (Grouped residual transformations 2017), `DenseNet121`, `DenseNet169`, `DenseNet201` (Dense connectivity with transition layers 2017), `SqueezeNet` (Fire modules 2016), `WideResNet`.
  - **Mobile, Edge & Efficient Networks**: `MobileNetV1` (Depthwise separable convolutions 2017), `MobileNetV2` (Inverted residuals with linear bottlenecks 2018), `MobileNetV3` (Hard-swish and NAS tuned 2019), `ShuffleNetV1`, `ShuffleNetV2` (Channel split and channel shuffle 2018), `EfficientNetB0` to B7 (MBConv with compound scaling 2019).
  - **Modern Pure CNNs**: `ConvNeXtTiny`, `ConvNeXt` (Liu et al. 2022 modern pure 7x7 depthwise conv architecture).
  - **Semantic & Medical Segmentation**: `UNet` (Ronneberger et al. 2015 Contracting & Expansive paths with skip connections), `FCN8s`, `FCN16s`, `FCN32s` (Long et al. 2015 Fully Convolutional Networks), `SegNet` (Index unpooling), `DeepLabV3+` (Atrous Spatial Pyramid Pooling - ASPP).
  - **Object Detection & Heads**: Anchor Box generation, Intersection over Union (IoU), Generalized IoU (GIoU), Distance IoU (DIoU), Complete IoU (CIoU), Non-Maximum Suppression (NMS, Soft-NMS), `YOLOHead` (Grid bounding box & class confidence loss), `SSDHead` (Single Shot MultiBox Detector with multi-scale feature anchors), `RetinaNetHead` (Focal Loss for dense anchor classification), Feature Pyramid Network (`FPN`), Path Aggregation Network (`PANet`).
  - **Vision Transformers (ViT)**: `VisionTransformer` (ViT-Tiny, ViT-Base, ViT-Large with patch extraction, linear projection, class token, position embeddings), `SwinTransformer` (Shifted Window Attention with cyclic shift and patch merging 2021), `DeiT` (Data-efficient Image Transformers with Distillation Token 2021).
  - **Visual Explainability (XAI)**: `GradCAM` (Gradient-weighted Class Activation Mapping), `GradCAM++`, `Score-CAM`, `LayerCAM`, `Guided Backpropagation`, `Saliency Maps`.

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
  - `Mamba / S6 Selective State Space Model`: Discrete-time parameter-dependent scan $h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t$, $y_t = \mathbf{C}_t h_t$.
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
  - `Direct Preference Optimization` (DPO): Direct policy parameter alignment without auxiliary reward models:
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
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
  - `WGAN-GP`: Wasserstein GAN with exact Gradient Penalty constraint:
    $$\mathcal{L}_{\text{WGAN-GP}} = \mathbb{E}[D(\tilde{x})] - \mathbb{E}[D(x)] + \lambda \mathbb{E}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$
  - `StyleGAN2 / StyleGAN-XL`: Mapping Network $f: \mathcal{Z} \rightarrow \mathcal{W}$, Style Modulation/Demodulation, Noise injection, and Path Length Regularization.
  - `Pix2Pix` & `CycleGAN`: Paired and Unpaired Image-to-Image translation with PatchGAN discriminator and cycle consistency loss.
- **Diffusion Models & Deterministic Samplers**:
  - `DDPM` (Denoising Diffusion Probabilistic Models): Linear/Cosine $\beta_t$ variance schedulers, sinusoidal time embeddings, reverse Gaussian denoising.
  - `DDIM` (Denoising Diffusion Implicit Models): Deterministic fast sampling loop achieving high fidelity in 15-20 steps.
  - `Classifier-Free Guidance (CFG)`: Conditional vs unconditional score interpolation:
    $$v_{\text{guided}} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
  - `DPM-Solver` & `Euler A` high-order ODE samplers.
- **Latent Diffusion & Flow Matching**:
  - `Latent Diffusion Models (LDM / Stable Diffusion style)`: VAE latent space diffusion with Cross-Attention text/context conditioning.
  - `Rectified Flow Matching (Flux / SD3)`: Optimal transport straight-line velocity field ODE solver:
    $$\frac{dx_t}{dt} = v_t(x_t), \quad x_t = (1 - t) x_0 + t x_1$$
  - `Normalizing Flows (RealNVP, GLOW)`: Invertible affine coupling layers with exact log-determinant Jacobian computation.
- **Adapters & Fine-Tuning**:
  - `LoRA (Low-Rank Adaptation)`: Parameter-efficient rank-$r$ adaptation $\Delta W = \frac{\alpha}{r} B \cdot A$ for all linear/convolutional layers.
  - `ControlNet`: Spatial condition injection via zero-initialized convolution layers.

---

### Phase 13: Graph Neural Networks & Geometric Deep Learning (`models/gnn/`)
- **Graph Foundations & Spectral Operations**:
  - Graph Adjacency, Degree Matrix, Symmetric Normalized Laplacian ($\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$), Random-Walk Normalization ($\tilde{D}^{-1} \tilde{A}$).
  - Dense $\leftrightarrow$ Sparse COO Edge Index conversion.
  - Global Graph Readout Pooling: Mean, Max, Sum, Attention-based readout.
- **Core GNN Layers**:
  - `GCNLayer` (Kipf & Welling): First-order localized spectral graph convolution:
    $$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$
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
  - Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, F-$\beta$ Score.
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
  - `IntegratedGradients`: Axiomatic path integral attribution satisfying Completeness and Implementation Invariance:
    $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$
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
  - Sequential chaining: Loading $\rightarrow$ Cleaning $\rightarrow$ Preprocessing $\rightarrow$ Transformations $\rightarrow$ Modeling $\rightarrow$ Evaluation.
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
  - `QLearning`: Epsilon-Greedy exploration, Bellman optimality update:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$
  - Custom GridWorld, Bandit, and CartPole simulator environments.
- **Deep Q-Networks (DQN)**:
  - `DQN`: Neural Q-function approximator with Experience Replay Buffer and periodically updated Target Network.
  - `Double DQN`: Decoupled action selection and action evaluation mitigating overestimation bias.
  - `Dueling DQN`: Separate State Value $V(s)$ and Advantage $A(s, a)$ stream decomposition:
    $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)$$
  - `Prioritized Experience Replay (PER)`: TD-error proportional transition sampling.
- **Policy Gradients & Actor-Critic**:
  - `REINFORCE`: Monte Carlo policy gradient with state-value baseline subtraction.
  - `Advantage Actor-Critic (A2C)`: Synchronous Actor-Critic estimating advantage $A(s, a) = Q(s, a) - V(s)$.
  - `Proximal Policy Optimization (PPO)`: Clipped surrogate objective:
    $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  - `Soft Actor-Critic (SAC)`: Off-policy maximum entropy Actor-Critic maximizing expected reward plus policy entropy $\mathcal{H}(\pi(\cdot|s))$.
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
