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

## Highlights & Sovereign Philosophy

1. **Zero Heavy Dependencies**: Built completely from first principles using pure NumPy, SciPy, Pandas, Matplotlib, Seaborn, and OpenCV-headless. Strictly zero PyTorch, TensorFlow, Scikit-Learn, Transformers, HuggingFace, Librosa, or SHAP dependencies.
2. **Ultra-Lightweight Distribution**: Full-featured ML/DL/Vision/NLP/Audio/Generative universe in a tiny PyPI wheel (< 1 MB).
3. **Computer Vision Universe**:
   - **Backbones & Classifiers**: LeNet, AlexNet, VGG (11/13/16/19), ResNet (18/34/50/101/152), InceptionV3, MobileNet (V1/V2/V3), DenseNet (121/161/169/201), EfficientNet (B0-B7), ShuffleNetV2, SqueezeNet, Vision Transformer (ViT), Swin Transformer (Shifted Windows), DeiT (Distillation Token), ConvNeXt.
   - **Object Detection & Segmentation**: YOLOv1, Faster R-CNN (RPN + RoI Align), SSD (Multi-scale default boxes), RetinaNet with Focal Loss, Feature Pyramid Networks (FPN).
   - **Vision XAI & Saliency**: Grad-CAM, Score-CAM, Eigen-CAM.
4. **NLP Transformers & Modern LLMs**:
   - **Tokenizers**: Byte-Pair Encoding (BPE), WordPiece, SentencePiece (Unigram language model), PorterStemmer, TF-IDF, BM25+.
   - **Attention Mechanisms**: Multi-Head Attention, RoPE (Rotary Position Embeddings), ALiBi, Sliding Window Attention (SWA), Grouped Query Attention (GQA), Multi-Head Latent Attention (DeepSeek MLA with low-rank KV compression & decoupled RoPE).
   - **LLM Architectures**: Mixture of Experts (MoE with auxiliary load balancing), RoBERTa, DeBERTa (3-term Disentangled Self-Attention), Mistral (SWA + GQA + SwiGLU + RoPE), Gemma (GeGLU + Scaled Embeddings + RMSNorm).
5. **Audio Signal Processing & Speech Transformers**:
   - **Signal Processing**: STFT, ISTFT (overlap-add with window normalization), Mel-Spectrograms, MFCCs, Delta features, SpecAugment.
   - **Speech Architectures**: Vectorized Conv1D / DepthwiseConv1D, Conformer (Macaron FFN + MHA + ConvModule + GLU), Audio Spectrogram Transformer (AST), Wav2Vec2 (Temporal Feature Encoder + Transformer).
6. **Generative AI Universe**:
   - **Autoencoders**: Variational Autoencoders (VAE with Reparameterization Trick & KL loss), Vector-Quantized VAE (VQ-VAE with Straight-Through Estimator codebook).
   - **Diffusion**: Denoising Diffusion Probabilistic Models (DDPM with sinusoidal time embeddings & reverse sampling).
   - **GANs**: Deep Convolutional GAN (DCGAN), Wasserstein GAN with Gradient Penalty (WGAN-GP).
7. **Graph Neural Networks (GNNs)**:
   - **Architectures**: GCN (Kipf & Welling), GAT (Multi-head attention), GraphSAGE (Mean, Max-pooling, Sum aggregators), GIN (Graph Isomorphism Network).
   - **Graph Utilities**: Symmetric & Random-Walk Adjacency Normalization, COO Edge Index Conversions, Global Graph Readout Pooling.
8. **Time Series & Forecasting Universe**:
   - **Statistical Models**: ARIMA(p, d, q) from scratch, Holt-Winters Exponential Smoothing (Level, Trend, Seasonality).
   - **Deep Forecasting**: N-BEATS (Doubly Residual generic & trend polynomial expansion), N-HiTS (Hierarchical multi-rate interpolation), PatchTST (Subseries patching + Channel-Independent Transformer), Seasonal Decomposition.
9. **Sovereign Explainable AI (XAI)**:
   - Integrated Gradients (Axiomatic attribution with Completeness), SmoothGrad (Gaussian noise smoothing), DeepLIFT (Difference-from-reference conservation), KernelSHAP, Permutation Feature Importance, Partial Dependence Plots (PDP).
10. **Sovereign AutoML Subsystem**:
    - `AutoTrainer`: Automated dataset profiling, multi-model candidate benchmarking, cross-validated leaderboard ranking, and champion selection.
    - `BayesianOptimization`: Gaussian Process surrogate with Expected Improvement (EI) and Upper Confidence Bound (UCB).
    - `Hyperband`: Successive Halving bandit tuner.

---

## Installation

```bash
pip install --upgrade chokkhu
```

---

## Quickstart: Complete Machine Learning & Deep Learning Suite

```python
import chokkhu as ck
import numpy as np

# 1. Automated Machine Learning (AutoML) in 1 Line
res = ck.auto_train(X_train, y_train, task="classification")
print(res.summary())
y_pred = res.predict(X_test)

# 2. Train any Vision, NLP, Audio, Generative, GNN, or Time Series Model
gcn_model = ck.train(model="gcn", X_train=node_features, adj=adj_matrix, out_features=3)
arima_model = ck.train(model="arima", X_train=time_series, p=2, d=1, q=1)
vae_model = ck.train(model="vae", X_train=images_flat, latent_dim=16)

# 3. Explain any Neural Model with Integrated Gradients or DeepLIFT
exp = ck.explain(gcn_model, node_features, method="integrated_gradients")
print(exp.summary())
```

---

## Complete Feature Matrix Summary

| Subsystem | Components & Architectures |
| :--- | :--- |
| **Tabular & ML** | Linear/Logistic Regression, Ridge, Lasso, ElasticNet, KNN, NaiveBayes, SVM, DecisionTree, RandomForest, GBM, KMeans, DBSCAN, Hierarchical, Q-Learning |
| **Data Engine** | Multi-Format I/O, Statistical & Image EDA, Missing Value Imputers (KNN, MICE), Outlier Isolation, Scalers, Resampling (SMOTE, ADASYN, Tomek) |
| **Vision Universe**| LeNet, AlexNet, VGG (11-19), ResNet (18-152), InceptionV3, MobileNet (V1-V3), DenseNet, EfficientNet (B0-B7), ShuffleNetV2, SqueezeNet, ViT, Swin, DeiT, ConvNeXt; YOLOv1, Faster R-CNN, SSD, RetinaNet, FPN; Grad-CAM, Score-CAM, Eigen-CAM |
| **NLP & LLMs** | Autograd Transformers, BPE / WordPiece / SentencePiece, RoPE, ALiBi, SWA, GQA, DeepSeek MLA, MoE with load balancing, RoBERTa, DeBERTa, Mistral, Gemma, PorterStemmer, TF-IDF, BM25+ |
| **Audio & Speech** | STFT, ISTFT, Mel-Spectrogram, MFCC, SpecAugment, Conv1D, Conformer, Audio Spectrogram Transformer (AST), Wav2Vec2 |
| **Generative AI** | Variational Autoencoders (VAE), Vector-Quantized VAE (VQ-VAE), Denoising Diffusion Probabilistic Models (DDPM), DCGAN, WGAN-GP |
| **GNNs** | GCN (Kipf & Welling), GAT (Multi-Head), GraphSAGE (Mean/Max/Sum), GIN, Symmetric Normalization, Global Pooling |
| **Time Series** | ARIMA(p,d,q), Holt-Winters Exponential Smoothing, N-BEATS, N-HiTS, PatchTST, Seasonal Decomposition, Lag Matrix |
| **Explainable AI** | Integrated Gradients, SmoothGrad, DeepLIFT, KernelSHAP, Permutation Feature Importance, Partial Dependence (PDP) |
| **AutoML** | AutoTrainer, Leaderboard Champion Selection, Gaussian Process Bayesian Optimization (EI/UCB), Hyperband Bandit Tuner |
| **Pipeline & CLI**| Leak-Free `pipeline()` engine, Serialization, `chokkhu eda / clean / pipeline / --version` CLI |

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
