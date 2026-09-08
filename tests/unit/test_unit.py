from __future__ import annotations

import chokkhu


def test_package_metadata():
    assert hasattr(chokkhu, "__version__")
    assert isinstance(chokkhu.__version__, str)


def test_package_exports():
    assert hasattr(chokkhu, "clean")
    assert hasattr(chokkhu, "preprocess")
    assert hasattr(chokkhu, "split")
    assert hasattr(chokkhu, "load")
    assert hasattr(chokkhu, "save")
    assert hasattr(chokkhu, "eda")
    assert hasattr(chokkhu, "train")
    assert hasattr(chokkhu, "evaluate")
    assert hasattr(chokkhu, "explain")
    assert hasattr(chokkhu, "pipeline")
    # Vision
    assert hasattr(chokkhu, "ResNet18")
    assert hasattr(chokkhu, "ConvNeXt")
    assert hasattr(chokkhu, "ViT")
    assert hasattr(chokkhu, "SwinTransformer")
    assert hasattr(chokkhu, "YOLOHead")
    assert hasattr(chokkhu, "GradCAM")
    # NLP
    assert hasattr(chokkhu, "BERT")
    assert hasattr(chokkhu, "GPT")
    assert hasattr(chokkhu, "LLaMA")
    assert hasattr(chokkhu, "Mistral")
    assert hasattr(chokkhu, "Gemma")
    assert hasattr(chokkhu, "MoE")
    assert hasattr(chokkhu, "DeepSeekMLA")
    assert hasattr(chokkhu, "BPETokenizer")
    # Audio
    assert hasattr(chokkhu, "stft")
    assert hasattr(chokkhu, "mfcc")
    assert hasattr(chokkhu, "Conformer")
    assert hasattr(chokkhu, "Wav2Vec2")
    # Generative AI
    assert hasattr(chokkhu, "VAE")
    assert hasattr(chokkhu, "VQVAE")
    assert hasattr(chokkhu, "DDPM")
    assert hasattr(chokkhu, "WGANGP")
    # GNN
    assert hasattr(chokkhu, "GCN")
    assert hasattr(chokkhu, "GAT")
    assert hasattr(chokkhu, "GraphSAGE")
    assert hasattr(chokkhu, "GIN")
    # Time Series
    assert hasattr(chokkhu, "ARIMA")
    assert hasattr(chokkhu, "NBEATS")
    assert hasattr(chokkhu, "NHITS")
    assert hasattr(chokkhu, "PatchTST")
    # Neural XAI
    assert hasattr(chokkhu, "IntegratedGradients")
    assert hasattr(chokkhu, "SmoothGrad")
    assert hasattr(chokkhu, "DeepLIFT")
    # AutoML
    assert hasattr(chokkhu, "AutoTrainer")
    assert hasattr(chokkhu, "auto_train")
    assert hasattr(chokkhu, "BayesianOptimization")
    assert hasattr(chokkhu, "Hyperband")
