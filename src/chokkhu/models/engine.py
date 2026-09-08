from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger
from .base import ChokkhuModel
from .ml import (
    DBSCAN,
    KNN,
    SVM,
    DecisionTree,
    GradientBoosting,
    HierarchicalClustering,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    NeuralNetwork,
    RandomForest,
)
from .rl import QLearning


def _get_default_param_grid(model: str) -> Dict[str, List[Any]]:
    if model in ("random_forest", "rf"):
        return {
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        }
    elif model in ("decision_tree", "dt"):
        return {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
        }
    elif model == "knn":
        return {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        }
    elif model == "svm":
        return {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
        }
    elif model in ("gradient_boosting", "gbm"):
        return {
            "n_estimators": [20, 50],
            "learning_rate": [0.05, 0.1],
        }
    elif model in (
        "linear_regression",
        "logistic_regression",
        "ridge",
        "lasso",
        "elastic_net",
    ):
        return {
            "learning_rate": [0.001, 0.01, 0.1],
        }
    return {}


def _create_model_instance(
    model: str,
    task: str,
    random_state: Optional[int],
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> ChokkhuModel:
    in_c = kwargs.get(
        "in_channels",
        (X_train.shape[1] if X_train is not None and X_train.ndim == 4 else 3),
    )
    num_c = kwargs.get(
        "num_classes", (len(np.unique(y_train)) if y_train is not None else 10)
    )

    if model == "linear_regression":
        return LinearRegression(**kwargs)
    elif model == "ridge":
        return LinearRegression(
            method="gradient_descent", regularization="ridge", **kwargs
        )
    elif model == "lasso":
        return LinearRegression(
            method="gradient_descent", regularization="lasso", **kwargs
        )
    elif model == "elastic_net":
        return LinearRegression(
            method="gradient_descent", regularization="elastic_net", **kwargs
        )
    elif model == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif model == "knn":
        return KNN(task=task if task != "auto" else "classification", **kwargs)
    elif model == "naive_bayes":
        return NaiveBayes(**kwargs)
    elif model == "kmeans":
        return KMeans(random_state=random_state, **kwargs)
    elif model == "dbscan":
        return DBSCAN(**kwargs)
    elif model == "hierarchical":
        return HierarchicalClustering(**kwargs)
    elif model == "svm":
        return SVM(**kwargs)
    elif model == "decision_tree":
        return DecisionTree(task=task if task != "auto" else "classification", **kwargs)
    elif model in ("random_forest", "rf"):
        return RandomForest(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model in ("gradient_boosting", "gbm"):
        return GradientBoosting(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model in ("neural_network", "mlp", "dense"):
        return NeuralNetwork(
            task=task,
            random_state=random_state,
            **kwargs,
        )
    # --- Vision Universe ---
    elif model in ("lenet", "lenet5"):
        from .vision import LeNet5

        in_c_lenet = kwargs.get(
            "in_channels",
            (X_train.shape[1] if X_train is not None and X_train.ndim == 4 else 1),
        )
        return LeNet5(num_classes=num_c, in_channels=in_c_lenet)
    elif model in ("alexnet",):
        from .vision import AlexNet

        return AlexNet(num_classes=num_c, in_channels=in_c)
    elif model in ("zfnet",):
        from .vision import ZFNet

        return ZFNet(num_classes=num_c, in_channels=in_c)
    elif model in ("vgg11",):
        from .vision import VGG11

        return VGG11(num_classes=num_c, in_channels=in_c)
    elif model in ("vgg13",):
        from .vision import VGG13

        return VGG13(num_classes=num_c, in_channels=in_c)
    elif model in ("vgg", "vgg16"):
        from .vision import VGG16

        return VGG16(num_classes=num_c, in_channels=in_c)
    elif model in ("vgg19",):
        from .vision import VGG19

        return VGG19(num_classes=num_c, in_channels=in_c)
    elif model in ("googlenet", "inception", "inception_v1", "inceptionv1"):
        from .vision import GoogLeNet

        return GoogLeNet(num_classes=num_c, in_channels=in_c)
    elif model in ("inception_v3", "inceptionv3"):
        from .vision import InceptionV3

        return InceptionV3(num_classes=num_c, in_channels=in_c)
    elif model in ("resnet", "resnet18"):
        from .vision import ResNet18

        return ResNet18(num_classes=num_c, in_channels=in_c)
    elif model in ("resnet34",):
        from .vision import ResNet34

        return ResNet34(num_classes=num_c, in_channels=in_c)
    elif model in ("resnet50",):
        from .vision import ResNet50

        return ResNet50(num_classes=num_c, in_channels=in_c)
    elif model in ("resnet101",):
        from .vision import ResNet101

        return ResNet101(num_classes=num_c, in_channels=in_c)
    elif model in ("resnet152",):
        from .vision import ResNet152

        return ResNet152(num_classes=num_c, in_channels=in_c)
    elif model in ("resnext", "resnext50"):
        from .vision import ResNeXt50

        return ResNeXt50(num_classes=num_c, in_channels=in_c)
    elif model in ("resnext101",):
        from .vision import ResNeXt101

        return ResNeXt101(num_classes=num_c, in_channels=in_c)
    elif model in ("densenet", "densenet121"):
        from .vision import DenseNet121

        return DenseNet121(num_classes=num_c, in_channels=in_c)
    elif model in ("densenet169",):
        from .vision import DenseNet169

        return DenseNet169(num_classes=num_c, in_channels=in_c)
    elif model in ("densenet201",):
        from .vision import DenseNet201

        return DenseNet201(num_classes=num_c, in_channels=in_c)
    elif model in ("squeezenet",):
        from .vision import SqueezeNet

        return SqueezeNet(num_classes=num_c, in_channels=in_c)
    elif model in ("mobilenet", "mobilenet_v1", "mobilenetv1"):
        from .vision import MobileNetV1

        return MobileNetV1(num_classes=num_c, in_channels=in_c)
    elif model in ("mobilenet_v2", "mobilenetv2"):
        from .vision import MobileNetV2

        return MobileNetV2(num_classes=num_c, in_channels=in_c)
    elif model in ("mobilenet_v3", "mobilenetv3"):
        from .vision import MobileNetV3

        return MobileNetV3(num_classes=num_c, in_channels=in_c)
    elif model in ("shufflenet", "shufflenet_v1", "shufflenet_v2", "shufflenetv2"):
        from .vision import ShuffleNetV2

        return ShuffleNetV2(num_classes=num_c, in_channels=in_c)
    elif model in ("efficientnet", "efficientnet_b0", "efficientnetb0"):
        from .vision import EfficientNetB0

        return EfficientNetB0(num_classes=num_c, in_channels=in_c)
    elif model in ("convnext", "convnext_tiny", "convnexttiny"):
        from .vision import ConvNeXtTiny

        return ConvNeXtTiny(num_classes=num_c, in_channels=in_c)
    elif model in ("unet",):
        from .vision import UNet

        out_c = kwargs.get("out_channels", 1)
        return UNet(in_channels=in_c, out_channels=out_c)
    elif model in ("fcn", "fcn8s"):
        from .vision import FCN8s

        return FCN8s(num_classes=num_c, in_channels=in_c)
    elif model in ("vit", "vision_transformer", "vit_tiny", "vittiny"):
        from .vision import VisionTransformer

        return VisionTransformer(in_channels=in_c, num_classes=num_c)
    # --- NLP & Sovereign Transformer Universe ---
    elif model in ("bert", "bert_classification", "bert_classifier"):
        from .nlp import BertForSequenceClassification

        vocab_s = kwargs.pop("vocab_size", 30522)
        n_c = kwargs.pop("num_classes", num_c)
        return BertForSequenceClassification(
            vocab_size=vocab_s, num_classes=n_c, **kwargs
        )
    elif model in ("bert_mlm", "bert_masked_lm"):
        from .nlp import BertForMaskedLM

        vocab_s = kwargs.pop("vocab_size", 30522)
        return BertForMaskedLM(vocab_size=vocab_s, **kwargs)
    elif model in ("gpt", "minigpt", "gpt_causal_lm", "gpt2"):
        from .nlp import GPTForCausalLM

        vocab_s = kwargs.pop("vocab_size", 50257)
        return GPTForCausalLM(vocab_size=vocab_s, **kwargs)
    elif model in ("llama", "llama2", "llama3", "mistral"):
        from .nlp import LLaMA

        vocab_s = kwargs.pop("vocab_size", 32000)
        return LLaMA(vocab_size=vocab_s, **kwargs)
    elif model in ("seq2seq", "t5", "seq2seq_transformer"):
        from .nlp import Seq2SeqTransformer

        src_v = kwargs.pop("src_vocab_size", 10000)
        tgt_v = kwargs.pop("tgt_vocab_size", 10000)
        return Seq2SeqTransformer(src_vocab_size=src_v, tgt_vocab_size=tgt_v, **kwargs)
    elif model in ("rnn", "elman_rnn"):
        from .nlp import RNN

        in_d = kwargs.pop(
            "input_dim", (X_train.shape[-1] if X_train is not None else 64)
        )
        h_d = kwargs.pop("hidden_dim", 128)
        return RNN(input_dim=in_d, hidden_dim=h_d, **kwargs)
    elif model in ("lstm",):
        from .nlp import LSTM

        in_d = kwargs.pop(
            "input_dim", (X_train.shape[-1] if X_train is not None else 64)
        )
        h_d = kwargs.pop("hidden_dim", 128)
        return LSTM(input_dim=in_d, hidden_dim=h_d, **kwargs)
    elif model in ("bilstm", "bi_lstm"):
        from .nlp import BiLSTM

        in_d = kwargs.pop(
            "input_dim", (X_train.shape[-1] if X_train is not None else 64)
        )
        h_d = kwargs.pop("hidden_dim", 128)
        return BiLSTM(input_dim=in_d, hidden_dim=h_d, **kwargs)
    elif model in ("gru",):
        from .nlp import GRU

        in_d = kwargs.pop(
            "input_dim", (X_train.shape[-1] if X_train is not None else 64)
        )
        h_d = kwargs.pop("hidden_dim", 128)
        return GRU(input_dim=in_d, hidden_dim=h_d, **kwargs)
    elif model in ("sequential", "cnn"):
        from .dl import Sequential

        return Sequential(task=task if task != "auto" else "classification", **kwargs)
    elif model == "q_learning":
        return QLearning(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Model {model} is not supported yet.")


def train(
    model: str,
    X_train: Any = None,
    y_train: Any = None,
    task: str = "auto",
    random_state: int | None = None,
    tune: bool = False,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 3,
    verbose: bool = True,
    **kwargs: Any,
) -> ChokkhuModel:
    if verbose:
        Logger.info(f"Training model: {model} (task: {task})")

    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        X_train = X_train.values
    if y_train is not None and isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train = y_train.values

    if X_train is not None and not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if y_train is not None and not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)

    fit_param_names = {
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "loss_fn",
        "X_val",
        "y_val",
        "callbacks",
    }
    fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_param_names}
    init_kwargs = {k: v for k, v in kwargs.items() if k not in fit_param_names}

    best_params: Dict[str, Any] = {}
    if tune and X_train is not None and y_train is not None:
        grid = param_grid or _get_default_param_grid(model)
        if grid:
            if verbose:
                Logger.info(
                    "Hyperparameter tuning enabled. Evaluating candidate configurations..."
                )
            keys = list(grid.keys())
            combinations = [
                dict(zip(keys, v)) for v in itertools.product(*grid.values())
            ]

            n_samples = len(X_train)
            indices = np.arange(n_samples)
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

            best_score = float("-inf")
            best_config = init_kwargs

            fold_size = max(1, n_samples // cv)
            for cfg in combinations:
                merged_kwargs = {**init_kwargs, **cfg}
                scores = []
                for f_idx in range(cv):
                    val_idx = indices[f_idx * fold_size : (f_idx + 1) * fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)

                    m = _create_model_instance(
                        model,
                        task,
                        random_state,
                        X_train=X_train,
                        y_train=y_train,
                        **merged_kwargs,
                    )
                    m.fit(X_train[tr_idx], y_train[tr_idx])
                    preds = m.predict(X_train[val_idx])

                    if task == "regression" or np.issubdtype(
                        y_train.dtype, np.floating
                    ):
                        ss_res: float = float(np.sum((y_train[val_idx] - preds) ** 2))
                        ss_tot: float = float(
                            np.sum((y_train[val_idx] - np.mean(y_train[val_idx])) ** 2)
                        )
                        score = 1.0 - (ss_res / max(1e-9, ss_tot))
                    else:
                        score = float(np.mean(y_train[val_idx] == preds))
                    scores.append(score)

                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = merged_kwargs
                    best_params = cfg

            init_kwargs = best_config
            if verbose:
                Logger.info(
                    f"Tuning complete. Best Score: {best_score:.4f}, Best Params: {best_params}"
                )

    model_obj = _create_model_instance(
        model, task, random_state, X_train=X_train, y_train=y_train, **init_kwargs
    )

    try:
        model_obj.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model_obj.fit(X_train, y_train)

    if best_params:
        setattr(model_obj, "best_params_", best_params)

    if verbose:
        Logger.info(f"Successfully trained {model}")

    return model_obj
