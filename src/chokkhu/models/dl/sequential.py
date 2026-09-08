from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger
from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from .callbacks import Callback
from .layers import Module
from .losses import CrossEntropyLoss, Loss, MSELoss
from .optimizers import Adam, Optimizer


class Sequential(Module, ChokkhuModel):
    """Sequential Neural Network Container."""

    def __init__(
        self, layers: Optional[List[Module]] = None, task: str = "classification"
    ) -> None:
        super().__init__()
        self.task = task
        self.layers_list: List[Module] = []
        if layers:
            for lyr in layers:
                self.add(lyr)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_score": [],
            "val_score": [],
        }

    def add(self, layer: Module) -> None:
        self.layers_list.append(layer)
        setattr(self, f"layer_{len(self.layers_list)}", layer)

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)
        out = x
        for lyr in self.layers_list:
            out = lyr(out)
        return out

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Loss] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True,
    ) -> Sequential:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_arr = y.values
            else:
                y_arr = np.asarray(y)
        else:
            y_arr = None

        if X_val is not None:
            X_val_arr = (
                X_val.values
                if isinstance(X_val, (pd.DataFrame, pd.Series))
                else np.asarray(X_val, dtype=np.float64)
            )
            y_val_arr = (
                y_val.values
                if isinstance(y_val, (pd.DataFrame, pd.Series))
                else np.asarray(y_val)
            )
        else:
            X_val_arr, y_val_arr = None, None

        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=lr)

        if loss_fn is None:
            if self.task == "classification":
                loss_fn = CrossEntropyLoss()
            else:
                loss_fn = MSELoss()

        n_samples = len(X_arr)
        callbacks = callbacks or []

        for epoch in range(1, epochs + 1):
            self.train(True)
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            n_batches = int(np.ceil(n_samples / batch_size))

            for b in range(n_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                x_b = Tensor(X_arr[batch_idx], requires_grad=False)
                y_b = Tensor(
                    y_arr[batch_idx] if y_arr is not None else X_arr[batch_idx],
                    requires_grad=False,
                )

                optimizer.zero_grad()
                pred = self.forward(x_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.data)

            avg_loss = total_loss / max(1, n_batches)
            self.history["train_loss"].append(avg_loss)

            logs = {"train_loss": avg_loss}
            if X_val_arr is not None and y_val_arr is not None:
                self.eval()
                x_val_t = Tensor(X_val_arr, requires_grad=False)
                y_val_t = Tensor(y_val_arr, requires_grad=False)
                val_pred = self.forward(x_val_t)
                val_l = float(loss_fn(val_pred, y_val_t).data)
                self.history["val_loss"].append(val_l)
                logs["val_loss"] = val_l

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

            if any(getattr(cb, "stop_training", False) for cb in callbacks):
                if verbose:
                    Logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self.eval()
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X_t = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X_t).data
        if self.task == "classification":
            if out.ndim == 2 and out.shape[1] > 1:
                return np.argmax(out, axis=1)
            elif out.ndim == 2 and out.shape[1] == 1:
                return (out.flatten() > 0.5).astype(int)
            return (out > 0.5).astype(int)
        return out.flatten() if out.ndim == 2 and out.shape[1] == 1 else out

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self.eval()
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X_t = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X_t).data
        if out.ndim == 2 and out.shape[1] > 1:
            max_out = np.max(out, axis=1, keepdims=True)
            exp_out = np.exp(out - max_out)
            return exp_out / np.sum(exp_out, axis=1, keepdims=True)
        elif out.ndim == 2 and out.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-np.clip(out, -500.0, 500.0)))
            return np.hstack([1.0 - p, p])
        p = 1.0 / (1.0 + np.exp(-np.clip(out, -500.0, 500.0))).reshape(-1, 1)
        return np.hstack([1.0 - p, p])
