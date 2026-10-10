from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from chokkhu.core.math import minimize
from chokkhu.models.base import ChokkhuModel


class NeuralNetwork(ChokkhuModel):
    """Multi-Layer Perceptron (MLP) deep learning neural network implemented from scratch in pure NumPy."""

    def __init__(
        self,
        hidden_layer_sizes: Union[Sequence[int], int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: Union[str, float] = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        task: str = "auto",
        loss: str = "auto",
        leaky_relu_alpha: float = 0.01,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        layers: Optional[Sequence[int]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> None:
        if layers is not None:
            hidden_layer_sizes = layers
        if epochs is not None:
            max_iter = epochs
        if lr is not None:
            learning_rate_init = lr
        if isinstance(learning_rate, (int, float)):
            learning_rate_init = float(learning_rate)
            learning_rate = "constant"

        if isinstance(hidden_layer_sizes, int):
            self.layers_config = [hidden_layer_sizes]
        else:
            self.layers_config = list(hidden_layer_sizes)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.learning_rate = str(learning_rate)
        self.learning_rate_init = float(learning_rate_init)
        self.power_t = float(power_t)
        self.max_iter = int(max_iter)
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = float(tol)
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = float(momentum)
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = float(validation_fraction)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.n_iter_no_change = int(n_iter_no_change)
        self.max_fun = int(max_fun)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.task = task
        self.loss = loss
        self.leaky_relu_alpha = float(leaky_relu_alpha)
        self.class_weight = class_weight

        # Aliases for backward compatibility
        self.layers = self.layers_config
        self.lr = self.learning_rate_init
        self.epochs = self.max_iter

        self.coefs_: List[np.ndarray] = []
        self.intercepts_: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 1
        self.n_features_in_: int = 0
        self.n_layers_: int = 0
        self.n_outputs_: int = 0
        self.out_activation_: str = "identity"
        self.loss_: float = 0.0
        self.loss_curve_: List[float] = []
        self.validation_scores_: Optional[List[float]] = None
        self.best_loss_: Optional[float] = None
        self.best_validation_score_: Optional[float] = None
        self.n_iter_: int = 0

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        act = self.activation
        if act == "relu":
            return np.maximum(0.0, Z)
        elif act in ("sigmoid", "logistic"):
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -30.0, 30.0)))
        elif act == "tanh":
            return np.tanh(Z)
        elif act == "gelu":
            s = np.sqrt(2.0 / np.pi)
            u = s * (Z + 0.044715 * (Z**3))
            return 0.5 * Z * (1.0 + np.tanh(u))
        elif act in ("silu", "swish"):
            sig = 1.0 / (1.0 + np.exp(-np.clip(Z, -30.0, 30.0)))
            return Z * sig
        elif act == "leaky_relu":
            return np.where(Z > 0.0, Z, self.leaky_relu_alpha * Z)
        elif act == "elu":
            return np.where(Z > 0.0, Z, np.exp(np.clip(Z, -30.0, 0.0)) - 1.0)
        return Z

    def _activate_derivative(self, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        act = self.activation
        if act == "relu":
            return (Z > 0.0).astype(np.float64)
        elif act in ("sigmoid", "logistic"):
            return A * (1.0 - A)
        elif act == "tanh":
            return 1.0 - A**2
        elif act == "gelu":
            s = np.sqrt(2.0 / np.pi)
            u = s * (Z + 0.044715 * (Z**3))
            tanh_u = np.tanh(u)
            du = s * (1.0 + 3.0 * 0.044715 * (Z**2))
            dtanh = 1.0 - tanh_u**2
            return 0.5 * (1.0 + tanh_u) + 0.5 * Z * dtanh * du
        elif act in ("silu", "swish"):
            sig = 1.0 / (1.0 + np.exp(-np.clip(Z, -30.0, 30.0)))
            return sig * (1.0 + Z * (1.0 - sig))
        elif act == "leaky_relu":
            return np.where(Z > 0.0, 1.0, self.leaky_relu_alpha)
        elif act == "elu":
            return np.where(Z > 0.0, 1.0, A + 1.0)
        return np.ones_like(Z)

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True) + 1e-15)

    def _init_weights(
        self, in_features: int, out_dim: int, rng: np.random.Generator
    ) -> None:
        layer_dims = [in_features] + self.layers_config + [out_dim]
        self.n_layers_ = len(layer_dims)
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            if self.activation in (
                "relu",
                "gelu",
                "silu",
                "swish",
                "leaky_relu",
                "elu",
            ):
                limit = np.sqrt(2.0 / fan_in)
            else:
                limit = np.sqrt(2.0 / (fan_in + fan_out))
            W: np.ndarray = rng.normal(0.0, limit, size=(fan_in, fan_out))
            b: np.ndarray = np.zeros((1, fan_out), dtype=np.float64)
            self.coefs_.append(W)
            self.intercepts_.append(b)
        self.weights = self.coefs_
        self.biases = self.intercepts_

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> NeuralNetwork:
        if y is None:
            raise ValueError("Target y cannot be None for NeuralNetwork.")
        for k in ("epochs", "max_iter", "batch_size", "lr", "learning_rate_init"):
            if k in kwargs:
                setattr(
                    self, "max_iter" if "iter" in k or "epoch" in k else k, kwargs[k]
                )

        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        n_samples, in_features = X_arr.shape
        self.n_features_in_ = in_features

        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(y_arr.dtype, np.floating)
                else "classification"
            )

        if self.task == "classification":
            self.classes_ = np.unique(y_arr)
            self.n_classes = len(self.classes_)
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_indices = np.array([class_to_idx[val] for val in y_arr])
            if self.n_classes == 2:
                Y = y_indices.reshape(-1, 1).astype(np.float64)
                out_dim = 1
                self.out_activation_ = "logistic"
            else:
                Y = np.zeros((n_samples, self.n_classes), dtype=np.float64)
                Y[np.arange(n_samples), y_indices] = 1.0
                out_dim = self.n_classes
                self.out_activation_ = "softmax"

            if self.class_weight is not None:
                if self.class_weight == "balanced":
                    counts = np.bincount(y_indices)
                    cw_map = {
                        c: n_samples / (self.n_classes * max(1, counts[i]))
                        for i, c in enumerate(self.classes_)
                    }
                elif isinstance(self.class_weight, dict):
                    cw_map = self.class_weight
                else:
                    cw_map = {c: 1.0 for c in self.classes_}
                sample_weights = np.array(
                    [cw_map.get(v, 1.0) for v in y_arr], dtype=np.float64
                )
            else:
                sample_weights = np.ones(n_samples, dtype=np.float64)
        else:
            self.classes_ = None
            self.n_classes = 1
            self.out_activation_ = "identity"
            if y_arr.ndim == 1:
                Y = y_arr.reshape(-1, 1).astype(np.float64)
            else:
                Y = y_arr.astype(np.float64)
            out_dim = Y.shape[1]
            sample_weights = np.ones(n_samples, dtype=np.float64)

        self.n_outputs_ = out_dim
        layer_dims = [in_features] + self.layers_config + [out_dim]
        if not (self.warm_start and len(self.coefs_) == len(layer_dims) - 1):
            self._init_weights(in_features, out_dim, rng)

        if self.early_stopping:
            n_val = max(1, int(n_samples * self.validation_fraction))
            perm = rng.permutation(n_samples)
            v_idx, t_idx = perm[:n_val], perm[n_val:]
            X_tr, Y_tr, sw_tr = X_arr[t_idx], Y[t_idx], sample_weights[t_idx]
            X_val, y_val = X_arr[v_idx], y_arr[v_idx]
            self.validation_scores_ = []
        else:
            X_tr, Y_tr, sw_tr = X_arr, Y, sample_weights
            X_val, y_val = None, None
            self.validation_scores_ = None

        n_train = len(X_tr)
        b_size = (
            n_train if self.batch_size == "auto" else min(int(self.batch_size), n_train)
        )
        curr_lr = self.learning_rate_init

        if self.solver == "lbfgs":
            self._fit_lbfgs(X_tr, Y_tr, sw_tr)
            return self

        # Momentum & Adaptive optimizer caches
        v_W = [np.zeros_like(W) for W in self.coefs_]
        v_b = [np.zeros_like(b) for b in self.intercepts_]
        m_W = [np.zeros_like(W) for W in self.coefs_]
        m_b = [np.zeros_like(b) for b in self.intercepts_]
        opt_step = 0
        best_score = -float("inf") if self.early_stopping else float("inf")
        best_coefs = [W.copy() for W in self.coefs_]
        best_intercepts = [b.copy() for b in self.intercepts_]
        no_improve = 0
        self.loss_curve_ = []

        for epoch in range(self.max_iter):
            if self.learning_rate == "invscaling":
                curr_lr = self.learning_rate_init / ((epoch + 1) ** self.power_t)

            perm = rng.permutation(n_train) if self.shuffle else np.arange(n_train)
            X_shuf, Y_shuf, sw_shuf = X_tr[perm], Y_tr[perm], sw_tr[perm]

            for start in range(0, n_train, b_size):
                end = min(start + b_size, n_train)
                xb, yb, swb = X_shuf[start:end], Y_shuf[start:end], sw_shuf[start:end]
                cur_bsize = end - start

                # Forward pass
                acts = [xb]
                zs: List[np.ndarray] = []
                dropout_masks: List[Optional[np.ndarray]] = []

                for idx in range(len(self.coefs_) - 1):
                    z = acts[-1] @ self.coefs_[idx] + self.intercepts_[idx]
                    zs.append(z)
                    a = self._activate(z)
                    if self.dropout > 0.0:
                        mask = (
                            rng.uniform(0.0, 1.0, size=a.shape) >= self.dropout
                        ).astype(np.float64) / (1.0 - self.dropout)
                        a = a * mask
                        dropout_masks.append(mask)
                    else:
                        dropout_masks.append(None)
                    acts.append(a)

                z_out = acts[-1] @ self.coefs_[-1] + self.intercepts_[-1]
                zs.append(z_out)

                if self.task == "classification":
                    if self.n_classes == 2:
                        a_out = 1.0 / (1.0 + np.exp(-np.clip(z_out, -30.0, 30.0)))
                        dz = (a_out - yb) * (swb[:, None] / cur_bsize)
                    else:
                        a_out = self._softmax(z_out)
                        dz = (a_out - yb) * (swb[:, None] / cur_bsize)
                else:
                    a_out = z_out
                    diff = a_out - yb
                    if self.loss == "mae" or self.loss == "absolute_error":
                        dz = np.sign(diff) * (swb[:, None] / cur_bsize)
                    elif self.loss == "huber":
                        delta = 1.0
                        dz = np.where(
                            np.abs(diff) <= delta, diff, delta * np.sign(diff)
                        ) * (swb[:, None] / cur_bsize)
                    else:
                        dz = diff * (swb[:, None] / cur_bsize)

                acts.append(a_out)

                # Backward pass
                opt_step += 1
                for idx in reversed(range(len(self.coefs_))):
                    a_prev = acts[idx]
                    dw = a_prev.T @ dz
                    db = np.sum(dz, axis=0, keepdims=True)

                    if self.solver != "adamw" and self.alpha > 0.0:
                        dw += (self.alpha / n_train) * self.coefs_[idx]

                    if idx > 0:
                        da_prev = dz @ self.coefs_[idx].T
                        if dropout_masks[idx - 1] is not None:
                            da_prev = da_prev * dropout_masks[idx - 1]
                        dz = da_prev * self._activate_derivative(zs[idx - 1], acts[idx])

                    # Parameter updates
                    if self.solver == "sgd":
                        if self.momentum > 0.0:
                            v_W[idx] = self.momentum * v_W[idx] + dw
                            v_b[idx] = self.momentum * v_b[idx] + db
                            step_w = (
                                dw + self.momentum * v_W[idx]
                                if self.nesterovs_momentum
                                else v_W[idx]
                            )
                            step_b = (
                                db + self.momentum * v_b[idx]
                                if self.nesterovs_momentum
                                else v_b[idx]
                            )
                        else:
                            step_w, step_b = dw, db
                        self.coefs_[idx] -= curr_lr * step_w
                        self.intercepts_[idx] -= curr_lr * step_b

                    elif self.solver in ("adam", "adamw"):
                        if self.solver == "adamw" and self.weight_decay > 0.0:
                            self.coefs_[idx] -= (
                                curr_lr * self.weight_decay * self.coefs_[idx]
                            )
                        m_W[idx] = self.beta_1 * m_W[idx] + (1.0 - self.beta_1) * dw
                        m_b[idx] = self.beta_1 * m_b[idx] + (1.0 - self.beta_1) * db
                        v_W[idx] = self.beta_2 * v_W[idx] + (1.0 - self.beta_2) * (
                            dw**2
                        )
                        v_b[idx] = self.beta_2 * v_b[idx] + (1.0 - self.beta_2) * (
                            db**2
                        )
                        m_w_hat = m_W[idx] / (1.0 - (self.beta_1**opt_step))
                        m_b_hat = m_b[idx] / (1.0 - (self.beta_1**opt_step))
                        v_w_hat = v_W[idx] / (1.0 - (self.beta_2**opt_step))
                        v_b_hat = v_b[idx] / (1.0 - (self.beta_2**opt_step))
                        self.coefs_[idx] -= (
                            curr_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                        )
                        self.intercepts_[idx] -= (
                            curr_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                        )

                    elif self.solver == "rmsprop":
                        v_W[idx] = self.beta_2 * v_W[idx] + (1.0 - self.beta_2) * (
                            dw**2
                        )
                        v_b[idx] = self.beta_2 * v_b[idx] + (1.0 - self.beta_2) * (
                            db**2
                        )
                        self.coefs_[idx] -= (
                            curr_lr * dw / (np.sqrt(v_W[idx]) + self.epsilon)
                        )
                        self.intercepts_[idx] -= (
                            curr_lr * db / (np.sqrt(v_b[idx]) + self.epsilon)
                        )

                    elif self.solver == "adagrad":
                        v_W[idx] += dw**2
                        v_b[idx] += db**2
                        self.coefs_[idx] -= (
                            curr_lr * dw / (np.sqrt(v_W[idx]) + self.epsilon)
                        )
                        self.intercepts_[idx] -= (
                            curr_lr * db / (np.sqrt(v_b[idx]) + self.epsilon)
                        )

            # Epoch metrics & Early stopping
            epoch_loss = self._compute_loss(X_tr, Y_tr, sw_tr)
            self.loss_curve_.append(epoch_loss)

            if self.learning_rate == "adaptive" and len(self.loss_curve_) > 1:
                if self.loss_curve_[-2] - self.loss_curve_[-1] < self.tol:
                    curr_lr = max(curr_lr / 5.0, 1e-6)

            if self.early_stopping and X_val is not None:
                val_score = self.score(X_val, y_val)
                self.validation_scores_.append(val_score)
                if val_score > best_score + self.tol:
                    best_score = val_score
                    best_coefs = [W.copy() for W in self.coefs_]
                    best_intercepts = [b.copy() for b in self.intercepts_]
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.n_iter_no_change:
                    self.coefs_ = best_coefs
                    self.intercepts_ = best_intercepts
                    break
            else:
                if (
                    len(self.loss_curve_) > 1
                    and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol
                ):
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self.n_iter_no_change:
                    break

        self.loss_ = self.loss_curve_[-1]
        self.best_loss_ = min(self.loss_curve_) if self.loss_curve_ else self.loss_
        if self.validation_scores_:
            self.best_validation_score_ = max(self.validation_scores_)
        self.n_iter_ = epoch + 1
        self.weights = self.coefs_
        self.biases = self.intercepts_
        return self

    def _compute_loss(self, X: np.ndarray, Y: np.ndarray, sw: np.ndarray) -> float:
        out = self._forward(X)
        n = len(X)
        if self.task == "classification":
            if self.n_classes == 2:
                eps = 1e-15
                loss = -np.mean(
                    sw[:, None]
                    * (
                        Y * np.log(np.clip(out, eps, 1.0))
                        + (1.0 - Y) * np.log(np.clip(1.0 - out, eps, 1.0))
                    )
                )
            else:
                eps = 1e-15
                loss = -np.mean(
                    sw[:, None]
                    * np.sum(Y * np.log(np.clip(out, eps, 1.0)), axis=1, keepdims=True)
                )
        else:
            diff = out - Y
            if self.loss == "mae" or self.loss == "absolute_error":
                loss = np.mean(sw[:, None] * np.abs(diff))
            elif self.loss == "huber":
                delta = 1.0
                h = np.where(
                    np.abs(diff) <= delta,
                    0.5 * (diff**2),
                    delta * (np.abs(diff) - 0.5 * delta),
                )
                loss = np.mean(sw[:, None] * h)
            else:
                loss = 0.5 * np.mean(sw[:, None] * (diff**2))

        reg = 0.0
        if self.alpha > 0.0:
            reg = 0.5 * (self.alpha / n) * sum(float(np.sum(w**2)) for w in self.coefs_)
        return float(loss + reg)

    def _fit_lbfgs(self, X: np.ndarray, Y: np.ndarray, sw: np.ndarray) -> None:
        shapes_w = [w.shape for w in self.coefs_]
        shapes_b = [b.shape for b in self.intercepts_]
        sizes_w = [w.size for w in self.coefs_]
        sizes_b = [b.size for b in self.intercepts_]

        def pack(coefs: List[np.ndarray], intercepts: List[np.ndarray]) -> np.ndarray:
            parts = [w.ravel() for w in coefs] + [b.ravel() for b in intercepts]
            return np.concatenate(parts)

        def unpack(theta: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            coefs: List[np.ndarray] = []
            intercepts: List[np.ndarray] = []
            cur = 0
            for s, sh in zip(sizes_w, shapes_w):
                coefs.append(theta[cur : cur + s].reshape(sh))
                cur += s
            for s, sh in zip(sizes_b, shapes_b):
                intercepts.append(theta[cur : cur + s].reshape(sh))
                cur += s
            return coefs, intercepts

        n_train = len(X)

        def obj(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            c, inter = unpack(theta)
            # forward
            acts = [X]
            zs = []
            for idx in range(len(c) - 1):
                z = acts[-1] @ c[idx] + inter[idx]
                zs.append(z)
                acts.append(self._activate(z))
            z_out = acts[-1] @ c[-1] + inter[-1]
            zs.append(z_out)

            if self.task == "classification":
                if self.n_classes == 2:
                    a_out = 1.0 / (1.0 + np.exp(-np.clip(z_out, -30.0, 30.0)))
                    dz = (a_out - Y) * (sw[:, None] / n_train)
                    loss = -np.mean(
                        sw[:, None]
                        * (
                            Y * np.log(np.clip(a_out, 1e-15, 1.0))
                            + (1.0 - Y) * np.log(np.clip(1.0 - a_out, 1e-15, 1.0))
                        )
                    )
                else:
                    a_out = self._softmax(z_out)
                    dz = (a_out - Y) * (sw[:, None] / n_train)
                    loss = -np.mean(
                        sw[:, None]
                        * np.sum(
                            Y * np.log(np.clip(a_out, 1e-15, 1.0)),
                            axis=1,
                            keepdims=True,
                        )
                    )
            else:
                a_out = z_out
                dz = (a_out - Y) * (sw[:, None] / n_train)
                loss = 0.5 * np.mean(sw[:, None] * ((a_out - Y) ** 2))

            acts.append(a_out)
            grad_w: List[np.ndarray] = []
            grad_b: List[np.ndarray] = []
            for idx in reversed(range(len(c))):
                dw = acts[idx].T @ dz + (self.alpha / n_train) * c[idx]
                db = np.sum(dz, axis=0, keepdims=True)
                grad_w.insert(0, dw)
                grad_b.insert(0, db)
                if idx > 0:
                    dz = (dz @ c[idx].T) * self._activate_derivative(
                        zs[idx - 1], acts[idx]
                    )

            reg = 0.5 * (self.alpha / n_train) * sum(float(np.sum(w**2)) for w in c)
            return float(loss + reg), pack(grad_w, grad_b)

        init_theta = pack(self.coefs_, self.intercepts_)
        res = minimize(
            fun=lambda th: obj(th)[0],
            x0=init_theta,
            jac=lambda th: obj(th)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.coefs_, self.intercepts_ = unpack(res.x)
        self.weights = self.coefs_
        self.biases = self.intercepts_
        self.loss_ = float(res.fun)
        self.loss_curve_ = [self.loss_]
        self.n_iter_ = getattr(res, "nit", self.max_iter)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        for idx in range(len(self.coefs_) - 1):
            Z = A @ self.coefs_[idx] + self.intercepts_[idx]
            A = self._activate(Z)
        Z_out = A @ self.coefs_[-1] + self.intercepts_[-1]
        if self.task == "classification":
            if self.n_classes == 2:
                return 1.0 / (1.0 + np.exp(-np.clip(Z_out, -30.0, 30.0)))
            else:
                return self._softmax(Z_out)
        return Z_out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        probs = self._forward(X)
        if self.n_classes == 2:
            return np.hstack([1.0 - probs, probs])
        return probs

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1.0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward(X)
        if self.task == "classification":
            if self.classes_ is None:
                return (out >= 0.5).astype(int).flatten()
            if self.n_classes == 2:
                preds = (out.flatten() >= 0.5).astype(int)
                return self.classes_[preds]
            else:
                idx = np.argmax(out, axis=1)
                return self.classes_[idx]
        return out.flatten() if out.shape[1] == 1 else out

    def score(
        self, X: Any, y: Any = None, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        preds = np.asarray(self.predict(X))
        if self.task == "classification":
            y_arr = np.asarray(y)
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=np.float64)
                correct = (preds == y_arr).astype(np.float64)
                return float(
                    float(np.sum(correct * sw)) / max(1e-12, float(np.sum(sw)))
                )
            return float(np.mean(preds == y_arr))
        else:
            y_arr = np.asarray(y, dtype=np.float64)
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=np.float64)
                ss_res = float(np.sum(sw * ((y_arr - preds) ** 2)))
                avg_y = float(np.sum(sw * y_arr)) / max(1e-12, float(np.sum(sw)))
                ss_tot = float(np.sum(sw * ((y_arr - avg_y) ** 2)))
            else:
                ss_res = float(np.sum((y_arr - preds) ** 2))
                ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
            if ss_tot == 0.0:
                return 1.0 if ss_res == 0.0 else 0.0
            return float(1.0 - (ss_res / ss_tot))


class MLPClassifier(NeuralNetwork):
    """Multi-Layer Perceptron Classifier in pure NumPy."""

    def __init__(
        self,
        hidden_layer_sizes: Union[Sequence[int], int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: Union[str, float] = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        loss: str = "log_loss",
        leaky_relu_alpha: float = 0.01,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        layers: Optional[Sequence[int]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
            weight_decay=weight_decay,
            dropout=dropout,
            task="classification",
            loss=loss,
            leaky_relu_alpha=leaky_relu_alpha,
            class_weight=class_weight,
            layers=layers,
            epochs=epochs,
            lr=lr,
        )


class MLPRegressor(NeuralNetwork):
    """Multi-Layer Perceptron Regressor in pure NumPy."""

    def __init__(
        self,
        hidden_layer_sizes: Union[Sequence[int], int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: Union[str, float] = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        loss: str = "squared_error",
        leaky_relu_alpha: float = 0.01,
        layers: Optional[Sequence[int]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
            weight_decay=weight_decay,
            dropout=dropout,
            task="regression",
            loss=loss,
            leaky_relu_alpha=leaky_relu_alpha,
            layers=layers,
            epochs=epochs,
            lr=lr,
        )
