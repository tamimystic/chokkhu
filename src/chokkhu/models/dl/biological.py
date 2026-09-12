"""Biologically Plausible Learning & Direct Feedback Alignment in Pure NumPy.

References:
- Lillicrap et al. (2016): "Random synaptic feedback weights support error backpropagation for deep learning" (Nature Comms).
- Nøkland (2016): "Direct Feedback Alignment Provides Learning in Deep Neural Networks" (NeurIPS 2016).
- Lee et al. (2015): "Difference Target Propagation" (ECML-PKDD 2015).
- Jaderberg et al. (2017): "Decoupled Neural Interfaces using Synthetic Gradients" (ICML 2017).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class DirectFeedbackAlignmentNetwork:
    """Direct Feedback Alignment (DFA) Neural Network (Nøkland, 2016).

    Eliminates the biological weight transport problem by transmitting the top-layer error vector
    e = y_pred - y_true directly to each intermediate hidden layer via fixed, random feedback matrices B_l.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = "tanh",
        lr: float = 0.01,
        random_state: int = 42,
    ) -> None:
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least [in_dim, out_dim]")

        self.layer_dims = [int(d) for d in layer_dims]
        self.num_layers = len(self.layer_dims) - 1
        self.activation = activation
        self.lr = float(lr)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Forward weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for layer_idx in range(self.num_layers):
            fan_in = self.layer_dims[layer_idx]
            fan_out = self.layer_dims[layer_idx + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(
                np.float64
            )
            b: np.ndarray = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

        # Fixed random feedback matrices B_l from output dimension to each hidden layer
        out_dim = self.layer_dims[-1]
        self.feedback_matrices: List[np.ndarray] = []
        for layer_idx in range(self.num_layers - 1):
            hidden_dim = self.layer_dims[layer_idx + 1]
            # B_l has shape (out_dim, hidden_dim)
            limit_fb = np.sqrt(6.0 / (out_dim + hidden_dim))
            b_mat = self.rng.uniform(
                -limit_fb, limit_fb, size=(out_dim, hidden_dim)
            ).astype(np.float64)
            self.feedback_matrices.append(b_mat)

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0.0, z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        return z

    def _act_deriv(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Derivative of activation with respect to pre-activation z."""
        if self.activation == "tanh":
            return 1.0 - a**2
        elif self.activation == "relu":
            return (z > 0.0).astype(np.float64)
        elif self.activation == "sigmoid":
            return a * (1.0 - a)
        return np.ones_like(a)

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass recording activations and pre-activations."""
        curr_h = np.asarray(x, dtype=np.float64)
        activations = [curr_h]
        pre_activations = []

        for layer_idx in range(self.num_layers):
            z = curr_h @ self.weights[layer_idx] + self.biases[layer_idx]
            pre_activations.append(z)
            if layer_idx == self.num_layers - 1:
                # Linear output layer
                curr_h = z
            else:
                curr_h = self._act(z)
            activations.append(curr_h)

        return curr_h, activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute network predictions."""
        out, _, _ = self.forward(x)
        return out

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Execute one DFA parameter update without backpropagating through forward weights."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        N = x_arr.shape[0]

        y_pred, activations, pre_activations = self.forward(x_arr)
        loss = float(0.5 * np.mean((y_pred - y_arr) ** 2))

        # Output error: e = (y_pred - y) / N of shape (N, out_dim)
        top_error = (y_pred - y_arr) / float(N)

        # Output layer update (direct delta = top_error)
        grad_w_last = activations[-2].T @ top_error
        grad_b_last = np.sum(top_error, axis=0, keepdims=True)

        self.weights[-1] -= self.lr * grad_w_last
        self.biases[-1] -= self.lr * grad_b_last

        # Hidden layer DFA updates: delta_l = (top_error @ B_l) * sigma'(z_l)
        for layer_idx in range(self.num_layers - 1):
            B_l = self.feedback_matrices[layer_idx]  # (out_dim, hidden_dim)
            proj_error = top_error @ B_l  # (N, hidden_dim)
            act_l = activations[layer_idx + 1]
            z_l = pre_activations[layer_idx]

            delta_l = proj_error * self._act_deriv(act_l, z_l)  # (N, hidden_dim)

            grad_w = activations[layer_idx].T @ delta_l
            grad_b = np.sum(delta_l, axis=0, keepdims=True)

            self.weights[layer_idx] -= self.lr * grad_w
            self.biases[layer_idx] -= self.lr * grad_b

        return loss


class DifferenceTargetPropagationNetwork:
    """Difference Target Propagation (DTP) Network (Lee et al., 2015).

    Replaces backpropagated gradients with layer-wise inverse targets:
        hat{h}_{l-1} = h_{l-1} - g_l(h_l) + g_l(hat{h}_l)
    where g_l is a learned inverse feedback mapping.
    """

    def __init__(
        self,
        layer_dims: List[int],
        lr_forward: float = 0.01,
        lr_feedback: float = 0.01,
        noise_std: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.layer_dims = [int(d) for d in layer_dims]
        self.num_layers = len(self.layer_dims) - 1
        self.lr_forward = float(lr_forward)
        self.lr_feedback = float(lr_feedback)
        self.noise_std = float(noise_std)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Forward parameters: f_l(h_{l-1}) = tanh(h_{l-1} W_l + b_l)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Inverse feedback parameters: g_l(h_l) = tanh(h_l V_l + c_l)
        self.fb_weights: List[np.ndarray] = []
        self.fb_biases: List[np.ndarray] = []

        for layer_idx in range(self.num_layers):
            fan_in = self.layer_dims[layer_idx]
            fan_out = self.layer_dims[layer_idx + 1]

            limit_f = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit_f, limit_f, size=(fan_in, fan_out)).astype(
                np.float64
            )
            b: np.ndarray = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

            # Feedback mapping from fan_out -> fan_in
            v = self.rng.uniform(-limit_f, limit_f, size=(fan_out, fan_in)).astype(
                np.float64
            )
            c: np.ndarray = np.zeros((1, fan_in), dtype=np.float64)
            self.fb_weights.append(v)
            self.fb_biases.append(c)

    def _f_step(self, h_prev: np.ndarray, layer_idx: int) -> np.ndarray:
        """Forward mapping for layer l."""
        z = h_prev @ self.weights[layer_idx] + self.biases[layer_idx]
        return np.tanh(z) if layer_idx < self.num_layers - 1 else z

    def _g_step(self, h_curr: np.ndarray, layer_idx: int) -> np.ndarray:
        """Learned inverse mapping for layer l."""
        z = h_curr @ self.fb_weights[layer_idx] + self.fb_biases[layer_idx]
        return np.tanh(z)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute forward activations."""
        curr_h = np.asarray(x, dtype=np.float64)
        activations = [curr_h]
        for layer_idx in range(self.num_layers):
            curr_h = self._f_step(curr_h, layer_idx)
            activations.append(curr_h)
        return curr_h, activations

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform one step of Difference Target Propagation."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        N = x_arr.shape[0]

        y_pred, activations = self.forward(x_arr)
        loss = float(0.5 * np.mean((y_pred - y_arr) ** 2))

        # 1. Output target
        targets: List[Optional[np.ndarray]] = [None] * (self.num_layers + 1)
        targets[-1] = y_arr

        # 2. Backpropagate difference targets backward: hat{h}_{l-1} = h_{l-1} - g_l(h_l) + g_l(hat{h}_l)
        for layer_idx in range(self.num_layers - 1, 0, -1):
            h_prev = activations[layer_idx]
            h_curr = activations[layer_idx + 1]
            hat_h_curr = targets[layer_idx + 1]

            assert hat_h_curr is not None
            g_h_curr = self._g_step(h_curr, layer_idx)
            g_hat_h = self._g_step(hat_h_curr, layer_idx)

            hat_h_prev = h_prev - g_h_curr + g_hat_h
            targets[layer_idx] = hat_h_prev

        # 3. Update forward weights to match targets: min ||f_l(h_{l-1}) - hat{h}_l||^2
        for layer_idx in range(self.num_layers):
            h_prev = activations[layer_idx]
            target_l = targets[layer_idx + 1]
            assert target_l is not None

            # Gradient on forward error with tanh derivative
            pred_l = self._f_step(h_prev, layer_idx)
            err_f = (pred_l - target_l) / float(N)
            if layer_idx < self.num_layers - 1:
                err_f = err_f * (1.0 - pred_l**2)

            self.weights[layer_idx] -= self.lr_forward * (h_prev.T @ err_f)
            self.biases[layer_idx] -= self.lr_forward * np.sum(
                err_f, axis=0, keepdims=True
            )

        # 4. Train inverse feedback mappings with exploratory noise: min ||g_l(f_l(h + eps)) - (h + eps)||^2
        for layer_idx in range(self.num_layers):
            h_prev = activations[layer_idx]
            noise = self.rng.randn(*h_prev.shape) * self.noise_std
            h_perturbed = h_prev + noise

            f_perturbed = self._f_step(h_perturbed, layer_idx)
            reconstructed_h = self._g_step(f_perturbed, layer_idx)

            err_g = (reconstructed_h - h_perturbed) / float(N)
            err_g = err_g * (1.0 - reconstructed_h**2)

            self.fb_weights[layer_idx] -= self.lr_feedback * (f_perturbed.T @ err_g)
            self.fb_biases[layer_idx] -= self.lr_feedback * np.sum(
                err_g, axis=0, keepdims=True
            )

        return loss


class DecoupledSyntheticGradientLayer:
    """Decoupled Neural Interface (DNI) with Synthetic Gradient Prediction (Jaderberg et al., 2017)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lr: float = 0.01,
        lr_synth: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.lr = float(lr)
        self.lr_synth = float(lr_synth)
        self.rng = np.random.RandomState(random_state)

        # Main layer forward parameters
        limit_f = np.sqrt(6.0 / (in_dim + out_dim))
        self.W: np.ndarray = self.rng.uniform(
            -limit_f, limit_f, size=(in_dim, out_dim)
        ).astype(np.float64)
        self.b: np.ndarray = np.zeros((1, out_dim), dtype=np.float64)

        # Gradient synthesizer: M(h) = h @ W_synth + b_synth -> predicts delta h
        limit_s = np.sqrt(6.0 / (out_dim + out_dim))
        self.W_synth: np.ndarray = self.rng.uniform(
            -limit_s, limit_s, size=(out_dim, out_dim)
        ).astype(np.float64)
        self.b_synth: np.ndarray = np.zeros((1, out_dim), dtype=np.float64)

    def forward(self, h_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass computing layer activations and predicting synthetic gradients."""
        h_arr = np.asarray(h_in, dtype=np.float64)
        z = h_arr @ self.W + self.b
        h_out = np.tanh(z)
        # Predict synthetic gradient delta h_out
        synthetic_grad = h_out @ self.W_synth + self.b_synth
        return h_out, synthetic_grad

    def update_with_synthetic_grad(
        self, h_in: np.ndarray, h_out: np.ndarray, synth_grad: np.ndarray
    ) -> None:
        """Asynchronously update layer weights immediately using synthetic gradient."""
        N = h_in.shape[0]
        # Delta z = synth_grad * (1 - h_out^2)
        dz = synth_grad * (1.0 - h_out**2)
        gw = (h_in.T @ dz) / float(N)
        gb = np.sum(dz, axis=0, keepdims=True) / float(N)

        self.W -= self.lr * gw
        self.b -= self.lr * gb

    def update_synthesizer(self, h_out: np.ndarray, target_grad: np.ndarray) -> float:
        """Train gradient synthesizer to match true or downstream gradient."""
        N = h_out.shape[0]
        pred_grad = h_out @ self.W_synth + self.b_synth
        err = (pred_grad - target_grad) / float(N)
        synth_loss = float(np.mean((pred_grad - target_grad) ** 2))

        gw_s = h_out.T @ err
        gb_s = np.sum(err, axis=0, keepdims=True)

        self.W_synth -= self.lr_synth * gw_s
        self.b_synth -= self.lr_synth * gb_s
        return synth_loss
