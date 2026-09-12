"""Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire Neurons in Pure NumPy.

References:
- Gerstner & Kistler (2002): "Spiking Neuron Models" (Cambridge Univ Press).
- Neftci et al. (2019): "Surrogate Gradient Learning in Spiking Neural Networks" (IEEE SPM).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class LIFNeuron:
    """Leaky Integrate-and-Fire (LIF) Spiking Neuron with Surrogate Gradient Backpropagation."""

    def __init__(
        self,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_scale: float = 5.0,
    ) -> None:
        self.tau_m = float(tau_m)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.decay = float(np.exp(-1.0 / self.tau_m))
        self.surrogate_scale = float(surrogate_scale)

    def surrogate_gradient(self, v: np.ndarray) -> np.ndarray:
        """Fast sigmoid surrogate derivative: dS/dV = 1 / (1 + k|V - V_th|)^2."""
        diff = np.abs(v - self.v_threshold)
        return 1.0 / ((1.0 + self.surrogate_scale * diff) ** 2)


class SpikingNeuralNetwork:
    """Multi-Layer Feedforward Spiking Neural Network (SNN) with BPTT Surrogate Gradients."""

    def __init__(
        self,
        layer_sizes: List[int],
        num_classes: int = 2,
        time_steps: int = 10,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.layer_sizes = [int(s) for s in layer_sizes]
        self.num_classes = int(num_classes)
        self.time_steps = int(time_steps)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.lif = LIFNeuron(tau_m=tau_m, v_threshold=v_threshold)
        self.num_layers = len(self.layer_sizes) - 1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for layer_idx in range(self.num_layers):
            in_d = self.layer_sizes[layer_idx]
            out_d = self.layer_sizes[layer_idx + 1]
            scale = np.sqrt(2.0 / in_d)
            w: np.ndarray = self.rng.randn(in_d, out_d) * scale
            b: np.ndarray = np.zeros(out_d, dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    def forward(
        self, X: np.ndarray, time_steps: Optional[int] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Simulate SNN dynamics over T time steps.

        Returns:
            mean_firing_rates: [N, C] output spike rates averaged over time.
            all_spikes: List of spike arrays per layer of shape [T, N, D_l].
            all_voltages: List of membrane voltage arrays per layer of shape [T, N, D_l].
        """
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        T = int(time_steps) if time_steps is not None else self.time_steps

        all_spikes: List[np.ndarray] = [
            np.zeros((T, n_samples, self.layer_sizes[layer_idx + 1]), dtype=np.float64)
            for layer_idx in range(self.num_layers)
        ]
        all_voltages: List[np.ndarray] = [
            np.zeros((T, n_samples, self.layer_sizes[layer_idx + 1]), dtype=np.float64)
            for layer_idx in range(self.num_layers)
        ]

        # Initial membrane potentials
        v_mem: List[np.ndarray] = [
            np.zeros((n_samples, self.layer_sizes[layer_idx + 1]), dtype=np.float64)
            for layer_idx in range(self.num_layers)
        ]

        for t in range(T):
            curr_input = x_arr  # Static input injected at each step

            for layer_idx in range(self.num_layers):
                W = self.weights[layer_idx]
                b = self.biases[layer_idx]

                # Synaptic current
                synaptic_in = np.dot(curr_input, W) + b

                # Subthreshold decay + synaptic current
                v_mem[layer_idx] = self.lif.decay * v_mem[layer_idx] + synaptic_in

                # Spike generation: S = 1 if V >= V_th
                spike = (v_mem[layer_idx] >= self.lif.v_threshold).astype(np.float64)

                # Reset membrane potential
                v_mem[layer_idx] = v_mem[layer_idx] - spike * self.lif.v_threshold

                all_voltages[layer_idx][t] = v_mem[layer_idx]
                all_spikes[layer_idx][t] = spike
                curr_input = spike

        # Output layer firing rate across time
        output_rates: np.ndarray = np.mean(all_spikes[-1], axis=0)
        return output_rates, all_spikes, all_voltages

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> List[float]:
        """Train SNN using BPTT with surrogate gradients."""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]

        history: List[float] = []

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            epoch_loss: List[float] = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                xb = x_arr[batch_idx]
                yb = y_arr[batch_idx]
                bs = len(xb)
                if bs == 0:
                    continue

                out_rates, spikes_hist, volt_hist = self.forward(xb)

                # Softmax over output firing rates
                shifted = out_rates - np.max(out_rates, axis=-1, keepdims=True)
                exp_vals = np.exp(shifted)
                probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(bs), yb] = 1.0
                loss = float(
                    -np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-12)), axis=1))
                )
                epoch_loss.append(loss)

                # Top-level derivative
                d_out_rates = (probs - one_hot) / float(bs)

                # Gradient accumulation across time steps
                d_W = [np.zeros_like(w) for w in self.weights]
                d_b = [np.zeros_like(b) for b in self.biases]

                # BPTT through layers and time
                d_spike_last = np.tile(
                    d_out_rates / float(self.time_steps), (self.time_steps, 1, 1)
                )

                for t in range(self.time_steps):
                    d_s = d_spike_last[t]
                    v_t = volt_hist[-1][t]
                    surrogate = self.lif.surrogate_gradient(v_t)
                    d_v = d_s * surrogate

                    h_in = spikes_hist[-2][t] if self.num_layers > 1 else xb
                    d_W[-1] += np.dot(h_in.T, d_v)
                    d_b[-1] += np.sum(d_v, axis=0)

                # SGD update
                for layer_idx in range(self.num_layers):
                    self.weights[layer_idx] -= self.learning_rate * np.clip(
                        d_W[layer_idx], -5.0, 5.0
                    )
                    self.biases[layer_idx] -= self.learning_rate * np.clip(
                        d_b[layer_idx], -5.0, 5.0
                    )

            avg_loss = float(np.mean(epoch_loss))
            history.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"SNN Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        rates, _, _ = self.forward(X)
        return np.argmax(rates, axis=-1)
