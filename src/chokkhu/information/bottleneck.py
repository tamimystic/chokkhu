"""Deep Variational Information Bottleneck (VIB) in Pure NumPy.

References:
- Alemi et al. (2017): "Deep Variational Information Bottleneck" (ICLR 2017).
- Tishby et al. (2000): "The Information Bottleneck Method" (Allerton).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
import numpy as np


class DeepVariationalInformationBottleneck:
    """Deep Variational Information Bottleneck (VIB) Classifier & Stochastic Feature Extractor.

    Compresses input X into a stochastic latent representation Z ~ N(mu(X), sigma^2(X))
    while optimizing the Information Bottleneck variational bound:
        Loss = E_{q_theta(z|x)} [ CrossEntropy(y, Decoder(z)) ] + beta * KL( q_theta(z|x) || N(0, I) )
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        num_classes: int = 2,
        beta: float = 1e-3,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Initialize network weights (He / Xavier initialization)
        self._init_weights()

        # Adam optimizer state
        self._init_adam()

    def _init_weights(self) -> None:
        """Initialize encoder and decoder parameters."""
        # Encoder: X -> Hidden -> (mu, log_var)
        scale_enc1 = np.sqrt(2.0 / self.input_dim)
        self.W_enc1 = self.rng.randn(self.input_dim, self.hidden_dim) * scale_enc1
        self.b_enc1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        scale_mu = np.sqrt(2.0 / self.hidden_dim)
        self.W_mu = self.rng.randn(self.hidden_dim, self.latent_dim) * scale_mu
        self.b_mu: np.ndarray = np.zeros(self.latent_dim, dtype=np.float64)

        scale_logvar = np.sqrt(2.0 / self.hidden_dim)
        self.W_logvar = self.rng.randn(self.hidden_dim, self.latent_dim) * scale_logvar
        self.b_logvar: np.ndarray = np.zeros(self.latent_dim, dtype=np.float64)

        # Decoder: Z -> Hidden -> Logits
        scale_dec1 = np.sqrt(2.0 / self.latent_dim)
        self.W_dec1 = self.rng.randn(self.latent_dim, self.hidden_dim) * scale_dec1
        self.b_dec1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        scale_dec2 = np.sqrt(2.0 / self.hidden_dim)
        self.W_dec2 = self.rng.randn(self.hidden_dim, self.num_classes) * scale_dec2
        self.b_dec2: np.ndarray = np.zeros(self.num_classes, dtype=np.float64)

    def _init_adam(self) -> None:
        """Initialize Adam moment buffers."""
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.param_names = [
            "W_enc1",
            "b_enc1",
            "W_mu",
            "b_mu",
            "W_logvar",
            "b_logvar",
            "W_dec1",
            "b_dec1",
            "W_dec2",
            "b_dec2",
        ]
        for name in self.param_names:
            p = getattr(self, name)
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)
        self.t = 0

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def encode(
        self, X: np.ndarray, sample: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Encode input X into latent parameters mu, logvar or sampled latent Z."""
        x_arr = np.asarray(X, dtype=np.float64)
        h_enc = np.maximum(0.0, np.dot(x_arr, self.W_enc1) + self.b_enc1)
        mu = np.dot(h_enc, self.W_mu) + self.b_mu
        logvar = np.clip(np.dot(h_enc, self.W_logvar) + self.b_logvar, -10.0, 10.0)

        if sample:
            std = np.exp(0.5 * logvar)
            eps = self.rng.randn(*mu.shape)
            z = mu + std * eps
            return z, mu, logvar
        return mu

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent Z into class logits."""
        z_arr = np.asarray(Z, dtype=np.float64)
        h_dec = np.maximum(0.0, np.dot(z_arr, self.W_dec1) + self.b_dec1)
        logits = np.dot(h_dec, self.W_dec2) + self.b_dec2
        return logits

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass with cached activations for backward pass."""
        x_arr = np.asarray(X, dtype=np.float64)
        # Encoder
        h_enc = np.maximum(0.0, np.dot(x_arr, self.W_enc1) + self.b_enc1)
        mu = np.dot(h_enc, self.W_mu) + self.b_mu
        logvar = np.clip(np.dot(h_enc, self.W_logvar) + self.b_logvar, -10.0, 10.0)

        # Reparameterization
        std = np.exp(0.5 * logvar)
        eps = self.rng.randn(*mu.shape)
        z = mu + std * eps

        # Decoder
        h_dec = np.maximum(0.0, np.dot(z, self.W_dec1) + self.b_dec1)
        logits = np.dot(h_dec, self.W_dec2) + self.b_dec2
        probs = self._softmax(logits)

        return probs, z, mu, logvar, h_enc, h_dec

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train Deep VIB using stochastic gradient descent with Adam optimizer.

        Returns:
            Dictionary containing loss histories: 'total_loss', 'ce_loss', 'kl_loss'.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]

        history: Dict[str, List[float]] = {
            "total_loss": [],
            "ce_loss": [],
            "kl_loss": [],
        }

        beta_1, beta_2, eps = 0.9, 0.999, 1e-8

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            epoch_total_loss: List[float] = []
            epoch_ce_loss: List[float] = []
            epoch_kl_loss: List[float] = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                xb = x_arr[batch_idx]
                yb = y_arr[batch_idx]
                bs = len(xb)
                if bs == 0:
                    continue

                # 1. Forward
                h_enc = np.maximum(0.0, np.dot(xb, self.W_enc1) + self.b_enc1)
                mu = np.dot(h_enc, self.W_mu) + self.b_mu
                logvar = np.clip(
                    np.dot(h_enc, self.W_logvar) + self.b_logvar, -10.0, 10.0
                )
                std = np.exp(0.5 * logvar)
                eps_noise = self.rng.randn(*mu.shape)
                z = mu + std * eps_noise

                h_dec = np.maximum(0.0, np.dot(z, self.W_dec1) + self.b_dec1)
                logits = np.dot(h_dec, self.W_dec2) + self.b_dec2
                probs = self._softmax(logits)

                # 2. Compute Loss
                # Cross-entropy
                log_probs = np.log(np.maximum(probs, 1e-12))
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(bs), yb] = 1.0
                ce_loss = float(-np.mean(np.sum(one_hot * log_probs, axis=1)))

                # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                kl_divergence = -0.5 * np.sum(
                    1.0 + logvar - mu**2 - np.exp(logvar), axis=1
                )
                kl_loss = float(np.mean(kl_divergence))

                total_loss = ce_loss + self.beta * kl_loss

                epoch_total_loss.append(total_loss)
                epoch_ce_loss.append(ce_loss)
                epoch_kl_loss.append(kl_loss)

                # 3. Backprop
                # dL / dlogits = (probs - one_hot) / bs
                d_logits = (probs - one_hot) / float(bs)
                d_W_dec2 = np.dot(h_dec.T, d_logits)
                d_b_dec2 = np.sum(d_logits, axis=0)

                d_h_dec = np.dot(d_logits, self.W_dec2.T) * (h_dec > 0.0)
                d_W_dec1 = np.dot(z.T, d_h_dec)
                d_b_dec1 = np.sum(d_h_dec, axis=0)

                d_z = np.dot(d_h_dec, self.W_dec1.T)

                # Reparameterization gradients + KL gradients
                # dL / dmu = d_z + beta * (mu / bs)
                d_mu = d_z + self.beta * (mu / float(bs))

                # dL / dlogvar = d_z * 0.5 * std * eps + beta * 0.5 * (exp(logvar) - 1) / bs
                d_logvar = d_z * (0.5 * std * eps_noise) + self.beta * 0.5 * (
                    np.exp(logvar) - 1.0
                ) / float(bs)

                # Encoder backprop
                d_h_enc = np.dot(d_mu, self.W_mu.T) + np.dot(d_logvar, self.W_logvar.T)
                d_h_enc = d_h_enc * (h_enc > 0.0)

                d_W_mu = np.dot(h_enc.T, d_mu)
                d_b_mu = np.sum(d_mu, axis=0)

                d_W_logvar = np.dot(h_enc.T, d_logvar)
                d_b_logvar = np.sum(d_logvar, axis=0)

                d_W_enc1 = np.dot(xb.T, d_h_enc)
                d_b_enc1 = np.sum(d_h_enc, axis=0)

                grads = {
                    "W_enc1": d_W_enc1,
                    "b_enc1": d_b_enc1,
                    "W_mu": d_W_mu,
                    "b_mu": d_b_mu,
                    "W_logvar": d_W_logvar,
                    "b_logvar": d_b_logvar,
                    "W_dec1": d_W_dec1,
                    "b_dec1": d_b_dec1,
                    "W_dec2": d_W_dec2,
                    "b_dec2": d_b_dec2,
                }

                # 4. Adam parameter update
                self.t += 1
                for name in self.param_names:
                    g = np.clip(grads[name], -5.0, 5.0)
                    self.m[name] = beta_1 * self.m[name] + (1.0 - beta_1) * g
                    self.v[name] = beta_2 * self.v[name] + (1.0 - beta_2) * (g**2)

                    m_hat = self.m[name] / (1.0 - beta_1**self.t)
                    v_hat = self.v[name] / (1.0 - beta_2**self.t)

                    param = getattr(self, name)
                    param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            mean_tot = float(np.mean(epoch_total_loss))
            mean_ce = float(np.mean(epoch_ce_loss))
            mean_kl = float(np.mean(epoch_kl_loss))
            history["total_loss"].append(mean_tot)
            history["ce_loss"].append(mean_ce)
            history["kl_loss"].append(mean_kl)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - Total: {mean_tot:.4f}, CE: {mean_ce:.4f}, KL: {mean_kl:.4f}"
                )

        return history

    def predict_proba(self, X: np.ndarray, num_samples: int = 5) -> np.ndarray:
        """Predict class probability distributions with MC integration over latent space."""
        x_arr = np.asarray(X, dtype=np.float64)
        n = x_arr.shape[0]
        h_enc = np.maximum(0.0, np.dot(x_arr, self.W_enc1) + self.b_enc1)
        mu = np.dot(h_enc, self.W_mu) + self.b_mu
        logvar = np.clip(np.dot(h_enc, self.W_logvar) + self.b_logvar, -10.0, 10.0)
        std = np.exp(0.5 * logvar)

        accum_probs = np.zeros((n, self.num_classes), dtype=np.float64)
        for _ in range(num_samples):
            eps = self.rng.randn(*mu.shape)
            z = mu + std * eps
            h_dec = np.maximum(0.0, np.dot(z, self.W_dec1) + self.b_dec1)
            logits = np.dot(h_dec, self.W_dec2) + self.b_dec2
            accum_probs += self._softmax(logits)

        return accum_probs / float(num_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class labels."""
        probs = self.predict_proba(X, num_samples=5)
        return np.argmax(probs, axis=1)
