"""Kronecker-Factored Approximate Curvature (K-FAC) Second-Order Natural Gradient Optimizer.

Formulated from first principles in pure NumPy following Martens & Grosse (ICML 2015).
Approximates the Fisher Information Matrix (FIM) as Kronecker products of layer activation
covariances and pre-activation gradient covariances for scalable curvature-aware optimization.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class KFAC:
    r"""Kronecker-Factored Approximate Curvature (K-FAC) Optimizer.

    Approximates the intractable block-diagonal Fisher Information Matrix $F_l$ for dense layers:
    F_l \approx A_{l-1} \otimes S_l \implies F_l^{-1} \approx A_{l-1}^{-1} \otimes S_l^{-1}

    where $A_{l-1} = \mathbb{E}[a_{l-1} a_{l-1}^T]$ is the activation covariance and
    $S_l = \mathbb{E}[s_l s_l^T]$ is the pre-activation derivative covariance.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate $\eta$.
    damping : float, default=1e-3
        Tikhonov damping factor $\gamma$ for positive-definite factor inversion.
    momentum : float, default=0.9
        Momentum coefficient for parameter velocity updates.
    ema_decay : float, default=0.95
        Exponential moving average decay $\rho$ for Kronecker covariance factors.
    inv_freq : int, default=10
        Frequency of exact matrix inversion passes.
    """

    def __init__(
        self,
        lr: float = 0.01,
        damping: float = 1e-3,
        momentum: float = 0.9,
        ema_decay: float = 0.95,
        inv_freq: int = 10,
    ) -> None:
        self.lr = float(lr)
        self.damping = float(damping)
        self.momentum = float(momentum)
        self.ema_decay = float(ema_decay)
        self.inv_freq = max(1, int(inv_freq))

        self.step_count = 0

        # Layer factor covariances: A = a^T a / B, S = s^T s / B
        self.A_factors: Dict[str, np.ndarray] = {}
        self.S_factors: Dict[str, np.ndarray] = {}

        # Inverted factors
        self.A_inv: Dict[str, np.ndarray] = {}
        self.S_inv: Dict[str, np.ndarray] = {}

        # Momentum velocities
        self.velocities: Dict[str, np.ndarray] = {}

    def update_factors(
        self,
        layer_id: str,
        activations: np.ndarray,
        pre_grads: np.ndarray,
    ) -> None:
        r"""Update running EMA covariance matrices A and S for a linear layer.

        Parameters
        ----------
        layer_id : str
            Unique identifier for the layer.
        activations : np.ndarray of shape (B, in_dim)
            Layer input activations a_{l-1} (including bias column if applicable).
        pre_grads : np.ndarray of shape (B, out_dim)
            Gradients with respect to pre-activation linear outputs: s_l = \nabla_{z_l} \mathcal{L}.
        """
        B = activations.shape[0]
        a = np.asarray(activations, dtype=float)
        s = np.asarray(pre_grads, dtype=float)

        # Batch covariance matrices
        A_batch = np.dot(a.T, a) / B
        S_batch = np.dot(s.T, s) / B

        rho = self.ema_decay
        if layer_id not in self.A_factors:
            self.A_factors[layer_id] = A_batch
            self.S_factors[layer_id] = S_batch
        else:
            self.A_factors[layer_id] = (
                rho * self.A_factors[layer_id] + (1.0 - rho) * A_batch
            )
            self.S_factors[layer_id] = (
                rho * self.S_factors[layer_id] + (1.0 - rho) * S_batch
            )

    def compute_inverses(self, force: bool = False) -> None:
        """Compute damped matrix inverses for all registered layer Kronecker factors."""
        if not force and (self.step_count % self.inv_freq != 0) and self.A_inv:
            return

        for layer_id in self.A_factors:
            A = self.A_factors[layer_id]
            S = self.S_factors[layer_id]

            d_a = A.shape[0]
            d_s = S.shape[0]

            tr_A = np.trace(A) / d_a + 1e-12
            tr_S = np.trace(S) / d_s + 1e-12

            # Optimal trace scaling factor pi
            pi = float(np.sqrt(tr_A / tr_S))
            gamma = self.damping

            damp_A = gamma * pi
            damp_S = gamma / (pi + 1e-12)

            A_damped = A + damp_A * np.eye(d_a)
            S_damped = S + damp_S * np.eye(d_s)

            try:
                self.A_inv[layer_id] = np.linalg.inv(A_damped)
            except np.linalg.LinAlgError:
                self.A_inv[layer_id] = np.linalg.pinv(A_damped)

            try:
                self.S_inv[layer_id] = np.linalg.inv(S_damped)
            except np.linalg.LinAlgError:
                self.S_inv[layer_id] = np.linalg.pinv(S_damped)

    def precondition_gradient(
        self,
        layer_id: str,
        grad_weight: np.ndarray,
    ) -> np.ndarray:
        r"""Compute natural gradient precondition: \nabla^{\text{nat}} W = S^{-1} \nabla_W L A^{-1}.

        Parameters
        ----------
        layer_id : str
            Layer identifier.
        grad_weight : np.ndarray of shape (out_dim, in_dim)
            Standard Euclidean gradient matrix \nabla_W \mathcal{L}.

        Returns
        -------
        nat_grad : np.ndarray of shape (out_dim, in_dim)
            Preconditioned natural gradient.
        """
        if layer_id not in self.A_inv or layer_id not in self.S_inv:
            self.compute_inverses(force=True)

        A_inv = self.A_inv[layer_id]
        S_inv = self.S_inv[layer_id]

        # Natural gradient: S_inv @ grad @ A_inv
        nat_grad = np.dot(S_inv, np.dot(grad_weight, A_inv))
        return nat_grad

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
    ) -> None:
        """Apply preconditioned natural gradient step with momentum.

        Parameters
        ----------
        params : Dict[str, np.ndarray]
            Dictionary of parameter matrices to be updated in-place.
        grads : Dict[str, np.ndarray]
            Dictionary of parameter gradients.
        """
        self.step_count += 1
        self.compute_inverses()

        for name, param in params.items():
            if name in grads:
                g = grads[name]
                if name in self.A_factors:
                    nat_g = self.precondition_gradient(name, g)
                else:
                    nat_g = g

                if name not in self.velocities:
                    self.velocities[name] = np.zeros_like(param)

                v = self.momentum * self.velocities[name] + self.lr * nat_g
                self.velocities[name] = v
                param -= v
