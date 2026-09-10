"""Federated Learning server, client, FedAvg, and FedProx algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FederatedClient:
    """Federated Learning client participating in decentralized model training.

    Parameters
    ----------
    client_id : str
        Unique identifier for the client node.
    X : np.ndarray
        Local client feature dataset.
    y : np.ndarray
        Local client target labels/values.
    mu : float
        FedProx proximal penalty coefficient (mu = 0 for standard FedAvg).
    lr : float
        Local optimization learning rate.
    random_state : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        mu: float = 0.0,
        lr: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        self.client_id = client_id
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n_samples = len(self.X)
        self.mu = float(mu)
        self.lr = float(lr)
        self.rng = np.random.default_rng(random_state)

    def train_local(
        self,
        global_weights: Dict[str, np.ndarray],
        local_epochs: int = 5,
        batch_size: int = 16,
    ) -> Tuple[Dict[str, np.ndarray], int, float]:
        """Perform local SGD training starting from global weights.

        Returns
        -------
        updated_weights : Dict[str, np.ndarray]
            Updated local model weights.
        num_samples : int
            Number of local training samples.
        mean_loss : float
            Average training loss during final local epoch.
        """
        # Deep copy initial weights
        weights = {k: v.copy() for k, v in global_weights.items()}
        global_ref = {k: v.copy() for k, v in global_weights.items()}

        losses: List[float] = []
        n = self.n_samples
        W = weights.get("W", np.zeros((self.X.shape[1], 1)))
        b = weights.get("b", np.zeros((1,)))

        for _ in range(local_epochs):
            indices = self.rng.permutation(n)
            epoch_loss = 0.0
            num_batches = int(np.ceil(n / batch_size))

            for batch_i in range(num_batches):
                b_idx = indices[
                    batch_i * batch_size : min((batch_i + 1) * batch_size, n)
                ]
                X_b, y_b = self.X[b_idx], self.y[b_idx]
                if len(y_b.shape) == 1:
                    y_b = y_b[:, None]

                # Linear forward: y_pred = X_b @ W + b
                preds = np.dot(X_b, W) + b
                residuals = preds - y_b
                mse_loss = float(np.mean(residuals**2))

                # Gradients with FedProx proximal term: grad += mu * (W - W_global)
                grad_W = (2.0 / len(X_b)) * np.dot(X_b.T, residuals)
                grad_b = (2.0 / len(X_b)) * np.sum(residuals, axis=0)

                if self.mu > 0.0:
                    grad_W += self.mu * (W - global_ref["W"])
                    grad_b += self.mu * (b - global_ref["b"])
                    prox_loss = (
                        0.5
                        * self.mu
                        * (
                            float(np.sum((W - global_ref["W"]) ** 2))
                            + float(np.sum((b - global_ref["b"]) ** 2))
                        )
                    )
                    mse_loss += prox_loss

                W -= self.lr * grad_W
                b -= self.lr * grad_b
                epoch_loss += mse_loss

            losses.append(epoch_loss / max(1, num_batches))

        weights["W"] = W
        weights["b"] = b
        final_loss = losses[-1] if losses else 0.0
        return weights, n, final_loss


class FederatedServer:
    """Federated Learning orchestrator executing FedAvg and FedProx rounds.

    Parameters
    ----------
    initial_weights : Dict[str, np.ndarray]
        Initial global model parameters (e.g. {'W': ..., 'b': ...}).
    clients : Optional[List[FederatedClient]]
        List of registered federated client nodes.
    strategy : str
        Aggregation strategy: 'fedavg' or 'fedprox'.
    random_state : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        initial_weights: Dict[str, np.ndarray],
        clients: Optional[List[FederatedClient]] = None,
        strategy: str = "fedavg",
        random_state: Optional[int] = None,
    ) -> None:
        self.weights = {k: v.copy() for k, v in initial_weights.items()}
        self.clients: List[FederatedClient] = clients or []
        self.strategy = strategy.lower()
        self.rng = np.random.default_rng(random_state)
        self.history: List[Dict[str, Any]] = []

    def add_client(self, client: FederatedClient) -> None:
        """Register a new client node to the server."""
        self.clients.append(client)

    def train_round(
        self,
        fraction_fit: float = 1.0,
        local_epochs: int = 5,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Execute one federated communication round.

        1. Select participating clients.
        2. Broadcast global model weights.
        3. Clients train locally.
        4. Aggregate local weights into new global model.
        """
        if not self.clients:
            raise ValueError("No clients registered with the federated server.")

        num_clients = len(self.clients)
        num_selected = max(1, int(np.ceil(num_clients * fraction_fit)))
        selected_idx = self.rng.choice(num_clients, size=num_selected, replace=False)
        selected_clients = [self.clients[i] for i in selected_idx]

        client_updates: List[Tuple[Dict[str, np.ndarray], int]] = []
        losses: List[float] = []

        for client in selected_clients:
            local_weights, n_samples, loss = client.train_local(
                global_weights=self.weights,
                local_epochs=local_epochs,
                batch_size=batch_size,
            )
            client_updates.append((local_weights, n_samples))
            losses.append(loss)

        # Weighted Federated Averaging
        total_samples = sum(n for _, n in client_updates)
        new_weights: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.weights.items()
        }

        for client_weights, n_samples in client_updates:
            weight_factor = float(n_samples) / float(total_samples)
            for k in self.weights:
                new_weights[k] += client_weights[k] * weight_factor

        self.weights = new_weights
        round_info = {
            "round": len(self.history) + 1,
            "participating_clients": num_selected,
            "mean_client_loss": float(np.mean(losses)),
        }
        self.history.append(round_info)
        return round_info

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return the current aggregated global model weights."""
        return {k: v.copy() for k, v in self.weights.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with the global linear model."""
        X_arr = np.asarray(X, dtype=np.float64)
        W = self.weights["W"]
        b = self.weights["b"]
        return np.dot(X_arr, W) + b
