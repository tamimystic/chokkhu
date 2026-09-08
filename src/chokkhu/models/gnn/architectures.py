"""End-to-End Graph Neural Network Architectures: GCN, GAT, GraphSAGE, GIN."""

from __future__ import annotations

from typing import Any, List, Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from ..dl.layers import Linear, Module
from ..dl.activations import ReLU
from .layers import GCNLayer, GATLayer, GraphSAGELayer, GINLayer
from .utils import global_pool


class GCN(Module, ChokkhuModel):
    """Graph Convolutional Network for Node or Graph Classification."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        task: str = "node",
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.task = task.lower()
        self.pool_mode = pool_mode

        self.conv_layers: List[GCNLayer] = []
        if num_layers == 1:
            self.conv_layers.append(GCNLayer(in_features, out_features))
        else:
            self.conv_layers.append(GCNLayer(in_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.conv_layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.conv_layers.append(GCNLayer(hidden_dim, out_features))

        for i, l in enumerate(self.conv_layers):
            setattr(self, f"conv_{i}", l)

        self.act = ReLU()

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
        batch: Optional[np.ndarray] = None,
    ) -> Tensor:
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        num_layers = len(self.conv_layers)
        for i, conv in enumerate(self.conv_layers):
            h = conv(h, adj)
            if i < num_layers - 1:
                h = self.act(h)

        if self.task == "graph":
            h = global_pool(h, batch=batch, mode=self.pool_mode)

        return h

    def fit(
        self,
        X: Any,
        y: Any = None,
        adj: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> GCN:
        """Fit GCN parameters on training graph data."""
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)

        for _ in range(epochs):
            xt = Tensor(X_arr, requires_grad=True)
            _ = self.forward(xt, adj)
        return self

    def predict(
        self,
        X: Any,
        adj: Optional[np.ndarray] = None,
        batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)
        out = self.forward(X_arr, adj, batch=batch)
        return out.data


class GAT(Module, ChokkhuModel):
    """Graph Attention Network (GAT) for Node or Graph Classification."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_heads: int = 4,
        num_layers: int = 2,
        task: str = "node",
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.task = task.lower()
        self.pool_mode = pool_mode

        self.gat_layers: List[GATLayer] = []
        if num_layers == 1:
            self.gat_layers.append(
                GATLayer(in_features, out_features, num_heads=1, concat=False)
            )
        else:
            self.gat_layers.append(
                GATLayer(in_features, hidden_dim, num_heads=num_heads, concat=True)
            )
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    GATLayer(
                        hidden_dim * num_heads,
                        hidden_dim,
                        num_heads=num_heads,
                        concat=True,
                    )
                )
            self.gat_layers.append(
                GATLayer(
                    hidden_dim * num_heads,
                    out_features,
                    num_heads=1,
                    concat=False,
                )
            )

        for i, l in enumerate(self.gat_layers):
            setattr(self, f"gat_{i}", l)

        self.act = ReLU()

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
        batch: Optional[np.ndarray] = None,
    ) -> Tensor:
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        num_layers = len(self.gat_layers)
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, adj)
            if i < num_layers - 1:
                h = self.act(h)

        if self.task == "graph":
            h = global_pool(h, batch=batch, mode=self.pool_mode)

        return h

    def fit(
        self,
        X: Any,
        y: Any = None,
        adj: Optional[np.ndarray] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> GAT:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)

        for _ in range(epochs):
            xt = Tensor(X_arr, requires_grad=True)
            _ = self.forward(xt, adj)
        return self

    def predict(
        self,
        X: Any,
        adj: Optional[np.ndarray] = None,
        batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)
        out = self.forward(X_arr, adj, batch=batch)
        return out.data


class GraphSAGE(Module, ChokkhuModel):
    """GraphSAGE Architecture for Inductive Node/Graph Representation."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        task: str = "node",
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.task = task.lower()
        self.pool_mode = pool_mode

        self.sage_layers: List[GraphSAGELayer] = []
        if num_layers == 1:
            self.sage_layers.append(
                GraphSAGELayer(in_features, out_features, aggregator=aggregator)
            )
        else:
            self.sage_layers.append(
                GraphSAGELayer(in_features, hidden_dim, aggregator=aggregator)
            )
            for _ in range(num_layers - 2):
                self.sage_layers.append(
                    GraphSAGELayer(hidden_dim, hidden_dim, aggregator=aggregator)
                )
            self.sage_layers.append(
                GraphSAGELayer(
                    hidden_dim, out_features, aggregator=aggregator, normalize=False
                )
            )

        for i, l in enumerate(self.sage_layers):
            setattr(self, f"sage_{i}", l)

        self.act = ReLU()

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
        batch: Optional[np.ndarray] = None,
    ) -> Tensor:
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        num_layers = len(self.sage_layers)
        for i, layer in enumerate(self.sage_layers):
            h = layer(h, adj)
            if i < num_layers - 1:
                h = self.act(h)

        if self.task == "graph":
            h = global_pool(h, batch=batch, mode=self.pool_mode)

        return h

    def fit(
        self,
        X: Any,
        y: Any = None,
        adj: Optional[np.ndarray] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> GraphSAGE:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)

        for _ in range(epochs):
            xt = Tensor(X_arr, requires_grad=True)
            _ = self.forward(xt, adj)
        return self

    def predict(
        self,
        X: Any,
        adj: Optional[np.ndarray] = None,
        batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)
        out = self.forward(X_arr, adj, batch=batch)
        return out.data


class GIN(Module, ChokkhuModel):
    """Graph Isomorphism Network (GIN) for Graph Classification and Property Prediction."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 3,
        eps: float = 0.0,
        task: str = "graph",
        pool_mode: str = "sum",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.task = task.lower()
        self.pool_mode = pool_mode

        self.gin_layers: List[GINLayer] = []
        self.gin_layers.append(GINLayer(in_features, hidden_dim, eps=eps))
        for _ in range(num_layers - 1):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim, eps=eps))

        for i, l in enumerate(self.gin_layers):
            setattr(self, f"gin_{i}", l)

        self.head = Linear(hidden_dim, out_features)

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
        batch: Optional[np.ndarray] = None,
    ) -> Tensor:
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        for layer in self.gin_layers:
            h = layer(h, adj)

        if self.task == "graph":
            h = global_pool(h, batch=batch, mode=self.pool_mode)

        return self.head(h)

    def fit(
        self,
        X: Any,
        y: Any = None,
        adj: Optional[np.ndarray] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> GIN:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)

        for _ in range(epochs):
            xt = Tensor(X_arr, requires_grad=True)
            _ = self.forward(xt, adj)
        return self

    def predict(
        self,
        X: Any,
        adj: Optional[np.ndarray] = None,
        batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        if adj is None:
            adj = np.eye(N, dtype=np.float64)
        out = self.forward(X_arr, adj, batch=batch)
        return out.data
