"""Recurrent Neural Networks (RNN, LSTM, BiLSTM, GRU) from First Principles."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Module, Parameter
from ..base import ChokkhuModel


class RNN(Module, ChokkhuModel):
    """Multi-layer Elman Recurrent Neural Network (RNN) from Scratch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.has_bias = bias
        self.batch_first = batch_first

        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []

        for layer in range(num_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            scale = 1.0 / np.sqrt(hidden_dim)

            w_ih_p = Parameter(np.random.uniform(-scale, scale, (hidden_dim, in_d)))
            w_hh_p = Parameter(
                np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))
            )
            self.w_ih.append(w_ih_p)
            self.w_hh.append(w_hh_p)
            setattr(self, f"w_ih_l{layer}", w_ih_p)
            setattr(self, f"w_hh_l{layer}", w_hh_p)

            if bias:
                b_ih_p = Parameter(np.zeros((hidden_dim,)))
                b_hh_p = Parameter(np.zeros((hidden_dim,)))
                self.b_ih.append(b_ih_p)
                self.b_hh.append(b_hh_p)
                setattr(self, f"b_ih_l{layer}", b_ih_p)
                setattr(self, f"b_hh_l{layer}", b_hh_p)

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass over sequence.

        Args:
            x: (batch_size, seq_len, input_dim) if batch_first else (seq_len, batch_size, input_dim)
            h_0: Optional initial hidden state (num_layers, batch_size, hidden_dim)
        Returns:
            (output, h_n)
        """
        x_data = x.data if self.batch_first else x.data.transpose(1, 0, 2)
        N, T, D = x_data.shape

        if h_0 is None:
            h_prev_layers = [
                np.zeros((N, self.hidden_dim), dtype=np.float64)
                for _ in range(self.num_layers)
            ]
        else:
            h_prev_layers = [h_0.data[idx] for idx in range(self.num_layers)]

        current_input = x_data

        for layer in range(self.num_layers):
            w_ih = self.w_ih[layer].data
            w_hh = self.w_hh[layer].data
            b_ih = self.b_ih[layer].data if self.has_bias else 0.0
            b_hh = self.b_hh[layer].data if self.has_bias else 0.0

            h_t = h_prev_layers[layer]
            seq_out = []

            for t in range(T):
                x_t = current_input[:, t, :]  # (N, in_d)
                gate = np.dot(x_t, w_ih.T) + np.dot(h_t, w_hh.T) + b_ih + b_hh
                if self.nonlinearity == "relu":
                    h_t = np.maximum(0.0, gate)
                else:
                    h_t = np.tanh(gate)
                seq_out.append(h_t[:, np.newaxis, :])

            current_input = np.concatenate(seq_out, axis=1)  # (N, T, hidden_dim)
            h_prev_layers[layer] = h_t

        output_data = (
            current_input if self.batch_first else current_input.transpose(1, 0, 2)
        )
        h_n_data = np.stack(h_prev_layers, axis=0)

        out_tensor = Tensor(output_data, requires_grad=x.requires_grad)
        h_n_tensor = Tensor(h_n_data, requires_grad=x.requires_grad)
        return out_tensor, h_n_tensor


class LSTM(Module, ChokkhuModel):
    """Multi-layer Long Short-Term Memory (LSTM) Architecture from Scratch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first
        self.dropout_rate = dropout

        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []

        for layer in range(num_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            scale = 1.0 / np.sqrt(hidden_dim)

            # 4 gates: input (i), forget (f), cell candidate (g), output (o)
            w_ih_p = Parameter(np.random.uniform(-scale, scale, (4 * hidden_dim, in_d)))
            w_hh_p = Parameter(
                np.random.uniform(-scale, scale, (4 * hidden_dim, hidden_dim))
            )
            self.w_ih.append(w_ih_p)
            self.w_hh.append(w_hh_p)
            setattr(self, f"w_ih_l{layer}", w_ih_p)
            setattr(self, f"w_hh_l{layer}", w_hh_p)

            if bias:
                b_ih_p = Parameter(np.zeros((4 * hidden_dim,)))
                b_hh_p = Parameter(np.zeros((4 * hidden_dim,)))
                # Initialize forget gate bias to 1.0 (standard best practice)
                b_ih_p.data[hidden_dim : 2 * hidden_dim] = 1.0
                self.b_ih.append(b_ih_p)
                self.b_hh.append(b_hh_p)
                setattr(self, f"b_ih_l{layer}", b_ih_p)
                setattr(self, f"b_hh_l{layer}", b_hh_p)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass across LSTM sequence."""
        x_data = x.data if self.batch_first else x.data.transpose(1, 0, 2)
        N, T, D = x_data.shape
        H = self.hidden_dim

        if hx is None:
            h_prev_layers = [
                np.zeros((N, H), dtype=np.float64) for _ in range(self.num_layers)
            ]
            c_prev_layers = [
                np.zeros((N, H), dtype=np.float64) for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hx
            h_prev_layers = [h_0.data[idx] for idx in range(self.num_layers)]
            c_prev_layers = [c_0.data[idx] for idx in range(self.num_layers)]

        current_input = x_data

        for layer in range(self.num_layers):
            w_ih = self.w_ih[layer].data
            w_hh = self.w_hh[layer].data
            b_ih = self.b_ih[layer].data if self.has_bias else 0.0
            b_hh = self.b_hh[layer].data if self.has_bias else 0.0

            h_t = h_prev_layers[layer]
            c_t = c_prev_layers[layer]
            seq_out = []

            for t in range(T):
                x_t = current_input[:, t, :]
                gates = np.dot(x_t, w_ih.T) + np.dot(h_t, w_hh.T) + b_ih + b_hh
                i_gate = self._sigmoid(gates[:, 0 * H : 1 * H])
                f_gate = self._sigmoid(gates[:, 1 * H : 2 * H])
                g_gate = np.tanh(gates[:, 2 * H : 3 * H])
                o_gate = self._sigmoid(gates[:, 3 * H : 4 * H])

                c_t = f_gate * c_t + i_gate * g_gate
                h_t = o_gate * np.tanh(c_t)
                seq_out.append(h_t[:, np.newaxis, :])

            current_input = np.concatenate(seq_out, axis=1)
            h_prev_layers[layer] = h_t
            c_prev_layers[layer] = c_t

        output_data = (
            current_input if self.batch_first else current_input.transpose(1, 0, 2)
        )
        h_n_data = np.stack(h_prev_layers, axis=0)
        c_n_data = np.stack(c_prev_layers, axis=0)

        out_tensor = Tensor(output_data, requires_grad=x.requires_grad)
        h_n_tensor = Tensor(h_n_data, requires_grad=x.requires_grad)
        c_n_tensor = Tensor(c_n_data, requires_grad=x.requires_grad)
        return out_tensor, (h_n_tensor, c_n_tensor)


class BiLSTM(Module, ChokkhuModel):
    """Bidirectional LSTM from Scratch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.forward_lstm = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
        )
        self.backward_lstm = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass concatenating forward and backward sequence representations."""
        x_data = x.data if self.batch_first else x.data.transpose(1, 0, 2)
        # Reverse sequence along time dimension for backward pass
        x_rev_data = x_data[:, ::-1, :]
        x_rev = Tensor(x_rev_data, requires_grad=x.requires_grad)

        f_out, (f_hn, f_cn) = self.forward_lstm(x)
        b_out, (b_hn, b_cn) = self.backward_lstm(x_rev)

        b_out_rev_data = b_out.data[:, ::-1, :]
        cat_out = np.concatenate([f_out.data, b_out_rev_data], axis=-1)
        cat_hn = np.concatenate([f_hn.data, b_hn.data], axis=-1)
        cat_cn = np.concatenate([f_cn.data, b_cn.data], axis=-1)

        return (
            Tensor(cat_out, requires_grad=x.requires_grad),
            (
                Tensor(cat_hn, requires_grad=x.requires_grad),
                Tensor(cat_cn, requires_grad=x.requires_grad),
            ),
        )


class GRU(Module, ChokkhuModel):
    """Gated Recurrent Unit (GRU) from Scratch (Cho et al., 2014)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first

        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []

        for layer in range(num_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            scale = 1.0 / np.sqrt(hidden_dim)

            # 3 gates: reset (r), update (z), new candidate (n)
            w_ih_p = Parameter(np.random.uniform(-scale, scale, (3 * hidden_dim, in_d)))
            w_hh_p = Parameter(
                np.random.uniform(-scale, scale, (3 * hidden_dim, hidden_dim))
            )
            self.w_ih.append(w_ih_p)
            self.w_hh.append(w_hh_p)
            setattr(self, f"w_ih_l{layer}", w_ih_p)
            setattr(self, f"w_hh_l{layer}", w_hh_p)

            if bias:
                b_ih_p = Parameter(np.zeros((3 * hidden_dim,)))
                b_hh_p = Parameter(np.zeros((3 * hidden_dim,)))
                self.b_ih.append(b_ih_p)
                self.b_hh.append(b_hh_p)
                setattr(self, f"b_ih_l{layer}", b_ih_p)
                setattr(self, f"b_hh_l{layer}", b_hh_p)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass across GRU sequence."""
        x_data = x.data if self.batch_first else x.data.transpose(1, 0, 2)
        N, T, D = x_data.shape
        H = self.hidden_dim

        if h_0 is None:
            h_prev_layers = [
                np.zeros((N, H), dtype=np.float64) for _ in range(self.num_layers)
            ]
        else:
            h_prev_layers = [h_0.data[idx] for idx in range(self.num_layers)]

        current_input = x_data

        for layer in range(self.num_layers):
            w_ih = self.w_ih[layer].data
            w_hh = self.w_hh[layer].data
            b_ih = self.b_ih[layer].data if self.has_bias else 0.0
            b_hh = self.b_hh[layer].data if self.has_bias else 0.0

            h_t = h_prev_layers[layer]
            seq_out = []

            for t in range(T):
                x_t = current_input[:, t, :]
                gate_ih = np.dot(x_t, w_ih.T) + b_ih
                gate_hh = np.dot(h_t, w_hh.T) + b_hh

                r_gate = self._sigmoid(
                    gate_ih[:, 0 * H : 1 * H] + gate_hh[:, 0 * H : 1 * H]
                )
                z_gate = self._sigmoid(
                    gate_ih[:, 1 * H : 2 * H] + gate_hh[:, 1 * H : 2 * H]
                )
                n_gate = np.tanh(
                    gate_ih[:, 2 * H : 3 * H] + r_gate * gate_hh[:, 2 * H : 3 * H]
                )

                h_t = (1.0 - z_gate) * n_gate + z_gate * h_t
                seq_out.append(h_t[:, np.newaxis, :])

            current_input = np.concatenate(seq_out, axis=1)
            h_prev_layers[layer] = h_t

        output_data = (
            current_input if self.batch_first else current_input.transpose(1, 0, 2)
        )
        h_n_data = np.stack(h_prev_layers, axis=0)

        return Tensor(output_data, requires_grad=x.requires_grad), Tensor(
            h_n_data, requires_grad=x.requires_grad
        )
