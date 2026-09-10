"""Quantum Circuit Simulator from first principles in pure NumPy."""

from __future__ import annotations

# no unused types
import numpy as np


class QuantumCircuit:
    """N-Qubit Quantum Circuit simulator based on complex state vector evolution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum circuit (N >= 1).
    """

    # Basic single-qubit gate matrices
    I2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    H_GATE = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z_GATE = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)

    def __init__(self, n_qubits: int = 1) -> None:
        if n_qubits < 1:
            raise ValueError(f"Number of qubits must be >= 1, got {n_qubits}")
        self.n_qubits = int(n_qubits)
        self.dim = 2**self.n_qubits
        self.reset()

    def reset(self) -> None:
        """Reset state vector to |0...0>."""
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j

    def set_state(self, state: np.ndarray) -> None:
        """Set explicit state vector (must be normalized)."""
        s = np.asarray(state, dtype=np.complex128).flatten()
        if len(s) != self.dim:
            raise ValueError(
                f"State vector length {len(s)} does not match circuit dimension {self.dim}"
            )
        norm = np.linalg.norm(s)
        if norm < 1e-12:
            raise ValueError("State vector norm cannot be zero.")
        self.state = s / norm

    def _apply_1q_gate(self, gate_matrix: np.ndarray, target: int) -> None:
        """Apply a single-qubit 2x2 unitary gate to the specified target qubit."""
        if not (0 <= target < self.n_qubits):
            raise IndexError(
                f"Target qubit {target} out of range [0, {self.n_qubits - 1}]"
            )

        # Build full 2^N x 2^N operator using Kronecker products
        op = np.array([[1.0]], dtype=np.complex128)
        for q in range(self.n_qubits):
            if q == target:
                op = np.kron(op, gate_matrix)
            else:
                op = np.kron(op, self.I2)

        self.state = np.dot(op, self.state)

    def h(self, target: int) -> QuantumCircuit:
        """Apply Hadamard gate."""
        self._apply_1q_gate(self.H_GATE, target)
        return self

    def x(self, target: int) -> QuantumCircuit:
        """Apply Pauli-X (NOT) gate."""
        self._apply_1q_gate(self.X_GATE, target)
        return self

    def y(self, target: int) -> QuantumCircuit:
        """Apply Pauli-Y gate."""
        self._apply_1q_gate(self.Y_GATE, target)
        return self

    def z(self, target: int) -> QuantumCircuit:
        """Apply Pauli-Z gate."""
        self._apply_1q_gate(self.Z_GATE, target)
        return self

    def s(self, target: int) -> QuantumCircuit:
        """Apply Phase (S) gate."""
        self._apply_1q_gate(self.S_GATE, target)
        return self

    def t(self, target: int) -> QuantumCircuit:
        """Apply T gate."""
        self._apply_1q_gate(self.T_GATE, target)
        return self

    def rx(self, theta: float, target: int) -> QuantumCircuit:
        """Apply parameterized X-rotation gate Rx(theta) = exp(-i * theta/2 * X)."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        rx_mat = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        self._apply_1q_gate(rx_mat, target)
        return self

    def ry(self, theta: float, target: int) -> QuantumCircuit:
        """Apply parameterized Y-rotation gate Ry(theta) = exp(-i * theta/2 * Y)."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        ry_mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self._apply_1q_gate(ry_mat, target)
        return self

    def rz(self, theta: float, target: int) -> QuantumCircuit:
        """Apply parameterized Z-rotation gate Rz(theta) = exp(-i * theta/2 * Z)."""
        e_neg = np.exp(-1j * theta / 2.0)
        e_pos = np.exp(1j * theta / 2.0)
        rz_mat = np.array([[e_neg, 0], [0, e_pos]], dtype=np.complex128)
        self._apply_1q_gate(rz_mat, target)
        return self

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """Apply Controlled-NOT (CNOT / CX) gate."""
        if control == target:
            raise ValueError("Control and target qubits must be distinct.")
        if not (0 <= control < self.n_qubits and 0 <= target < self.n_qubits):
            raise IndexError("Qubit index out of range.")

        # Projector |0><0| and |1><1|
        P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)

        # Term 0: |0><0|_c (x) I_t
        term0 = np.array([[1.0]], dtype=np.complex128)
        # Term 1: |1><1|_c (x) X_t
        term1 = np.array([[1.0]], dtype=np.complex128)

        for q in range(self.n_qubits):
            if q == control:
                term0 = np.kron(term0, P0)
                term1 = np.kron(term1, P1)
            elif q == target:
                term0 = np.kron(term0, self.I2)
                term1 = np.kron(term1, self.X_GATE)
            else:
                term0 = np.kron(term0, self.I2)
                term1 = np.kron(term1, self.I2)

        cnot_op = term0 + term1
        self.state = np.dot(cnot_op, self.state)
        return self

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Apply Controlled-Z gate."""
        self.h(target)
        self.cnot(control, target)
        self.h(target)
        return self

    def swap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply SWAP gate between two qubits."""
        self.cnot(qubit1, qubit2)
        self.cnot(qubit2, qubit1)
        self.cnot(qubit1, qubit2)
        return self

    def probabilities(self) -> np.ndarray:
        """Return computational basis state probabilities |amplitude|^2."""
        return np.abs(self.state) ** 2

    def expectation_z(self, target: int) -> float:
        """Calculate expectation value <Z> on a target qubit."""
        probs = self.probabilities()
        exp_val = 0.0
        for i in range(self.dim):
            # Check target qubit bit (0 -> +1, 1 -> -1)
            # In big-endian representation, bit is (i >> (n_qubits - 1 - target)) & 1
            bit = (i >> (self.n_qubits - 1 - target)) & 1
            sign = 1.0 if bit == 0 else -1.0
            exp_val += sign * float(probs[i])
        return float(exp_val)
