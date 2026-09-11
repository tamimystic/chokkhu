"""Game Theory, Normal-Form Matrix Games, and Nash Equilibrium Solver in pure NumPy."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class NashEquilibriumSolver:
    """Nash Equilibrium Solver for Two-Player Normal-Form Matrix Games in pure NumPy.

    Computes optimal mixed-strategy profiles and game values for zero-sum
    and general-sum bi-matrix games via fictitious play and linear programming.
    """

    def __init__(self, max_iter: int = 2000, tol: float = 1e-5) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def solve_zero_sum(
        self, payoff_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""Finds mixed-strategy Nash Equilibrium for 2-player zero-sum game using Fictitious Play.

        Parameters
        ----------
        payoff_matrix : np.ndarray
            Payoff matrix $M \in \mathbb{R}^{m \times n}$ for Row player (Col player gets $-M$).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            (row_strategy (m,), col_strategy (n,), game_value)
        """
        M = np.asarray(payoff_matrix, dtype=np.float64)
        m, n = M.shape

        row_counts = np.zeros(m, dtype=np.float64)
        col_counts = np.zeros(n, dtype=np.float64)

        # Initial choices
        row_action = 0
        col_action = 0
        row_counts[row_action] += 1.0
        col_counts[col_action] += 1.0

        for t in range(1, self.max_iter):
            # Row player best response to empirical col strategy
            p_col = col_counts / t
            row_payoffs = np.matmul(M, p_col)
            row_action = int(np.argmax(row_payoffs))

            # Col player best response to empirical row strategy (minimizes Row payoff)
            p_row = row_counts / t
            col_payoffs = np.matmul(p_row, M)
            col_action = int(np.argmin(col_payoffs))

            row_counts[row_action] += 1.0
            col_counts[col_action] += 1.0

        p_row_opt = row_counts / np.sum(row_counts)
        p_col_opt = col_counts / np.sum(col_counts)
        game_val = float(np.dot(p_row_opt, np.dot(M, p_col_opt)))

        return p_row_opt, p_col_opt, game_val

    def solve_bimatrix(
        self, payoff_matrix_a: np.ndarray, payoff_matrix_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """Computes mixed-strategy Nash Equilibrium for general-sum bi-matrix game (A, B)."""
        A = np.asarray(payoff_matrix_a, dtype=np.float64)
        B = np.asarray(payoff_matrix_b, dtype=np.float64)
        m, n = A.shape

        row_counts = np.zeros(m, dtype=np.float64)
        col_counts = np.zeros(n, dtype=np.float64)

        row_counts[0] = 1.0
        col_counts[0] = 1.0

        for t in range(1, self.max_iter):
            p_col = col_counts / t
            a_payoffs = np.matmul(A, p_col)
            row_counts[int(np.argmax(a_payoffs))] += 1.0

            p_row = row_counts / t
            b_payoffs = np.matmul(p_row, B)
            col_counts[int(np.argmax(b_payoffs))] += 1.0

        p_a = row_counts / np.sum(row_counts)
        p_b = col_counts / np.sum(col_counts)

        val_a = float(np.dot(p_a, np.dot(A, p_b)))
        val_b = float(np.dot(p_a, np.dot(B, p_b)))

        return p_a, p_b, (val_a, val_b)
