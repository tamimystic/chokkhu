"""Symbolic Feature Synthesis (Genetic Programming Feature Engineering) in pure NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SymbolicProgram:
    """Mathematical expression program tree for feature synthesis."""

    op: str  # "add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "tanh", "var"
    feature_idx: int = 0
    left: Optional[SymbolicProgram] = None
    right: Optional[SymbolicProgram] = None
    fitness: float = -1.0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluates program expression tree over input feature matrix X."""
        if self.op == "var":
            return X[:, self.feature_idx]

        if self.left is None:
            return np.zeros(len(X), dtype=np.float64)

        val_l = self.left.evaluate(X)

        # Unary operators
        if self.op == "sqrt":
            return np.sqrt(np.abs(val_l) + 1e-6)
        elif self.op == "log":
            return np.log1p(np.abs(val_l))
        elif self.op == "sin":
            return np.sin(val_l)
        elif self.op == "cos":
            return np.cos(val_l)
        elif self.op == "tanh":
            return np.tanh(val_l)

        # Binary operators
        if self.right is None:
            return val_l

        val_r = self.right.evaluate(X)
        if self.op == "add":
            return val_l + val_r
        elif self.op == "sub":
            return val_l - val_r
        elif self.op == "mul":
            return val_l * val_r
        elif self.op == "div":
            # Protected division
            denom = np.where(np.abs(val_r) < 1e-4, 1.0, val_r)
            return val_l / denom
        else:
            return val_l

    def to_formula(self, feature_names: Optional[List[str]] = None) -> str:
        """Returns human-readable mathematical formula representation."""
        if self.op == "var":
            return (
                feature_names[self.feature_idx]
                if feature_names
                else f"x_{self.feature_idx}"
            )
        if self.left is None:
            return "0"
        str_l = self.left.to_formula(feature_names)
        if self.op in ["sqrt", "log", "sin", "cos", "tanh"]:
            return f"{self.op}({str_l})"
        if self.right is not None:
            str_r = self.right.to_formula(feature_names)
            op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(
                self.op, self.op
            )
            return f"({str_l} {op_sym} {str_r})"
        return str_l


class SymbolicFeatureSynthesizer:
    r"""Genetic Programming Automated Feature Synthesizer in pure NumPy.

    Evolves non-linear symbolic mathematical feature trees ($+, -, 	imes, \div, \sqrt{\cdot}, \log, \sin$)
    to uncover high-information non-linear feature interactions and transformations.

    Parameters
    ----------
    n_features : int, default=5
        Number of top synthesized features to generate and retain.
    generations : int, default=4
        Number of evolutionary generations.
    population_size : int, default=20
        Number of candidate expression programs per generation.
    tournament_size : int, default=3
        Size of Pareto tournament selection pool.
    random_state : int | None, default=42
    """

    def __init__(
        self,
        n_features: int = 5,
        generations: int = 4,
        population_size: int = 20,
        tournament_size: int = 3,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_features = int(n_features)
        self.generations = int(generations)
        self.population_size = int(population_size)
        self.tournament_size = int(tournament_size)
        self.rng = np.random.default_rng(random_state)

        self.best_programs_: List[SymbolicProgram] = []
        self.feature_names_: List[str] = []

    def _random_program(
        self, n_cols: int, depth: int = 0, max_depth: int = 2
    ) -> SymbolicProgram:
        """Samples a random recursive program tree."""
        if depth >= max_depth or (depth > 0 and self.rng.random() < 0.4):
            idx = int(self.rng.integers(0, n_cols))
            return SymbolicProgram(op="var", feature_idx=idx)

        op = str(
            self.rng.choice(["add", "sub", "mul", "div", "sqrt", "log", "sin", "tanh"])
        )
        left = self._random_program(n_cols, depth + 1, max_depth)
        right = (
            self._random_program(n_cols, depth + 1, max_depth)
            if op in ["add", "sub", "mul", "div"]
            else None
        )
        return SymbolicProgram(op=op, left=left, right=right)

    def _evaluate_fitness(
        self, prog: SymbolicProgram, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Computes Pearson correlation fitness between synthesized feature and target y."""
        try:
            feat = prog.evaluate(X)
            if not np.all(np.isfinite(feat)) or np.std(feat) < 1e-6:
                return 0.0
            # Pearson correlation
            corr = np.corrcoef(feat, y.flatten())[0, 1]
            return float(abs(corr)) if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> SymbolicFeatureSynthesizer:
        """Evolves population of symbolic programs to discover informative feature representations."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_cols = X_arr.shape

        population = [self._random_program(n_cols) for _ in range(self.population_size)]

        for _ in range(self.generations):
            for prog in population:
                prog.fitness = self._evaluate_fitness(prog, X_arr, y_arr)

            population.sort(key=lambda p: p.fitness, reverse=True)

            next_pop = [population[0], population[1]]  # Elitism
            while len(next_pop) < self.population_size:
                candidates = self.rng.choice(
                    population,
                    size=min(self.tournament_size, len(population)),
                    replace=False,
                )
                winner = max(candidates, key=lambda p: p.fitness)

                # Mutate winner or generate fresh tree
                if self.rng.random() < 0.5:
                    child = self._random_program(n_cols, max_depth=2)
                else:
                    child = SymbolicProgram(
                        op=str(self.rng.choice(["sqrt", "log", "tanh", "sin"])),
                        left=winner,
                    )
                next_pop.append(child)

            population = next_pop

        # Select top unique programs
        for prog in population:
            prog.fitness = self._evaluate_fitness(prog, X_arr, y_arr)
        population.sort(key=lambda p: p.fitness, reverse=True)

        self.best_programs_ = population[: self.n_features]
        self.feature_names_ = [p.to_formula(feature_names) for p in self.best_programs_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Evaluates evolved programs on X to synthesize new non-linear feature columns."""
        X_arr = np.asarray(X, dtype=np.float64)
        synthesized: List[np.ndarray] = []

        for prog in self.best_programs_:
            feat = prog.evaluate(X_arr)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            synthesized.append(feat)

        return (
            np.column_stack(synthesized)
            if len(synthesized) > 0
            else np.zeros((len(X_arr), 0))
        )

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fits synthesizer and transforms X into synthesized feature matrix."""
        self.fit(X, y, feature_names=feature_names)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        """Returns list of mathematical formulas for evolved synthesized features."""
        return self.feature_names_
