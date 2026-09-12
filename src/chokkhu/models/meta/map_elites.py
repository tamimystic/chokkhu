"""MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) in Pure NumPy.

References:
- Mouret & Clune (2015): "Illuminating search spaces by mapping elites" (Nature/arXiv:1504.04909).
- Cully et al. (2015): "Robots that can adapt like animals" (Nature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Individual:
    """An elite solution within the MAP-Elites behavioral repertoire."""

    genome: np.ndarray
    fitness: float
    behavior: np.ndarray
    cell_index: Tuple[int, ...]


@dataclass
class MAPElitesResult:
    """Outcome of a MAP-Elites Quality-Diversity illumination run."""

    archive: Dict[Tuple[int, ...], Individual]
    coverage: float
    max_fitness: float
    qd_score: float
    total_evaluations: int
    history_coverage: List[float]
    history_max_fitness: List[float]
    history_qd_score: List[float]


class MAPElites:
    """Multi-dimensional Archive of Phenotypic Elites (MAP-Elites) Quality-Diversity Engine.

    Partitions a continuous behavioral/phenotypic space into an N-dimensional discretized grid,
    illuminating the search space by discovering the highest-performing elite in each niche.
    """

    def __init__(
        self,
        genome_dim: int,
        behavior_dims: Tuple[int, ...],
        behavior_bounds: List[Tuple[float, float]],
        genome_bounds: Optional[Tuple[float, float]] = (-5.0, 5.0),
        mutation_sigma: float = 0.1,
        crossover_prob: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.genome_dim = int(genome_dim)
        self.behavior_dims = tuple(int(b) for b in behavior_dims)
        self.behavior_bounds = [
            (float(low), float(high)) for low, high in behavior_bounds
        ]
        self.genome_bounds = genome_bounds
        self.mutation_sigma = float(mutation_sigma)
        self.crossover_prob = float(crossover_prob)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        assert len(self.behavior_dims) == len(
            self.behavior_bounds
        ), "behavior_dims and behavior_bounds must have the same length"
        self.total_cells = int(np.prod(self.behavior_dims))
        self.archive: Dict[Tuple[int, ...], Individual] = {}

    def _get_cell_index(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """Map continuous behavior descriptor coordinates to discrete cell grid coordinates."""
        b_arr = np.asarray(behavior, dtype=np.float64)
        indices: List[int] = []

        for d, (low, high) in enumerate(self.behavior_bounds):
            num_bins = self.behavior_dims[d]
            val = float(b_arr[d])
            # Normalize to [0, 1]
            norm_val = (val - low) / max(high - low, 1e-12)
            cell_idx = int(np.floor(norm_val * num_bins))
            cell_idx = max(0, min(num_bins - 1, cell_idx))
            indices.append(cell_idx)

        return tuple(indices)

    def _random_genome(self) -> np.ndarray:
        """Sample a uniformly random genome within genome bounds."""
        low, high = self.genome_bounds if self.genome_bounds else (-5.0, 5.0)
        return self.rng.uniform(low, high, size=self.genome_dim).astype(np.float64)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation with bounds clipping."""
        mutated = genome + self.rng.randn(self.genome_dim) * self.mutation_sigma
        if self.genome_bounds:
            low, high = self.genome_bounds
            mutated = np.clip(mutated, low, high)
        return mutated.astype(np.float64)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Line recombination / Iso+LineDD crossover."""
        alpha = self.rng.uniform(0.0, 1.0)
        child = alpha * parent1 + (1.0 - alpha) * parent2
        return child.astype(np.float64)

    def optimize(
        self,
        eval_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        num_iterations: int = 500,
        initial_samples: int = 50,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> MAPElitesResult:
        """Execute Quality-Diversity illumination over the behavior space.

        Args:
            eval_fn: Function mapping genome array to (fitness: float, behavior: np.ndarray).
            num_iterations: Number of evolutionary generational cycles.
            initial_samples: Number of initial random exploratory individuals.
            batch_size: Number of offspring produced per generation.

        Returns:
            MAPElitesResult containing archive, coverage, max_fitness, and histories.
        """
        self.archive.clear()
        history_coverage: List[float] = []
        history_max_fitness: List[float] = []
        history_qd_score: List[float] = []
        total_evals = 0

        # 1. Initial random exploration
        for _ in range(initial_samples):
            genome = self._random_genome()
            fitness, behavior = eval_fn(genome)
            total_evals += 1
            cell_idx = self._get_cell_index(behavior)

            if cell_idx not in self.archive or fitness > self.archive[cell_idx].fitness:
                self.archive[cell_idx] = Individual(
                    genome=genome,
                    fitness=float(fitness),
                    behavior=np.asarray(behavior, dtype=np.float64),
                    cell_index=cell_idx,
                )

        # 2. Main illumination loop
        for iteration in range(num_iterations):
            if not self.archive:
                continue

            elites_list = list(self.archive.values())

            for _ in range(batch_size):
                # Select random parent(s) from existing archive
                p1_idx = self.rng.randint(len(elites_list))
                parent1 = elites_list[p1_idx].genome

                if len(elites_list) > 1 and self.rng.rand() < self.crossover_prob:
                    p2_idx = self.rng.randint(len(elites_list))
                    parent2 = elites_list[p2_idx].genome
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = np.copy(parent1)

                # Mutate offspring
                mutated_offspring = self._mutate(offspring)

                # Evaluate offspring
                fitness, behavior = eval_fn(mutated_offspring)
                total_evals += 1
                cell_idx = self._get_cell_index(behavior)

                # Elite replacement
                if (
                    cell_idx not in self.archive
                    or fitness > self.archive[cell_idx].fitness
                ):
                    self.archive[cell_idx] = Individual(
                        genome=mutated_offspring,
                        fitness=float(fitness),
                        behavior=np.asarray(behavior, dtype=np.float64),
                        cell_index=cell_idx,
                    )

            # Record metrics
            coverage = float(len(self.archive) / self.total_cells)
            all_fitnesses = [ind.fitness for ind in self.archive.values()]
            max_fit = float(np.max(all_fitnesses)) if all_fitnesses else float("-inf")
            qd_score = float(np.sum(all_fitnesses)) if all_fitnesses else 0.0

            history_coverage.append(coverage)
            history_max_fitness.append(max_fit)
            history_qd_score.append(qd_score)

            if verbose and (iteration + 1) % 50 == 0:
                print(
                    f"MAP-Elites Gen {iteration+1}/{num_iterations} - "
                    f"Coverage: {coverage*100:.1f}%, Max Fitness: {max_fit:.4f}, QD Score: {qd_score:.2f}"
                )

        final_coverage = float(len(self.archive) / self.total_cells)
        all_fitnesses = [ind.fitness for ind in self.archive.values()]
        final_max_fit = float(np.max(all_fitnesses)) if all_fitnesses else float("-inf")
        final_qd = float(np.sum(all_fitnesses)) if all_fitnesses else 0.0

        return MAPElitesResult(
            archive=self.archive,
            coverage=final_coverage,
            max_fitness=final_max_fit,
            qd_score=final_qd,
            total_evaluations=total_evals,
            history_coverage=history_coverage,
            history_max_fitness=history_max_fitness,
            history_qd_score=history_qd_score,
        )

    def get_elites(self) -> List[Individual]:
        """Return all discovered elites in the archive."""
        return list(self.archive.values())

    def get_best_overall(self) -> Optional[Individual]:
        """Return single highest-fitness individual across all behavioral niches."""
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda ind: ind.fitness)

    def get_archive_grid(self) -> np.ndarray:
        """Return N-dimensional numpy array of fitness values across the behavioral grid."""
        grid: np.ndarray = np.full(self.behavior_dims, np.nan, dtype=np.float64)
        for cell_idx, ind in self.archive.items():
            grid[cell_idx] = ind.fitness
        return grid
