"""Multi-Agent Economy, Evolutionary Game Theory, and Combinatorial Auctions.

Formulated from first principles using Replicator-Mutator continuous population dynamics,
Vickrey-Clarke-Groves (VCG) combinatorial auction mechanisms, and Clarke pivot payments in pure NumPy.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class ReplicatorDynamics:
    r"""Continuous-Time Evolutionary Population Dynamics (Replicator-Mutator Engine).

    Models frequency-dependent selection and mutation in multi-agent game populations:
    \dot{x}_i = x_i (f_i(x) - ar{f}(x)) + \sum_j (\mu_{ji} x_j - \mu_{ij} x_i)

    Parameters
    ----------
    payoff_matrix : np.ndarray of shape (K, K)
        Payoff matrix A where A[i, j] is the payoff to strategy i against strategy j.
    mutation_rate : float, default=0.0
        Uniform mutation rate mu between strategies.
    dt : float, default=0.01
        Integration time step delta_t.
    """

    def __init__(
        self,
        payoff_matrix: np.ndarray,
        mutation_rate: float = 0.0,
        dt: float = 0.01,
    ) -> None:
        self.A = np.asarray(payoff_matrix, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"payoff_matrix must be square (K, K), got {self.A.shape}")

        self.num_strategies = self.A.shape[0]
        self.mutation_rate = float(mutation_rate)
        self.dt = float(dt)

    def fitness(self, population: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""Compute strategy fitness vector f(x) = A x and mean population fitness ar{f}(x) = x^T A x."""
        x = np.asarray(population, dtype=float).flatten()
        f_vec = np.dot(self.A, x)
        f_mean = float(np.dot(x, f_vec))
        return f_vec, f_mean

    def derivatives(self, population: np.ndarray) -> np.ndarray:
        r"""Compute time derivative \dot{x} on the probability simplex."""
        x = np.asarray(population, dtype=float).flatten()
        f_vec, f_mean = self.fitness(x)

        # Standard replicator selection: x_i * (f_i - f_mean)
        dx = x * (f_vec - f_mean)

        # Mutation drift
        if self.mutation_rate > 0.0 and self.num_strategies > 1:
            mut_in = np.sum(x) * (
                self.mutation_rate / (self.num_strategies - 1)
            ) - x * (self.mutation_rate / (self.num_strategies - 1))
            dx += mut_in - self.mutation_rate * x

        return dx

    def step(self, population: np.ndarray) -> np.ndarray:
        """Perform one RK4 integration step and project to unit simplex."""
        x = np.asarray(population, dtype=float).flatten()
        dt = self.dt

        k1 = self.derivatives(x)
        k2 = self.derivatives(x + 0.5 * dt * k1)
        k3 = self.derivatives(x + 0.5 * dt * k2)
        k4 = self.derivatives(x + dt * k3)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Project onto simplex
        x_next = np.maximum(0.0, x_next)
        s: float = float(np.sum(x_next))
        if s > 1e-12:
            x_next = x_next / s
        else:
            x_next = np.ones(self.num_strategies) / self.num_strategies

        return x_next

    def simulate(
        self,
        initial_population: np.ndarray,
        num_steps: int = 100,
    ) -> np.ndarray:
        """Simulate population trajectory over time."""
        x = np.asarray(initial_population, dtype=float).flatten()
        x = np.maximum(0.0, x)
        x = x / np.sum(x)

        trajectory = [x.copy()]
        for _ in range(num_steps):
            x = self.step(x)
            trajectory.append(x.copy())

        return np.array(trajectory)

    @classmethod
    def hawks_doves(
        cls, v: float = 2.0, c: float = 4.0, dt: float = 0.01
    ) -> "ReplicatorDynamics":
        """Factory for classical Hawk-Dove game with resource value V and fight cost C."""
        A = np.array(
            [
                [(v - c) / 2.0, v],
                [0.0, v / 2.0],
            ]
        )
        return cls(A, dt=dt)

    @classmethod
    def rock_paper_scissors(cls, dt: float = 0.01) -> "ReplicatorDynamics":
        """Factory for classical zero-sum Rock-Paper-Scissors cycle."""
        A = np.array(
            [
                [0.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        return cls(A, dt=dt)


class CombinatorialAuction:
    """Vickrey-Clarke-Groves (VCG) Combinatorial Bundle Auction Engine.

    Guarantees dominant-strategy incentive compatibility (DSIC / truth-telling)
    and social welfare maximization via Clarke pivot rule taxation.

    Parameters
    ----------
    items : List[str]
        List of distinct goods/items up for auction.
    bidders : List[str]
        List of participating agent identifiers.
    """

    def __init__(self, items: List[str], bidders: List[str]) -> None:
        self.items = list(items)
        self.bidders = list(bidders)
        # Bids stored as: list of (bidder, frozenset(items), value)
        self.bids: List[Tuple[str, frozenset, float]] = []

    def add_bid(self, bidder: str, item_bundle: List[str], bid_amount: float) -> None:
        """Submit a bundle valuation bid."""
        if bidder not in self.bidders:
            raise ValueError(f"Unknown bidder: {bidder}")
        bundle = frozenset(item_bundle)
        for itm in bundle:
            if itm not in self.items:
                raise ValueError(f"Unknown item in bundle: {itm}")
        self.bids.append((bidder, bundle, float(bid_amount)))

    def _solve_wdp(
        self,
        bids: List[Tuple[str, frozenset, float]],
        excluded_bidder: Optional[str] = None,
    ) -> Tuple[float, Dict[str, List[str]]]:
        """Solve Winner Determination Problem (WDP) maximizing total social welfare."""
        valid_bids = [
            b for b in bids if excluded_bidder is None or b[0] != excluded_bidder
        ]

        # Branch and bound / search over combinations of non-overlapping bids
        best_welfare = 0.0
        best_allocation: Dict[str, List[str]] = {
            b: [] for b in self.bidders if b != excluded_bidder
        }

        n_bids = len(valid_bids)
        if n_bids == 0:
            return 0.0, best_allocation

        # Helper recursive search
        def search(
            idx: int,
            current_items: Set[str],
            current_val: float,
            curr_alloc: Dict[str, List[str]],
        ) -> None:
            nonlocal best_welfare, best_allocation
            if idx == n_bids:
                if current_val > best_welfare:
                    best_welfare = current_val
                    best_allocation = {k: list(v) for k, v in curr_alloc.items()}
                return

            # Option 1: Skip bid idx
            search(idx + 1, current_items, current_val, curr_alloc)

            # Option 2: Accept bid idx if no item overlap and bidder doesn't already have disjoint win
            bidder, bundle, val = valid_bids[idx]
            if not (current_items & bundle) and not curr_alloc[bidder]:
                new_items = current_items | bundle
                curr_alloc[bidder] = list(bundle)
                search(idx + 1, new_items, current_val + val, curr_alloc)
                curr_alloc[bidder] = []

        initial_alloc: Dict[str, List[str]] = {
            b: [] for b in self.bidders if b != excluded_bidder
        }
        search(0, set(), 0.0, initial_alloc)
        return best_welfare, best_allocation

    def solve(self) -> Dict[str, Any]:
        """Solve the VCG auction: determine winning allocations and Clarke pivot payments.

        Returns
        -------
        result : Dict[str, Any]
            Dictionary containing:
            - allocations: Dict[bidder, List[items]]
            - payments: Dict[bidder, float]
            - social_welfare: float
        """
        # 1. Total social welfare with all bidders
        total_welfare, optimal_alloc = self._solve_wdp(self.bids)

        # 2. Clarke pivot payments for each winning bidder
        payments: Dict[str, float] = {b: 0.0 for b in self.bidders}

        for bidder in self.bidders:
            if optimal_alloc.get(bidder):  # Bidder won items
                # Welfare of others without bidder
                welfare_others_without, _ = self._solve_wdp(
                    self.bids, excluded_bidder=bidder
                )
                # Welfare of others in optimal allocation
                welfare_others_with = 0.0
                for b_other, items_won in optimal_alloc.items():
                    if b_other != bidder and items_won:
                        bundle_set = frozenset(items_won)
                        for b_id, b_bundle, b_val in self.bids:
                            if b_id == b_other and b_bundle == bundle_set:
                                welfare_others_with += b_val
                                break

                # VCG payment: p_i = W_{-i}^* - W_{-i}(alloc^*)
                payments[bidder] = max(
                    0.0, welfare_others_without - welfare_others_with
                )

        return {
            "allocations": optimal_alloc,
            "payments": payments,
            "social_welfare": float(total_welfare),
        }
