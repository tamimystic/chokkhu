"""Process Reward Model (PRM) for Step-Level Verification.

Reference:
    Lightman et al., "Let's Verify Step by Step", OpenAI 2023 (PRM800K).
"""

from typing import List, Optional, Callable
import numpy as np


class ProcessRewardModel:
    """Process Reward Model scoring correctness probabilities for individual reasoning steps."""

    def __init__(
        self, step_scorer: Optional[Callable[[str, str], float]] = None
    ) -> None:
        """Initialize PRM.

        Args:
            step_scorer: Optional function (context, step) -> float in [0.0, 1.0].
        """
        self.step_scorer = step_scorer

    def score_step(self, context: str, step: str) -> float:
        """Score single reasoning step given preceding context."""
        if self.step_scorer is not None:
            return float(np.clip(self.step_scorer(context, step), 0.0, 1.0))
        # Default heuristic: check if step is non-empty
        return 1.0 if step.strip() else 0.0

    def score_trajectory(
        self,
        steps: List[str],
        context: str = "",
        aggregation: str = "product",
        discount: float = 0.95,
    ) -> float:
        """Score full trajectory of reasoning steps using specified aggregation mode.

        Args:
            steps: List of reasoning step strings.
            context: Initial problem statement.
            aggregation: Aggregation mode ('product', 'min', 'mean', 'discounted').
            discount: Discount factor for 'discounted' mode.

        Returns:
            Scalar trajectory verification score in [0.0, 1.0].
        """
        if not steps:
            return 0.0

        scores: List[float] = []
        curr_context = context

        for step in steps:
            s = self.score_step(curr_context, step)
            scores.append(s)
            curr_context = f"{curr_context}\n{step}" if curr_context else step

        scores_arr = np.array(scores, dtype=np.float64)

        if aggregation == "product":
            return float(np.prod(scores_arr))
        elif aggregation == "min":
            return float(np.min(scores_arr))
        elif aggregation == "mean":
            return float(np.mean(scores_arr))
        elif aggregation == "discounted":
            weights: np.ndarray = np.array(
                [float(discount**i) for i in range(len(scores_arr))], dtype=np.float64
            )
            return float(np.sum(scores_arr * weights) / float(np.sum(weights)))
        else:
            raise ValueError(f"Unsupported aggregation '{aggregation}'")
