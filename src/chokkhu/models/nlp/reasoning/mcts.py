"""Monte Carlo Tree Search (MCTS) for Step-Level Chain-of-Thought Reasoning.

References:
    - Silver et al., "Mastering the Game of Go without Human Knowledge", Nature 2017 (PUCT).
    - Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np


class MCTSNode:
    """Node in the reasoning tree holding state, visit statistics, and children."""

    def __init__(
        self,
        state: str,
        parent: Optional["MCTSNode"] = None,
        action: Optional[str] = None,
        prior: float = 1.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action  # Step text that led to this state
        self.prior = float(prior)
        self.children: Dict[str, "MCTSNode"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_terminal = False
        self.reward = 0.0

    @property
    def q_value(self) -> float:
        """Average action value Q(s)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MonteCarloTreeSearchReasoning:
    """MCTS Reasoning engine implementing PUCT tree search over reasoning steps.

    Formula:
        UCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
    """

    def __init__(self, c_puct: float = 1.414) -> None:
        self.c_puct = float(c_puct)

    def select_child(self, node: MCTSNode) -> Tuple[str, MCTSNode]:
        """Select child maximizing PUCT exploration-exploitation formula."""
        total_visits = sum(child.visits for child in node.children.values())
        sqrt_total = np.sqrt(max(1, total_visits))

        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            u_score = self.c_puct * child.prior * sqrt_total / (1.0 + child.visits)
            puct_score = child.q_value + u_score

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore[return-value]

    def search(
        self,
        initial_prompt: str,
        step_generator: Callable[[str], List[Tuple[str, float]]],
        value_evaluator: Callable[[str], float],
        terminal_checker: Optional[Callable[[str], Tuple[bool, float]]] = None,
        num_simulations: int = 30,
        max_depth: int = 8,
    ) -> MCTSNode:
        """Execute MCTS search over reasoning steps.

        Args:
            initial_prompt: Problem statement / root state string.
            step_generator: Function mapping state string -> list of (candidate_step_text, prior_prob).
            value_evaluator: Function evaluating leaf state quality -> float in [-1.0, 1.0].
            terminal_checker: Optional function checking if state is terminal -> (is_term, terminal_reward).
            num_simulations: Number of MCTS simulation rollouts.
            max_depth: Maximum reasoning depth.

        Returns:
            Root node with populated search tree statistics.
        """
        root = MCTSNode(state=initial_prompt)

        for _ in range(num_simulations):
            node = root
            depth = 0

            # 1. Selection: Traverse down to unexpanded node or terminal leaf
            while node.is_expanded() and not node.is_terminal and depth < max_depth:
                _, node = self.select_child(node)
                depth += 1

            # 2. Check Terminal
            value = 0.0
            if terminal_checker is not None:
                is_term, term_reward = terminal_checker(node.state)
                if is_term:
                    node.is_terminal = True
                    node.reward = term_reward
                    value = term_reward

            # 3. Expansion: Expand children if not terminal
            if not node.is_terminal and depth < max_depth:
                candidates = step_generator(node.state)
                for step_text, prior in candidates:
                    new_state = f"{node.state}\n{step_text}" if node.state else step_text
                    child_node = MCTSNode(
                        state=new_state,
                        parent=node,
                        action=step_text,
                        prior=prior,
                    )
                    node.children[step_text] = child_node

                # 4. Evaluation: Evaluate leaf value
                value = value_evaluator(node.state)

            # 5. Backpropagation: Propagate value up to root
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += value
                curr = curr.parent

        return root

    def best_trajectory(self, root: MCTSNode) -> List[str]:
        """Extract most visited reasoning trajectory from search root."""
        trajectory: List[str] = []
        curr = root

        while curr.is_expanded():
            # Pick child with highest visit count
            best_action = max(curr.children.keys(), key=lambda a: curr.children[a].visits)
            trajectory.append(best_action)
            curr = curr.children[best_action]
            if curr.is_terminal:
                break

        return trajectory
