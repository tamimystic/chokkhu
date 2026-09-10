"""Advanced Reasoning Paradigms: Chain-of-Thought, Tree-of-Thoughts & Graph-of-Thoughts.

Pure Python / NumPy implementations of:
- ChainOfThought: Linear multi-step reasoning progression
- TreeOfThoughts (ToT): Tree search (BFS/DFS) over candidate reasoning branches
- GraphOfThoughts (GoT): Directed Acyclic Graph reasoning with vertex transformations and aggregation
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class ChainOfThought:
    """Chain-of-Thought (CoT) Sequential Reasoning Engine."""

    def __init__(self, steps: Optional[List[str]] = None) -> None:
        self.steps: List[str] = steps or []

    def add_step(self, thought: str) -> None:
        """Add reasoning step to chain."""
        self.steps.append(thought)

    def execute(self) -> str:
        """Concatenate reasoning steps into full rationale."""
        return "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(self.steps)])


class ThoughtNode:
    """Node in Tree-of-Thoughts (ToT) Search Graph."""

    def __init__(
        self,
        state: str,
        value: float = 0.0,
        parent: Optional["ThoughtNode"] = None,
    ) -> None:
        self.state = state
        self.value = value
        self.parent = parent
        self.children: List["ThoughtNode"] = []

    def add_child(self, state: str, value: float = 0.0) -> "ThoughtNode":
        child = ThoughtNode(state=state, value=value, parent=self)
        self.children.append(child)
        return child


class TreeOfThoughts:
    """Tree-of-Thoughts (ToT) Search Algorithm (Yao et al., 2023).

    Explores branches of thoughts via Breadth-First Search (BFS) or Depth-First Search (DFS).
    """

    def __init__(
        self,
        thought_generator: Callable[[str, int], List[str]],
        state_evaluator: Callable[[str], float],
        max_depth: int = 3,
        branching_factor: int = 3,
        search_algorithm: str = "bfs",
    ) -> None:
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.search_algorithm = search_algorithm

    def solve(self, initial_state: str) -> Tuple[str, float]:
        """Perform search over tree of thoughts and return best final state and score."""
        root = ThoughtNode(
            state=initial_state, value=self.state_evaluator(initial_state)
        )

        if self.search_algorithm == "bfs":
            current_level = [root]
            for depth in range(self.max_depth):
                next_level: List[ThoughtNode] = []
                for node in current_level:
                    candidates = self.thought_generator(
                        node.state, self.branching_factor
                    )
                    for c in candidates:
                        val = self.state_evaluator(c)
                        child = node.add_child(state=c, value=val)
                        next_level.append(child)

                if not next_level:
                    break
                # Keep top candidates
                next_level.sort(key=lambda n: n.value, reverse=True)
                current_level = next_level[: self.branching_factor]

            best_node = (
                max(current_level, key=lambda n: n.value) if current_level else root
            )
            return best_node.state, best_node.value

        else:
            # DFS Search
            best_state = initial_state
            best_value = root.value

            def dfs(node: ThoughtNode, depth: int) -> None:
                nonlocal best_state, best_value
                if depth >= self.max_depth:
                    if node.value > best_value:
                        best_value = node.value
                        best_state = node.state
                    return

                candidates = self.thought_generator(node.state, self.branching_factor)
                for c in candidates:
                    val = self.state_evaluator(c)
                    child = node.add_child(state=c, value=val)
                    dfs(child, depth + 1)

            dfs(root, 0)
            return best_state, best_value


class GraphOfThoughts:
    """Graph-of-Thoughts (GoT) Network Reasoning Engine.

    Allows merging, aggregating, and feedback loops across multiple reasoning vertices.
    """

    def __init__(self) -> None:
        self.vertices: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str]] = []

    def add_thought(self, vertex_id: str, content: str, score: float = 0.0) -> None:
        """Add reasoning vertex."""
        self.vertices[vertex_id] = {"content": content, "score": score}

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add directed dependency between thoughts."""
        if from_id in self.vertices and to_id in self.vertices:
            self.edges.append((from_id, to_id))

    def aggregate(self, vertex_ids: List[str], new_id: str) -> str:
        """Aggregate contents of multiple thoughts into a consolidated thought vertex."""
        contents = [
            self.vertices[vid]["content"] for vid in vertex_ids if vid in self.vertices
        ]
        scores = [
            self.vertices[vid]["score"] for vid in vertex_ids if vid in self.vertices
        ]
        agg_content = " + ".join(contents)
        agg_score = float(np.mean(scores)) if scores else 0.0

        self.add_thought(new_id, agg_content, agg_score)
        for vid in vertex_ids:
            self.add_edge(vid, new_id)

        return agg_content
