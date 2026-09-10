"""Multi-Agent Coordination, Reflection & Self-Consistency Systems.

Pure Python / NumPy implementations of:
- MultiAgentCoordinator: Orchestrator managing collaborative agent societies
- ReflexionEngine: Episodic memory self-reflection and error recovery loop
- SelfConsistency: Majority voting and confidence aggregation across stochastic reasoning paths
"""

from typing import Any, Callable, Dict, List, Tuple
from collections import Counter

from .react import ReActAgent


class MultiAgentCoordinator:
    """Orchestrator for Multi-Agent Collaboration."""

    def __init__(self) -> None:
        self.agents: Dict[str, ReActAgent] = {}
        self.shared_memory: Dict[str, Any] = {}

    def register_agent(self, name: str, agent: ReActAgent) -> None:
        """Register agent with a designated role name."""
        self.agents[name] = agent

    def execute_workflow(
        self, initial_query: str, pipeline_order: List[str]
    ) -> Dict[str, Any]:
        """Execute sequential multi-agent workflow where agent i passes output to agent i+1."""
        current_input = initial_query
        workflow_history: Dict[str, Any] = {}

        for agent_name in pipeline_order:
            if agent_name not in self.agents:
                raise KeyError(f"Agent '{agent_name}' is not registered.")

            agent = self.agents[agent_name]
            result = agent.run(current_input)
            output = result.get("output", "")

            workflow_history[agent_name] = result
            current_input = output

        return {
            "final_output": current_input,
            "workflow": workflow_history,
        }


class ReflexionEngine:
    """Reflexion: Self-Reflection & Memory-Guided Error Correction (Shinn et al., 2023)."""

    def __init__(
        self,
        agent: ReActAgent,
        evaluator: Callable[[str], Tuple[bool, str]],
        max_reflections: int = 3,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.max_reflections = max_reflections
        self.reflection_memory: List[str] = []

    def run_with_reflexion(self, task: str) -> Dict[str, Any]:
        """Execute task with iterative self-reflection upon failure."""
        self.reflection_memory = []

        for attempt in range(self.max_reflections + 1):
            # Augment task prompt with prior reflections
            if self.reflection_memory:
                context = "\n".join(
                    [f"- Past failure reflection: {r}" for r in self.reflection_memory]
                )
                augmented_task = f"{task}\n\nPast Experience / Reflections:\n{context}"
            else:
                augmented_task = task

            res = self.agent.run(augmented_task)
            output = res.get("output", "")

            success, feedback = self.evaluator(output)
            if success:
                return {
                    "success": True,
                    "output": output,
                    "attempts": attempt + 1,
                    "reflections": self.reflection_memory,
                }

            # Record reflection
            self.reflection_memory.append(feedback)

        return {
            "success": False,
            "output": output,
            "attempts": self.max_reflections + 1,
            "reflections": self.reflection_memory,
        }


class SelfConsistency:
    """Self-Consistency Sampling & Majority Voting (Wang et al., 2022)."""

    @staticmethod
    def majority_vote(answers: List[str]) -> Tuple[str, float]:
        """Compute majority vote and agreement confidence from sample reasoning paths."""
        if not answers:
            return "", 0.0

        counter = Counter(answers)
        most_common, count = counter.most_common(1)[0]
        confidence = float(count) / float(len(answers))

        return most_common, confidence
