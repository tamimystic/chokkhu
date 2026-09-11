"""STaR (Self-Taught Reasoner) & Reflexion Loop for Autonomous Self-Play Reasoning.

References:
    - Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning", NeurIPS 2022.
    - Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np


class STaR:
    """Self-Taught Reasoner (STaR) Bootstrapping Engine."""

    def __init__(self) -> None:
        pass

    def bootstrap(
        self,
        dataset: List[Dict[str, Any]],
        rationale_generator: Callable[[str, Optional[str]], Tuple[str, str]],
        verifier: Callable[[str, Any], bool],
        max_rationalizations: int = 2,
    ) -> List[Dict[str, Any]]:
        """Run STaR bootstrapping loop to filter and rationalize reasoning dataset.

        Args:
            dataset: List of items with 'prompt' and 'ground_truth'.
            rationale_generator: Callable (prompt, hint=None) -> (rationale, answer).
            verifier: Callable (answer, ground_truth) -> bool.
            max_rationalizations: Max retry rationalization attempts with ground truth hint.

        Returns:
            Curated dataset of verified (prompt, rationale, answer) tuples for fine-tuning.
        """
        curated: List[Dict[str, Any]] = []

        for item in dataset:
            prompt = item["prompt"]
            gt = item["ground_truth"]

            # 1. Direct generation attempt
            rationale, answer = rationale_generator(prompt, None)
            if verifier(answer, gt):
                curated.append({
                    "prompt": prompt,
                    "rationale": rationale,
                    "answer": answer,
                    "ground_truth": gt,
                    "rationalized": False,
                })
                continue

            # 2. Rationalization with ground truth hint if initial generation failed
            for attempt in range(max_rationalizations):
                hint = f"The correct answer is {gt}."
                rat_rationale, rat_answer = rationale_generator(prompt, hint)
                if verifier(rat_answer, gt):
                    curated.append({
                        "prompt": prompt,
                        "rationale": rat_rationale,
                        "answer": rat_answer,
                        "ground_truth": gt,
                        "rationalized": True,
                    })
                    break

        return curated


class ReflexionLoop:
    """Reflexion: Language Reasoning with Verbal Episodic Reinforcement."""

    def __init__(self, max_trials: int = 4) -> None:
        self.max_trials = int(max_trials)
        self.memory: List[str] = []

    def run(
        self,
        prompt: str,
        executor: Callable[[str, List[str]], Tuple[str, Any]],
        evaluator: Callable[[Any], Tuple[bool, str]],
        reflector: Callable[[str, str, str], str],
    ) -> Dict[str, Any]:
        """Execute iterative trial-and-error reasoning with linguistic self-reflection.

        Args:
            prompt: Task statement.
            executor: Callable (prompt, memory_reflections) -> (trajectory_text, output).
            evaluator: Callable (output) -> (is_success, error_feedback).
            reflector: Callable (prompt, trajectory_text, error_feedback) -> reflection_critique.

        Returns:
            Result dict with 'success', 'output', 'trials', 'reflections'.
        """
        self.memory = []

        for trial in range(1, self.max_trials + 1):
            trajectory_text, output = executor(prompt, self.memory)
            is_success, feedback = evaluator(output)

            if is_success:
                return {
                    "success": True,
                    "output": output,
                    "trials": trial,
                    "reflections": list(self.memory),
                }

            # Generate verbal reflection critique
            critique = reflector(prompt, trajectory_text, feedback)
            self.memory.append(critique)

        return {
            "success": False,
            "output": output,
            "trials": self.max_trials,
            "reflections": list(self.memory),
        }
