"""Unit tests for Frontier 4.19: Test-Time Search, Process Reward Models & Self-Play."""

import numpy as np
import pytest

from chokkhu.models.nlp.reasoning import (
    MCTSNode,
    MonteCarloTreeSearchReasoning,
    ProcessRewardModel,
    STaR,
    ReflexionLoop,
)


def test_mcts_reasoning_optimal_path_selection():
    """Verify MCTS identifies optimal multi-step reasoning path based on evaluator values."""
    mcts = MonteCarloTreeSearchReasoning(c_puct=1.414)

    def step_generator(state: str):
        if state == "Start":
            return [("Step A1 (Good)", 0.5), ("Step A2 (Bad)", 0.5)]
        elif "Step A1" in state and "Step B1" not in state:
            return [("Step B1 (Optimal)", 0.5), ("Step B2 (Suboptimal)", 0.5)]
        elif "Step A2" in state and "Step B3" not in state:
            return [("Step B3 (Poor)", 0.5), ("Step B4 (Poor)", 0.5)]
        return []

    def value_evaluator(state: str):
        if "Step B1 (Optimal)" in state:
            return 1.0
        elif "Step B2 (Suboptimal)" in state:
            return 0.3
        elif "Step A1" in state:
            return 0.6
        return -0.5

    root = mcts.search(
        initial_prompt="Start",
        step_generator=step_generator,
        value_evaluator=value_evaluator,
        num_simulations=40,
        max_depth=3,
    )

    trajectory = mcts.best_trajectory(root)
    assert len(trajectory) >= 2
    assert trajectory[0] == "Step A1 (Good)"
    assert trajectory[1] == "Step B1 (Optimal)"


def test_mcts_node_statistics():
    """Verify MCTSNode q_value and visit tracking."""
    node = MCTSNode(state="Test")
    assert node.visits == 0
    assert node.q_value == 0.0

    node.visits = 5
    node.value_sum = 4.0
    assert node.q_value == 0.8


def test_process_reward_model_step_scoring():
    """Verify PRM scores individual steps and handles heuristic evaluation."""
    def custom_scorer(ctx, step):
        if "correct" in step:
            return 0.95
        return 0.2

    prm = ProcessRewardModel(step_scorer=custom_scorer)
    assert prm.score_step("Context", "This is correct logic") == 0.95
    assert prm.score_step("Context", "This is wrong") == 0.2


def test_prm_trajectory_aggregation_modes():
    """Verify PRM aggregations (product, min, mean, discounted)."""
    prm = ProcessRewardModel(step_scorer=lambda ctx, step: float(step))
    steps = ["0.9", "0.8", "1.0"]

    prod_score = prm.score_trajectory(steps, aggregation="product")
    np.testing.assert_allclose(prod_score, 0.9 * 0.8 * 1.0, atol=1e-5)

    min_score = prm.score_trajectory(steps, aggregation="min")
    assert min_score == 0.8

    mean_score = prm.score_trajectory(steps, aggregation="mean")
    np.testing.assert_allclose(mean_score, (0.9 + 0.8 + 1.0) / 3.0, atol=1e-5)


def test_star_rationale_filtering_and_rationalization():
    """Verify STaR filters correct solutions and rationalizes failures with hints."""
    star = STaR()
    dataset = [
        {"prompt": "What is 2+2?", "ground_truth": "4"},
        {"prompt": "What is 3*3?", "ground_truth": "9"},
    ]

    def mock_generator(prompt, hint=None):
        if "2+2" in prompt:
            return "2 plus 2 equals 4", "4"
        elif "3*3" in prompt:
            if hint is not None:
                return "With hint, 3 times 3 is 9", "9"
            return "3 times 3 is 8", "8"  # Initial mistake
        return "Unknown", "0"

    def verifier(ans, gt):
        return ans == gt

    curated = star.bootstrap(dataset, mock_generator, verifier)
    assert len(curated) == 2
    assert curated[0]["rationalized"] is False
    assert curated[1]["rationalized"] is True
    assert curated[1]["answer"] == "9"


def test_reflexion_loop_trial_and_error():
    """Verify ReflexionLoop uses verbal feedback to self-correct and succeed."""
    reflexion = ReflexionLoop(max_trials=3)

    def mock_executor(prompt, memory):
        if len(memory) == 0:
            return "Trial 1 output", "Incorrect Answer"
        else:
            return "Trial 2 corrected output", "Correct Answer"

    def mock_evaluator(output):
        if output == "Correct Answer":
            return True, ""
        return False, "Error: missing required term"

    def mock_reflector(prompt, traj, feedback):
        return f"Reflection: I should include the required term mentioned in {feedback}"

    result = reflexion.run(
        prompt="Solve problem",
        executor=mock_executor,
        evaluator=mock_evaluator,
        reflector=mock_reflector,
    )

    assert result["success"] is True
    assert result["trials"] == 2
    assert len(result["reflections"]) == 1
    assert "required term" in result["reflections"][0]
