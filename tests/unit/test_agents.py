"""Unit tests for Autonomous AI Agents, Tool-Use & Multi-Agent Collaboration."""

import pytest

from chokkhu.models.nlp.agents import (
    Tool,
    ToolParameter,
    ToolRegistry,
    ReActAgent,
    ChainOfThought,
    TreeOfThoughts,
    GraphOfThoughts,
    MultiAgentCoordinator,
    ReflexionEngine,
    SelfConsistency,
)


def test_tool_and_registry() -> None:
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    param_a = ToolParameter(name="a", param_type="integer", description="First number")
    param_b = ToolParameter(name="b", param_type="integer", description="Second number")
    tool = Tool(
        name="add",
        description="Add two integers",
        func=add,
        parameters=[param_a, param_b],
    )

    registry.register(tool)

    assert registry.get("add") is not None
    assert len(registry.list_tools()) == 1

    schema = registry.get_schemas()[0]
    assert schema["name"] == "add"
    assert "parameters" in schema
    assert "a" in schema["parameters"]["properties"]

    # Dispatch execution
    result = registry.dispatch("add", a=5, b=10)
    assert result == 15

    with pytest.raises(KeyError):
        registry.dispatch("non_existent_tool")


def test_react_agent_execution() -> None:
    registry = ToolRegistry()

    def square(x: int) -> int:
        return x * x

    param_x = ToolParameter(
        name="x", param_type="integer", description="Number to square"
    )
    registry.register(
        Tool(
            name="square",
            description="Square a number",
            func=square,
            parameters=[param_x],
        )
    )

    # Mock deterministic LLM
    def mock_llm(prompt: str) -> str:
        if "Observation: 16" in prompt:
            return "Thought: The square of 4 is 16.\n" "Final Answer: 16"
        else:
            return (
                "Thought: I need to square the number 4.\n"
                "Action: square\n"
                'Action Input: {"x": 4}'
            )

    agent = ReActAgent(tools=registry, llm_callable=mock_llm, max_iterations=5)
    res = agent.run("What is 4 squared?")

    assert res["output"] == "16"
    assert res["iterations"] == 2
    assert len(res["trajectory"]) == 1
    assert res["trajectory"][0].action.tool == "square"
    assert res["trajectory"][0].observation == 16


def test_chain_of_thought() -> None:
    cot = ChainOfThought()
    cot.add_step("Identify user goal")
    cot.add_step("Formulate mathematical equation")
    cot.add_step("Solve equation and verify")

    rationale = cot.execute()
    assert "Step 1: Identify user goal" in rationale
    assert "Step 2: Formulate mathematical equation" in rationale
    assert "Step 3: Solve equation and verify" in rationale


def test_tree_of_thoughts_bfs_and_dfs() -> None:
    def thought_generator(state: str, num_branches: int) -> list[str]:
        return [f"{state} -> option_{i}" for i in range(num_branches)]

    def state_evaluator(state: str) -> float:
        # Give higher score if option_2 is selected
        return 1.0 if "option_2" in state else 0.5

    # BFS search
    tot_bfs = TreeOfThoughts(
        thought_generator=thought_generator,
        state_evaluator=state_evaluator,
        max_depth=2,
        branching_factor=3,
        search_algorithm="bfs",
    )
    best_state_bfs, best_val_bfs = tot_bfs.solve("Root")
    assert "option_2" in best_state_bfs
    assert best_val_bfs == 1.0

    # DFS search
    tot_dfs = TreeOfThoughts(
        thought_generator=thought_generator,
        state_evaluator=state_evaluator,
        max_depth=2,
        branching_factor=3,
        search_algorithm="dfs",
    )
    best_state_dfs, best_val_dfs = tot_dfs.solve("Root")
    assert best_val_dfs == 1.0


def test_graph_of_thoughts() -> None:
    got = GraphOfThoughts()
    got.add_thought("v1", "Hypothesis A", score=0.8)
    got.add_thought("v2", "Hypothesis B", score=0.6)

    agg = got.aggregate(["v1", "v2"], new_id="v3")
    assert agg == "Hypothesis A + Hypothesis B"
    assert "v3" in got.vertices
    assert pytest.approx(got.vertices["v3"]["score"], 1e-4) == 0.7
    assert ("v1", "v3") in got.edges
    assert ("v2", "v3") in got.edges


def test_multi_agent_coordinator() -> None:
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()

    def researcher_llm(prompt: str) -> str:
        return "Final Answer: Research summary: findings on topic"

    def writer_llm(prompt: str) -> str:
        return "Final Answer: Final structured article based on research summary"

    agent1 = ReActAgent(tools=registry1, llm_callable=researcher_llm)
    agent2 = ReActAgent(tools=registry2, llm_callable=writer_llm)

    coordinator = MultiAgentCoordinator()
    coordinator.register_agent("researcher", agent1)
    coordinator.register_agent("writer", agent2)

    workflow_res = coordinator.execute_workflow(
        initial_query="Explain Quantum Computing",
        pipeline_order=["researcher", "writer"],
    )

    assert "Final structured article" in workflow_res["final_output"]
    assert "researcher" in workflow_res["workflow"]
    assert "writer" in workflow_res["workflow"]


def test_reflexion_engine() -> None:
    registry = ToolRegistry()
    counter = 0

    def mock_agent_llm(prompt: str) -> str:
        nonlocal counter
        counter += 1
        if "Past failure reflection" in prompt:
            return "Final Answer: Correct Output"
        return "Final Answer: Buggy Output"

    agent = ReActAgent(tools=registry, llm_callable=mock_agent_llm)

    def evaluator(output: str) -> tuple[bool, str]:
        if "Correct" in output:
            return True, "Output is correct"
        return False, "Output contained buggy pattern, please fix"

    reflexion = ReflexionEngine(agent=agent, evaluator=evaluator, max_reflections=2)
    res = reflexion.run_with_reflexion(task="Generate code")

    assert res["success"] is True
    assert res["attempts"] == 2
    assert len(res["reflections"]) == 1


def test_self_consistency() -> None:
    answers = ["42", "42", "24", "42", "100"]
    winner, confidence = SelfConsistency.majority_vote(answers)

    assert winner == "42"
    assert pytest.approx(confidence, 1e-4) == 0.6  # 3 out of 5 = 60%
