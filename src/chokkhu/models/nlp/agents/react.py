"""ReAct (Reason + Act + Observe) Autonomous Agent Execution Engine.

Pure Python / NumPy implementation of Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models":
- Interleaved Thought, Action, and Observation execution cycles
- Safe trajectory tracing and step memory
- Dynamic loop termination and error recovery
"""

from typing import Any, Callable, Dict, List, Optional, Union
import json
import re

from .tools import ToolRegistry


class AgentAction:
    """Represents an action chosen by the agent to execute a tool."""

    def __init__(self, tool: str, tool_input: Dict[str, Any], log: str = "") -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

    def __repr__(self) -> str:
        return f"AgentAction(tool='{self.tool}', tool_input={self.tool_input})"


class AgentFinish:
    """Represents final response when agent completes its objective."""

    def __init__(self, return_values: Dict[str, Any], log: str = "") -> None:
        self.return_values = return_values
        self.log = log

    def __repr__(self) -> str:
        return f"AgentFinish(return_values={self.return_values})"


class AgentStep:
    """A single execution step recording Thought, Action, and Observation."""

    def __init__(self, action: AgentAction, observation: Any) -> None:
        self.action = action
        self.observation = observation

    def __repr__(self) -> str:
        return f"AgentStep(action={self.action}, observation={self.observation})"


class ReActAgent:
    """ReAct (Reasoning and Acting) Autonomous Execution Loop.

    Parameters
    ----------
    tools : ToolRegistry
        Registry containing available tools.
    llm_callable : optional callable
        Function taking prompt string and returning raw text generation.
    max_iterations : int, default=10
        Maximum allowed loop iterations before forced termination.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        llm_callable: Optional[Callable[[str], str]] = None,
        max_iterations: int = 10,
    ) -> None:
        self.tools = tools
        self.llm_callable = llm_callable
        self.max_iterations = max_iterations
        self.trajectory: List[AgentStep] = []

    def _format_prompt(self, user_query: str, history: List[AgentStep]) -> str:
        tools_desc = "\n".join(
            [f"- {t.name}: {t.description}" for t in self.tools.list_tools()]
        )
        prompt = (
            f"Answer the following query using the available tools.\n"
            f"Available Tools:\n{tools_desc}\n\n"
            f"Use the format:\n"
            f"Thought: [your reasoning]\n"
            f"Action: [tool_name]\n"
            f'Action Input: {{"param": value}}\n'
            f"Observation: [result]\n"
            f"... (repeat Thought/Action/Action Input/Observation as needed)\n"
            f"Thought: I know the final answer\n"
            f"Final Answer: [final result]\n\n"
            f"Question: {user_query}\n"
        )
        for step in history:
            prompt += f"{step.action.log}\nObservation: {step.observation}\n"
        return prompt

    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            ans = text.split("Final Answer:", 1)[1].strip()
            return AgentFinish(return_values={"output": ans}, log=text)

        # Parse Action and Action Input
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text)
        input_match = re.search(
            r"Action Input:\s*(\{.*?\}|.*?)(?=\n|$)", text, re.DOTALL
        )

        if action_match:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip() if input_match else "{}"
            try:
                tool_input = json.loads(raw_input)
            except Exception:
                tool_input = {"query": raw_input}

            return AgentAction(tool=tool_name, tool_input=tool_input, log=text)

        # Fallback finish
        return AgentFinish(return_values={"output": text.strip()}, log=text)

    def run(self, user_query: str) -> Dict[str, Any]:
        """Execute the full ReAct iterative reasoning loop."""
        self.trajectory = []

        for iteration in range(self.max_iterations):
            prompt = self._format_prompt(user_query, self.trajectory)

            if self.llm_callable is not None:
                llm_output = self.llm_callable(prompt)
            else:
                # Mock heuristic step for standalone execution
                if not self.trajectory and self.tools.list_tools():
                    first_tool = self.tools.list_tools()[0]
                    llm_output = f'Thought: I need to use {first_tool.name}.\nAction: {first_tool.name}\nAction Input: {{"query": "{user_query}"}}'
                else:
                    llm_output = f"Thought: I have solved the task.\nFinal Answer: Processed '{user_query}' successfully."

            decision = self._parse_output(llm_output)

            if isinstance(decision, AgentFinish):
                return {
                    "output": decision.return_values.get("output", ""),
                    "iterations": iteration + 1,
                    "trajectory": self.trajectory,
                }

            # Execute action
            try:
                observation = self.tools.dispatch(decision.tool, **decision.tool_input)
            except Exception as e:
                observation = f"Error executing tool '{decision.tool}': {str(e)}"

            step = AgentStep(action=decision, observation=observation)
            self.trajectory.append(step)

        return {
            "output": "Max iterations reached without explicit Final Answer.",
            "iterations": self.max_iterations,
            "trajectory": self.trajectory,
        }
