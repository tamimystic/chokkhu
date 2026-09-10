"""Autonomous AI Agents, Tool-Use & Multi-Agent Collaboration Framework.

Pure Python / NumPy implementations of:
- Tool, ToolParameter, ToolRegistry
- ReActAgent, AgentAction, AgentFinish, AgentStep
- ChainOfThought, TreeOfThoughts, GraphOfThoughts
- MultiAgentCoordinator, ReflexionEngine, SelfConsistency
"""

from .tools import (
    Tool,
    ToolParameter,
    ToolRegistry,
)
from .react import (
    ReActAgent,
    AgentAction,
    AgentFinish,
    AgentStep,
)
from .reasoning import (
    ChainOfThought,
    TreeOfThoughts,
    GraphOfThoughts,
)
from .coordinator import (
    MultiAgentCoordinator,
    ReflexionEngine,
    SelfConsistency,
)

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ReActAgent",
    "AgentAction",
    "AgentFinish",
    "AgentStep",
    "ChainOfThought",
    "TreeOfThoughts",
    "GraphOfThoughts",
    "MultiAgentCoordinator",
    "ReflexionEngine",
    "SelfConsistency",
]
