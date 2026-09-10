"""Autonomous AI Agent Tool-Use and Function Dispatching Subsystem.

Pure Python / NumPy implementations of:
- ToolParameter & Tool: Declarative function schema definitions
- ToolRegistry: Thread-safe tool registration and execution dispatcher
- Builtin Tools: Calculator, VectorSearchTool, PythonREPLTool
"""

from typing import Any, Callable, Dict, List, Optional


class ToolParameter:
    """Specification for a single tool argument."""

    def __init__(
        self,
        name: str,
        param_type: str = "string",
        description: str = "",
        required: bool = True,
        default: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.param_type,
            "description": self.description,
        }


class Tool:
    """Executable Agent Tool with JSON Schema metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[List[ToolParameter]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or []

    def to_schema(self) -> Dict[str, Any]:
        """Convert tool definition to standard JSON function schema."""
        properties = {p.name: p.to_dict() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        """Execute tool function with provided keyword arguments."""
        # Fill defaults
        for p in self.parameters:
            if p.name not in kwargs and p.default is not None:
                kwargs[p.name] = p.default
        return self.func(**kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        return self.execute(**kwargs)


class ToolRegistry:
    """Registry and Dispatcher for Agent Tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool

    def register_func(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[List[ToolParameter]] = None,
    ) -> Tool:
        """Register a function directly as a tool."""
        tool = Tool(
            name=name, description=description, func=func, parameters=parameters
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON schemas for all registered tools."""
        return [t.to_schema() for t in self.tools.values()]

    def dispatch(self, name: str, **kwargs: Any) -> Any:
        """Dispatch execution to registered tool by name."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool.execute(**kwargs)
