"""
tools/base.py

Abstract base class for all PAI tools.
Each tool is self-contained: it declares its own name, description,
and parameter schema so the LLM knows exactly when and how to call it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Outcome of a single tool execution.

    Every tool returns one of these instead of raising, so the caller can
    uniformly inspect ``success`` and read either ``output`` or ``error``.

    Attributes:
        success: True if the operation completed successfully.
        output: Human-readable result text. Empty when the operation failed;
            the failure detail is carried in ``error`` instead.
        error: Failure description. Empty on success.
    """
    success: bool
    output: str = ""     # safe default — error field carries failure detail
    error: str = ""      # populated on failure


class BaseTool(ABC):
    """Abstract interface implemented by every PAI tool.

    A tool is self-contained: it declares its own ``name``, ``description``,
    and parameter schema so the LLM knows when and how to call it, and it
    exposes a single ``execute`` entry point. Concrete subclasses must
    implement all four abstract members.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique snake_case identifier.

        Returns:
            The tool name as received by the LLM. Must match exactly so
            dispatch can route calls to this tool.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a plain-English description of the tool.

        This is the primary prompt the LLM reads to decide whether to call
        the tool, so it should be unambiguous about what the tool does and
        when to use it.

        Returns:
            The description string presented to the LLM.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Return the JSON Schema describing the tool's arguments.

        The schema must follow the OpenAI/Gemini function-calling format
        (a JSON Schema ``object`` with ``properties`` and ``required``).

        Returns:
            A JSON Schema object describing the accepted arguments.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Run the tool with the given arguments.

        Implementations should catch their own exceptions and report
        failures through the returned ``ToolResult`` rather than raising.

        Args:
            **kwargs: Arguments as specified in ``self.parameters``.

        Returns:
            A ``ToolResult`` with ``success=True`` and ``output`` on success,
            or ``success=False`` and ``error`` on failure.
        """
        ...

    def to_function_spec(self) -> dict:
        """Convert this tool to the function-calling spec sent to the LLM.

        Returns:
            A dict with ``name``, ``description``, and ``parameters`` keys,
            ready to be included in an LLM function/tool definition.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
