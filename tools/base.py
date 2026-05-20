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
    """Returned by every tool execution."""
    success: bool
    output: str = ""     # safe default — error field carries failure detail
    error: str = ""      # populated on failure


class BaseTool(ABC):
    """All tools implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier. Must match what the LLM receives."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Plain English description of what the tool does and when to use it.
        This is the primary prompt the LLM reads to decide whether to call
        this tool — write it to be unambiguous.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        JSON Schema object describing the tool's arguments.
        Must follow the OpenAI/Gemini function-calling schema format.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Run the tool with the given arguments.

        Args:
            **kwargs: Arguments as specified in self.parameters.

        Returns:
            ToolResult with success=True and output on success,
            or success=False and error on failure. Never raises.
        """
        ...

    def to_function_spec(self) -> dict:
        """Convert this tool to the JSON format sent to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
