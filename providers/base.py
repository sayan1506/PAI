"""
PAI Provider Base Module

Defines the abstract interface for LLM providers and the data structures
used for communication between the Agent and Provider layers.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ToolCall:
    """Represents a single tool invocation requested by the LLM.

    Attributes:
        id: Provider-assigned ID (used to match results).
        name: Tool name (must match BaseTool.name).
        arguments: Parsed argument dict.
    """

    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", "tool").
        content: The text content of the message.
        tool_calls: List of tool calls (populated when role is "assistant" and LLM requests tools).
        tool_call_id: ID matching a tool call (populated when role is "tool").
    """

    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""


@dataclass
class LLMResponse:
    """Represents a response from an LLM provider.

    Attributes:
        content: The generated text response.
        tool_calls: List of tool calls requested by the LLM (empty if text-only response).
        metadata: Provider-specific metadata (model, tokens, etc).
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the generate method and model_name property.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the configured model."""
        ...

    @abstractmethod
    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate a response from the LLM given a list of messages.

        Args:
            messages: The conversation history as a list of Message objects.

        Returns:
            An LLMResponse containing the generated text and optional metadata.

        Raises:
            ProviderError: If the LLM API call fails.
        """
        ...

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Generate a response with tool-calling support.

        Default implementation ignores tools and falls back to generate().
        Override in providers that support native function calling.

        Args:
            messages: Full conversation history including any tool results.
            tools: List of tool specs in JSON-Schema function format.

        Returns:
            LLMResponse. If the LLM chose a tool, response.tool_calls is
            populated and response.content may be empty.
        """
        return self.generate(messages)

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the provider is reachable and configured correctly.

        Returns:
            True if the provider is healthy and ready to use.

        Raises:
            ProviderError: If the provider is unreachable or misconfigured.
        """
        ...
