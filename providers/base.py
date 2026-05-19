"""
PAI Provider Base Module

Defines the abstract interface for LLM providers and the data structures
used for communication between the Agent and Provider layers.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", or "system").
        content: The text content of the message.
    """

    role: str
    content: str


@dataclass
class LLMResponse:
    """Represents a response from an LLM provider.

    Attributes:
        content: The generated text response.
        metadata: Provider-specific metadata (model, tokens, etc).
    """

    content: str
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

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the provider is reachable and configured correctly.

        Returns:
            True if the provider is healthy and ready to use.

        Raises:
            ProviderError: If the provider is unreachable or misconfigured.
        """
        ...
