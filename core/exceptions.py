"""
PAI Custom Exception Hierarchy

Provides structured exceptions for categorized error handling across all layers
of the PAI application. All custom exceptions inherit from PAIError, enabling
catch-all handling at the CLI layer while allowing specific handling at lower layers.
"""


class PAIError(Exception):
    """Base exception for all PAI errors."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(self.message)


class ConfigError(PAIError):
    """Raised when configuration is invalid or missing required values."""

    pass


class ProviderError(PAIError):
    """Raised when an LLM provider call fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when an LLM provider returns a 429 rate limit error."""

    pass


class AgentError(PAIError):
    """Raised when conversation processing fails."""

    pass


class ToolError(PAIError):
    """Raised when a tool execution fails."""

    pass


class AudioError(PAIError):
    """Raised when mic or audio output fails."""

    pass
