"""
PAI Configuration System

Loads environment variables from .env file and exposes them as module-level
variables with sensible defaults. Provides validate_config() to check that
required values are present based on the selected provider.
"""

import os
from dotenv import load_dotenv
from core.exceptions import ConfigError

# Load .env file into environment
load_dotenv()

# --- Module-level configuration variables ---

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
AGENT_NAME: str = os.getenv("AGENT_NAME", "PAI")
MAX_SESSION_TURNS: int = int(os.getenv("MAX_SESSION_TURNS", "20"))

# Valid provider names for validation
SUPPORTED_PROVIDERS: set = {"gemini", "ollama"}


def validate_config() -> None:
    """
    Validate the current configuration.

    Raises ConfigError if:
    - LLM_PROVIDER is not a supported provider name
    - LLM_PROVIDER is "gemini" and GEMINI_API_KEY is not set
    """
    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unknown LLM provider: '{LLM_PROVIDER}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ConfigError(
            "GEMINI_API_KEY is required when LLM_PROVIDER is set to 'gemini'. "
            "Set it in your .env file or environment variables."
        )
