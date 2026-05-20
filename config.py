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

# ── Voice ─────────────────────────────────────────────────────
WAKE_WORD: str = os.getenv("WAKE_WORD", "hey jarvis")
WAKE_WORD_SENSITIVITY: float = float(os.getenv("WAKE_WORD_SENSITIVITY", "0.5"))
STT_MODEL: str = os.getenv("STT_MODEL", "faster-whisper")
STT_SIZE: str = os.getenv("STT_SIZE", "base")
VOICE_ENABLED: bool = os.getenv("VOICE_ENABLED", "True").lower() in ("true", "1", "yes")
VAD_SILENCE_MS: int = int(os.getenv("VAD_SILENCE_MS", "1200"))
MIC_SAMPLE_RATE: int = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
MIC_CHANNELS: int = int(os.getenv("MIC_CHANNELS", "1"))
MIC_DEVICE: str = os.getenv("MIC_DEVICE", "")  # device index or empty for default

# ── Tools ─────────────────────────────────────────────────────
ENABLE_FILE_OPS: bool = os.getenv("ENABLE_FILE_OPS", "True").lower() in ("true", "1", "yes")
ENABLE_APP_LAUNCHER: bool = os.getenv("ENABLE_APP_LAUNCHER", "True").lower() in ("true", "1", "yes")
ENABLE_TERMINAL: bool = os.getenv("ENABLE_TERMINAL", "False").lower() in ("true", "1", "yes")
ENABLE_BROWSER: bool = os.getenv("ENABLE_BROWSER", "True").lower() in ("true", "1", "yes")
ENABLE_SCREEN_READER: bool = os.getenv("ENABLE_SCREEN_READER", "True").lower() in ("true", "1", "yes")
MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

# ── Platform ──────────────────────────────────────────────────
SCREEN_SCALE_FACTOR: float = float(os.getenv("SCREEN_SCALE_FACTOR", "1.0"))

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

    if ENABLE_TERMINAL:
        from core.logger import logger
        logger.warning(
            "ENABLE_TERMINAL is ON. Shell commands will be executed. "
            "Ensure you trust all inputs reaching the agent."
        )
