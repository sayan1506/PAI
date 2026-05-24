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

GITHUB_TOKEN:      str = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODEL:      str = os.getenv("GITHUB_MODEL", "gpt-4o")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL:   str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

OPENAI_API_KEY:    str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL:      str = os.getenv("OPENAI_MODEL", "gpt-4o")

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

# ── Vision / Screen reader ────────────────────────────────────────────
VISION_ENABLED: bool = os.getenv("VISION_ENABLED", "False").lower() in ("true", "1", "yes")
VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "gemini")
VISION_MODEL: str = os.getenv("VISION_MODEL", "gemma4:e4b")
SCREENSHOT_DELAY: float = float(os.getenv("SCREENSHOT_DELAY", "0.3"))

# ── Browser control ───────────────────────────────────────────────────
BROWSER_ENABLED: bool = os.getenv("BROWSER_ENABLED", "False").lower() in ("true", "1", "yes")
BROWSER_APP: str = os.getenv("BROWSER_APP", "chrome")
BROWSER_REMOTE_PORT: int = int(os.getenv("BROWSER_REMOTE_PORT", "9222"))
TYPING_SPEED: float = float(os.getenv("TYPING_SPEED", "0.04"))
BROWSER_BINARY: str = os.getenv("BROWSER_BINARY", "")  # override Chrome binary path (auto-detect if empty)
BROWSER_CDP_TIMEOUT: int = int(os.getenv("BROWSER_CDP_TIMEOUT", "10"))  # seconds to wait for debugging port
BROWSER_TYPE_DELAY: float = float(os.getenv("BROWSER_TYPE_DELAY", "0.04"))  # seconds between keystrokes

# ── Memory ────────────────────────────────────────────────────────────
MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "True").lower() in ("true", "1", "yes")
MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "~/.pai/memory.db")
MEMORY_INJECT_TOP_K: int = int(os.getenv("MEMORY_INJECT_TOP_K", "5"))

# ── TTS / Barge-in ───────────────────────────────────────────────────
TTS_ENABLED: bool = os.getenv("TTS_ENABLED", "True").lower() in ("true", "1", "yes")
TTS_ENGINE: str = os.getenv("TTS_ENGINE", "kokoro")
TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
TTS_VOICE: str = os.getenv("TTS_VOICE", "af_heart")
BARGE_IN_ENABLED: bool = os.getenv("BARGE_IN_ENABLED", "True").lower() in ("true", "1", "yes")

# ── UI ────────────────────────────────────────────────────────────────
OVERLAY_ENABLED: bool = os.getenv("OVERLAY_ENABLED", "True").lower() in ("true", "1", "yes")
NOTIFICATIONS_ENABLED: bool = os.getenv("NOTIFICATIONS_ENABLED", "True").lower() in ("true", "1", "yes")

# ── Weather ───────────────────────────────────────────────────────────
WEATHER_ENABLED:          bool  = os.getenv("WEATHER_ENABLED", "False").lower() in ("true", "1", "yes")
WEATHER_API_KEY:          str   = os.getenv("WEATHER_API_KEY", "")
WEATHER_UNITS:            str   = os.getenv("WEATHER_UNITS", "metric")   # metric | imperial
WEATHER_DEFAULT_LOCATION: str   = os.getenv("WEATHER_DEFAULT_LOCATION", "")

# ── Reminders ─────────────────────────────────────────────────────────
REMINDERS_ENABLED:        bool  = os.getenv("REMINDERS_ENABLED", "True").lower() in ("true", "1", "yes")
REMINDER_POLL_INTERVAL:   int   = int(os.getenv("REMINDER_POLL_INTERVAL", "10"))  # seconds between DB polls

# ── Phase 10 — Hardening & Polish ─────────────────────────────────────────────
SKIP_HEALTH_CHECK:   bool  = os.getenv("SKIP_HEALTH_CHECK",   "false").lower() in ("true", "1", "yes")
HOTKEY_ENABLED:      bool  = os.getenv("HOTKEY_ENABLED",      "true").lower()  in ("true", "1", "yes")
HOTKEY_COMBO:        str   = os.getenv("HOTKEY_COMBO",        "<ctrl>+<shift>+j")
GEMINI_DAILY_LIMIT:  int   = int(os.getenv("GEMINI_DAILY_LIMIT",  "1000"))
GEMINI_WARN_AT:      int   = int(os.getenv("GEMINI_WARN_AT",      "800"))
CONFIRM_DESTRUCTIVE: bool  = os.getenv("CONFIRM_DESTRUCTIVE", "true").lower()  in ("true", "1", "yes")

# Valid provider names for validation
SUPPORTED_PROVIDERS: set = {"gemini", "ollama", "github", "anthropic", "openai"}


def validate_config() -> None:
    """
    Validate the current configuration.

    Raises ConfigError if:
    - LLM_PROVIDER is not in SUPPORTED_PROVIDERS
    - The required API key for the selected provider is missing

    Provider key requirements:
      gemini    → GEMINI_API_KEY
      github    → GITHUB_TOKEN
      anthropic → ANTHROPIC_API_KEY
      openai    → OPENAI_API_KEY
      ollama    → no key required (local)
    """
    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unknown LLM provider: '{LLM_PROVIDER}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    _KEY_REQUIREMENTS = {
        "gemini":    ("GEMINI_API_KEY",    GEMINI_API_KEY),
        "github":    ("GITHUB_TOKEN",      GITHUB_TOKEN),
        "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        "openai":    ("OPENAI_API_KEY",    OPENAI_API_KEY),
    }

    if LLM_PROVIDER in _KEY_REQUIREMENTS:
        key_name, key_value = _KEY_REQUIREMENTS[LLM_PROVIDER]
        if not key_value:
            raise ConfigError(
                f"{key_name} is required when LLM_PROVIDER is '{LLM_PROVIDER}'. "
                f"Set it in your .env file."
            )

    if ENABLE_TERMINAL:
        from core.logger import logger
        logger.warning(
            "ENABLE_TERMINAL is ON. Shell commands will be executed. "
            "Ensure you trust all inputs reaching the agent."
        )
