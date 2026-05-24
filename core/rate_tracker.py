"""
core/rate_tracker.py

Gemini daily request tracker.
Counts API requests per provider per day, persisted in the PAI SQLite database.
"""

import config
from core.logger import logger


def record_request(provider: str) -> None:
    """Increment today's request count and emit warnings if needed."""
    if not config.MEMORY_ENABLED:
        return
    try:
        from memory.store import increment_daily_rate_count
        new_count = increment_daily_rate_count(provider)
        _log_if_needed(provider, new_count)
    except Exception as e:
        logger.debug(f"rate_tracker: could not record request for {provider}: {e}")


def get_count(provider: str) -> int:
    """Return today's request count for the given provider."""
    if not config.MEMORY_ENABLED:
        return 0
    try:
        from memory.store import get_daily_rate_count
        return get_daily_rate_count(provider)
    except Exception as e:
        logger.debug(f"rate_tracker: could not read count for {provider}: {e}")
        return 0


def is_near_limit(provider: str) -> bool:
    """Return True if today's count >= config.GEMINI_WARN_AT (gemini only)."""
    if provider != "gemini":
        return False
    return get_count(provider) >= config.GEMINI_WARN_AT


def is_at_limit(provider: str) -> bool:
    """Return True if today's count >= config.GEMINI_DAILY_LIMIT (gemini only)."""
    if provider != "gemini":
        return False
    return get_count(provider) >= config.GEMINI_DAILY_LIMIT


def _log_if_needed(provider: str, count: int) -> None:
    """Emit log warnings at the warn threshold and at the hard limit."""
    if provider != "gemini":
        return

    limit = config.GEMINI_DAILY_LIMIT
    warn_at = config.GEMINI_WARN_AT

    if count >= limit:
        logger.warning(
            f"Gemini daily limit reached: {count}/{limit} requests today. "
            f"Switch provider or wait until tomorrow (automatic Ollama fallback "
            f"is active if Ollama is running)."
        )
    elif count >= warn_at:
        logger.warning(
            f"Gemini usage: {count}/{limit} requests today "
            f"({limit - count} remaining). Approaching daily limit."
        )
    else:
        logger.debug(f"Gemini usage: {count}/{limit} requests today")
