"""
core/rate_tracker.py

Gemini daily request tracker.
Counts API requests per provider per day, persisted in the PAI SQLite database.
"""

import config
from core.logger import logger


def record_request(provider: str) -> None:
    """Increment today's request count for a provider and warn if needed.

    No-op when ``config.MEMORY_ENABLED`` is False. Failures to persist the
    count are swallowed and logged at DEBUG level so tracking never blocks a
    request.

    Args:
        provider: Provider name whose daily counter to increment (e.g. "gemini").

    Side effects:
        Writes to the SQLite rate-count table and may emit warning logs via
        :func:`_log_if_needed`.
    """
    if not config.MEMORY_ENABLED:
        return
    try:
        from memory.store import increment_daily_rate_count
        new_count = increment_daily_rate_count(provider)
        _log_if_needed(provider, new_count)
    except Exception as e:
        logger.debug(f"rate_tracker: could not record request for {provider}: {e}")


def get_count(provider: str) -> int:
    """Return today's recorded request count for a provider.

    No-op (returns 0) when ``config.MEMORY_ENABLED`` is False or on read error.

    Args:
        provider: Provider name to look up.

    Returns:
        The number of requests recorded for the provider today, or 0.
    """
    if not config.MEMORY_ENABLED:
        return 0
    try:
        from memory.store import get_daily_rate_count
        return get_daily_rate_count(provider)
    except Exception as e:
        logger.debug(f"rate_tracker: could not read count for {provider}: {e}")
        return 0


def is_near_limit(provider: str) -> bool:
    """Return True if the provider is approaching its daily request limit.

    Only meaningful for "gemini"; all other providers always return False.

    Args:
        provider: Provider name to check.

    Returns:
        True if ``provider`` is "gemini" and today's count has reached
        ``config.GEMINI_WARN_AT``.
    """
    if provider != "gemini":
        return False
    return get_count(provider) >= config.GEMINI_WARN_AT


def is_at_limit(provider: str) -> bool:
    """Return True if the provider has hit its hard daily request limit.

    Only meaningful for "gemini"; all other providers always return False.

    Args:
        provider: Provider name to check.

    Returns:
        True if ``provider`` is "gemini" and today's count has reached
        ``config.GEMINI_DAILY_LIMIT``.
    """
    if provider != "gemini":
        return False
    return get_count(provider) >= config.GEMINI_DAILY_LIMIT


def _log_if_needed(provider: str, count: int) -> None:
    """Emit usage log lines at the warning threshold and hard limit.

    Only acts for the "gemini" provider. Logs a WARNING once the count reaches
    the warn threshold or the daily limit, and a DEBUG line otherwise.

    Args:
        provider: Provider name the count belongs to.
        count: Today's request count after the latest increment.

    Side effects:
        Writes log records; does not modify any state.
    """
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
