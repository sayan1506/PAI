"""
providers/openai_provider.py

OpenAIProvider — standard OpenAI API (api.openai.com).

Uses config.OPENAI_API_KEY and config.OPENAI_MODEL (default: "gpt-4o").
Supports tool calling via the OpenAI function-calling API.
"""

from openai import OpenAI

import config
from core.logger import logger
from providers.openai_compat import _OpenAICompatBase


class OpenAIProvider(_OpenAICompatBase):
    """
    LLM provider for the OpenAI API.

    Requires config.OPENAI_API_KEY. Uses config.OPENAI_MODEL (default: "gpt-4o").
    Override with OPENAI_MODEL env var (e.g. "gpt-4o-mini").
    """

    def __init__(self):
        logger.info(
            f"OpenAIProvider initialised (model: {config.OPENAI_MODEL})"
        )

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model name."""
        return config.OPENAI_MODEL

    def _make_client(self) -> OpenAI:
        """
        Build a standard OpenAI client.

        Called on every request so OPENAI_API_KEY changes in tests are
        always picked up.
        """
        return OpenAI(api_key=config.OPENAI_API_KEY)
