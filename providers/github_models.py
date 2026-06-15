"""
providers/github_models.py

GitHubModelsProvider — uses GitHub Models' OpenAI-compatible inference
endpoint with a GitHub Personal Access Token.

Endpoint: https://models.inference.ai.azure.com
Auth:     Bearer <GITHUB_TOKEN>
Models:   gpt-4o (default), gpt-4o-mini, Llama-3.3-70B-Instruct, Phi-4, etc.

Free tier: available to all GitHub accounts with a PAT that has the
           "models" scope. Rate limits vary by model tier.

API docs: https://docs.github.com/en/github-models
"""

from openai import OpenAI

import config
from core.logger import logger
from providers.openai_compat import _OpenAICompatBase

_GITHUB_BASE_URL = "https://models.inference.ai.azure.com"


class GitHubModelsProvider(_OpenAICompatBase):
    """
    LLM provider for GitHub Models (OpenAI-compatible endpoint).

    Requires config.GITHUB_TOKEN to be set. The PAT must have the
    "models" (read) scope enabled at github.com/settings/tokens.

    Uses config.GITHUB_MODEL as the model name (default: "gpt-4o").
    Override with GITHUB_MODEL env var (e.g. "gpt-4o-mini").
    """

    def __init__(self):
        """Log initialisation of the GitHub Models provider.

        Configuration is read lazily inside ``_make_client`` on each request,
        so the constructor only records the configured model and endpoint.

        Side effects:
            Emits an info log with the configured model and endpoint URL.
        """
        logger.info(
            f"GitHubModelsProvider initialised "
            f"(model: {config.GITHUB_MODEL}, endpoint: {_GITHUB_BASE_URL})"
        )

    @property
    def model_name(self) -> str:
        """Return the configured GitHub Models model name."""
        return config.GITHUB_MODEL

    def _make_client(self) -> OpenAI:
        """
        Build an OpenAI client pointed at the GitHub Models inference endpoint.

        Called on every request so that GITHUB_TOKEN changes in config (e.g.
        during tests using monkeypatch) are always picked up.

        Returns:
            An ``OpenAI`` client whose ``base_url`` targets GitHub Models and
            whose key is ``config.GITHUB_TOKEN``.
        """
        return OpenAI(
            base_url=_GITHUB_BASE_URL,
            api_key=config.GITHUB_TOKEN,
        )
