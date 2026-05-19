"""
PAI Gemini Provider

Implements the LLM provider interface for Google's Gemini API using the
google-generativeai SDK. Uses the gemini-2.5-flash-lite model with
single-shot generation via generate_content().
"""

import google.generativeai as genai

import config
from core.exceptions import ProviderError, RateLimitError
from core.logger import logger
from providers.base import LLMProvider, LLMResponse, Message


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider.

    Configures the google-generativeai SDK with the API key from config
    and uses GenerativeModel for text generation.
    """

    MODEL_NAME = "gemini-2.5-flash-lite"

    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.MODEL_NAME)
        logger.info(f"GeminiProvider initialised with model: {self.MODEL_NAME}")

    @property
    def model_name(self) -> str:
        """Return the name of the configured Gemini model."""
        return self.MODEL_NAME

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate a response from Gemini given a list of messages.

        Converts the internal Message format to Gemini's contents format,
        mapping "assistant" role to "model" and all other roles to "user".

        Args:
            messages: The conversation history as a list of Message objects.

        Returns:
            An LLMResponse containing the generated text and model metadata.

        Raises:
            ProviderError: If the Gemini API call fails for any reason.
        """
        try:
            contents = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [msg.content]})

            result = self.model.generate_content(contents)
            response_text = result.text

            logger.debug(f"Gemini responded: {response_text[:100]}...")
            return LLMResponse(content=response_text, metadata={"model": self.MODEL_NAME})
        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini API error: {error_str}")
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit hit: {error_str}")
            raise ProviderError(f"Gemini failed: {error_str}")

    def health_check(self) -> bool:
        """Verify Gemini API key is valid by listing available models."""
        try:
            models = list(genai.list_models())
            if not models:
                raise ProviderError("Gemini API returned no models — check your API key.")
            logger.debug(f"Gemini health check passed ({len(models)} models available)")
            return True
        except Exception as e:
            raise ProviderError(f"Gemini health check failed: {e}")
