"""
PAI Ollama Provider

Implements the LLM provider interface for local Ollama models using the
REST API. POSTs to the /api/chat endpoint with stream disabled for
complete responses.
"""

import requests

import config
from core.exceptions import ProviderError
from core.logger import logger
from providers.base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider.

    Communicates with a running Ollama instance via its REST API.
    Reads host and model configuration from the config module.
    """

    def __init__(self):
        self.base_url = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
        logger.info(f"OllamaProvider initialised: {self.base_url} | model: {self.model}")

    @property
    def model_name(self) -> str:
        """Return the name of the configured Ollama model."""
        return self.model

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate a response from Ollama given a list of messages.

        Formats messages as a JSON payload and POSTs to the Ollama
        /api/chat endpoint with stream=False for a complete response.

        Args:
            messages: The conversation history as a list of Message objects.

        Returns:
            An LLMResponse containing the generated text and model metadata.

        Raises:
            ProviderError: If Ollama is unreachable or the API call fails.
        """
        try:
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
            }
            logger.debug(f"Sending to Ollama ({self.model}): {messages[-1].content[:100]}...")
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["message"]["content"]
            logger.debug(f"Ollama responded: {response_text[:100]}...")
            return LLMResponse(content=response_text, metadata={"model": self.model})
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}")

    def health_check(self) -> bool:
        """Verify Ollama is reachable by hitting the /api/tags endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.debug("Ollama health check passed")
            return True
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            raise ProviderError(f"Ollama health check failed: {e}")
