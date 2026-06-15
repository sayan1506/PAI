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
from providers.base import LLMProvider, LLMResponse, Message, ToolCall


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider.

    Communicates with a running Ollama instance via its REST API.
    Reads host and model configuration from the config module.

    Attributes:
        base_url: Base URL of the Ollama server (from ``config.OLLAMA_HOST``).
        model: Name of the Ollama model to use (from ``config.OLLAMA_MODEL``).
    """

    def __init__(self):
        """Read host and model from config and store them on the instance.

        Side effects:
            Emits an info log recording the configured endpoint and model.

        Attributes set:
            base_url: The Ollama server URL.
            model: The Ollama model name.
        """
        self.base_url = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
        logger.info(f"OllamaProvider initialised: {self.base_url} | model: {self.model}")

    @property
    def model_name(self) -> str:
        """Return the name of the configured Ollama model."""
        return self.model

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        """Convert a Message list to Ollama's chat format.

        Handles three message types:
        - user/assistant text messages
        - assistant messages with tool_calls → includes tool_calls array
        - tool result messages → role "tool" with content

        Args:
            messages: The conversation history as internal Message objects.

        Returns:
            A list of Ollama chat message dicts.
        """
        ollama_messages = []
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                # Assistant requested tool calls
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                ollama_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls,
                })
            elif msg.role == "tool":
                # Tool result
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                })
            else:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        return ollama_messages

    def _build_ollama_tools(self, tool_specs: list[dict]) -> list[dict]:
        """Convert tool specs to Ollama's tool format (OpenAI-compatible).

        Args:
            tool_specs: PAI tool specs, each with ``name``, ``description``,
                and ``parameters`` keys.

        Returns:
            A list of tool dicts wrapped in the OpenAI function envelope.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for spec in tool_specs
        ]

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

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Generate a response with tool-calling support via Ollama.

        Uses Ollama's /api/chat endpoint with the tools parameter.
        Ollama supports OpenAI-compatible tool calling for models that
        support it (llama3.1, gemma4, etc.).

        Args:
            messages: Full conversation history including any tool results.
            tools: List of tool specs in JSON-Schema function format.

        Returns:
            LLMResponse with tool_calls populated if the LLM chose a tool.
        """
        try:
            ollama_messages = self._build_messages(messages)
            ollama_tools = self._build_ollama_tools(tools) if tools else None

            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
            }
            if ollama_tools:
                payload["tools"] = ollama_tools

            logger.debug(f"Sending to Ollama with tools ({self.model})")
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            message = data.get("message", {})
            content = message.get("content", "")

            # Parse tool calls from response
            tool_calls = []
            raw_tool_calls = message.get("tool_calls", [])
            for i, tc in enumerate(raw_tool_calls):
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=func.get("name", f"call_{i}"),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                metadata={"model": self.model},
            )
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            raise ProviderError(f"Ollama tool call error: {e}")

    def health_check(self) -> bool:
        """Verify Ollama is reachable by hitting the /api/tags endpoint.

        Returns:
            True if the endpoint responds successfully.

        Raises:
            ProviderError: If Ollama is unreachable or the request fails.
        """
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
