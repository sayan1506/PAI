"""
providers/openai_compat.py

_OpenAICompatBase — shared message-conversion and tool-call logic for
providers that use the OpenAI chat completions API format (OpenAI and
GitHub Models).

Not a public provider — do not import this directly from application code.
Import GitHubModelsProvider or OpenAIProvider instead.
"""

import json

from openai import RateLimitError as _OpenAIRateLimitError
from openai import APIError as _OpenAIAPIError

from core.exceptions import ProviderError, RateLimitError
from core.logger import logger
from providers.base import LLMProvider, LLMResponse, Message, ToolCall


class _OpenAICompatBase(LLMProvider):
    """
    Abstract base for OpenAI-compatible providers.

    Subclasses must implement:
      - model_name (property)
      - _make_client()  → openai.OpenAI instance

    Provides:
      - _build_messages()        Internal Message list → OpenAI messages list
      - _build_tools()           PAI tool specs → OpenAI tools list
      - _parse_tool_calls()      OpenAI response → list[ToolCall]
      - generate()
      - generate_with_tools()
      - health_check()
    """

    def _make_client(self):
        """Return a configured openai.OpenAI client. Must be overridden."""
        raise NotImplementedError

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convert internal Message list to OpenAI chat message format.

        Mapping:
          role="user" or "system"          → {"role": "user", "content": text}
          role="assistant", no tool_calls  → {"role": "assistant", "content": text}
          role="assistant", with tool_calls→ {"role": "assistant", "content": text,
                                              "tool_calls": [...]}
          role="tool"                      → {"role": "tool",
                                              "content": text,
                                              "tool_call_id": tool_call_id}
        """
        result = []
        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                tool_calls_payload = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls_payload,
                })
            elif msg.role == "assistant":
                result.append({"role": "assistant", "content": msg.content})
            else:
                # user, system, or unknown → send as "user"
                result.append({"role": "user", "content": msg.content})
        return result

    def _build_tools(self, tool_specs: list[dict]) -> list[dict]:
        """Wrap PAI tool specs in the OpenAI function-calling envelope."""
        return [{"type": "function", "function": spec} for spec in tool_specs]

    def _parse_tool_calls(self, response) -> list[ToolCall]:
        """
        Extract tool calls from an OpenAI ChatCompletion response.

        response.choices[0].message.tool_calls is a list of objects with:
          .id                    — string
          .function.name         — string
          .function.arguments    — JSON-encoded string; parsed to dict here
        """
        raw = response.choices[0].message.tool_calls
        if not raw:
            return []
        tool_calls = []
        for tc in raw:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
                logger.warning(
                    f"Could not parse tool arguments for '{tc.function.name}': "
                    f"{tc.function.arguments!r}"
                )
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))
        return tool_calls

    def generate(self, messages: list[Message]) -> LLMResponse:
        """
        Generate a plain text response (no tool calling).

        Args:
            messages: Internal conversation history.

        Returns:
            LLMResponse with content and model metadata.

        Raises:
            RateLimitError: On HTTP 429 from the API.
            ProviderError:  On any other API failure.
        """
        client = self._make_client()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(messages),
            )
            text = response.choices[0].message.content or ""
            logger.debug(f"{self.__class__.__name__} responded: {text[:100]}")
            return LLMResponse(
                content=text,
                metadata={"model": self.model_name},
            )
        except _OpenAIRateLimitError as e:
            raise RateLimitError(f"{self.__class__.__name__} rate limit: {e}")
        except _OpenAIAPIError as e:
            raise ProviderError(f"{self.__class__.__name__} API error: {e}")

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """
        Generate a response with tool-calling support.

        Args:
            messages: Full conversation history including tool results.
            tools:    PAI tool specs (JSON-Schema function format).

        Returns:
            LLMResponse; tool_calls populated if the model chose a tool.

        Raises:
            RateLimitError / ProviderError (same as generate()).
        """
        client = self._make_client()
        try:
            kwargs = {
                "model": self.model_name,
                "messages": self._build_messages(messages),
            }
            if tools:
                kwargs["tools"] = self._build_tools(tools)

            response = client.chat.completions.create(**kwargs)

            tool_calls = self._parse_tool_calls(response)
            text = response.choices[0].message.content or ""

            return LLMResponse(
                content=text,
                tool_calls=tool_calls,
                metadata={"model": self.model_name},
            )
        except _OpenAIRateLimitError as e:
            raise RateLimitError(f"{self.__class__.__name__} rate limit: {e}")
        except _OpenAIAPIError as e:
            raise ProviderError(f"{self.__class__.__name__} API error: {e}")

    def health_check(self) -> bool:
        """
        Verify the provider is reachable by sending a minimal request.

        Sends a one-token completion to confirm the API key is valid and the
        endpoint is reachable.

        Returns:
            True on success.

        Raises:
            ProviderError if the check fails.
        """
        client = self._make_client()
        try:
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logger.debug(f"{self.__class__.__name__} health check passed")
            return True
        except _OpenAIRateLimitError:
            # Rate limited but reachable → treat as healthy
            logger.debug(f"{self.__class__.__name__} health check: rate limited but reachable")
            return True
        except Exception as e:
            raise ProviderError(f"{self.__class__.__name__} health check failed: {e}")
