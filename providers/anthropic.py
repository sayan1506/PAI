"""
providers/anthropic.py

AnthropicProvider — uses the Anthropic Python SDK to call Claude models.

Anthropic's API differs from OpenAI's in three key ways:
  1. System prompt is passed as a separate `system=` kwarg, NOT as a message.
  2. Tool calls are represented as typed blocks inside the content list
     (type="tool_use") instead of a separate tool_calls field.
  3. Tool results are user-role messages with type="tool_result" blocks,
     and consecutive results must be batched into a single user turn.

Message history layout (set by Agent):
  history[0]: role="user",      content=<system prompt>   ← extracted as system=
  history[1]: role="assistant", content="Understood..."   ← skipped (priming)
  history[2:]: normal conversation turns

API docs: https://docs.anthropic.com/en/api/messages
"""

import anthropic as _anthropic

import config
from core.exceptions import ProviderError, RateLimitError
from core.logger import logger
from providers.base import LLMProvider, LLMResponse, Message, ToolCall


class AnthropicProvider(LLMProvider):
    """
    LLM provider for Anthropic Claude.

    Requires config.ANTHROPIC_API_KEY. Uses config.ANTHROPIC_MODEL
    (default: "claude-sonnet-4-5"). Override with ANTHROPIC_MODEL env var.
    """

    MAX_TOKENS = 4096

    def __init__(self):
        self._client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        logger.info(
            f"AnthropicProvider initialised (model: {config.ANTHROPIC_MODEL})"
        )

    @property
    def model_name(self) -> str:
        """Return the configured Anthropic model name."""
        return config.ANTHROPIC_MODEL

    # ── Message conversion ─────────────────────────────────────────────────

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict]]:
        """
        Split the agent's history into a system string and an Anthropic
        messages list.

        Strategy:
          - history[0] (role="user") holds the system prompt → `system` param.
          - history[1] (role="assistant") is the priming response → skipped;
            including it would place an assistant turn before any user turn,
            which Anthropic rejects.
          - history[2:] are converted to Anthropic message dicts.

        Anthropic message format:
          user turn (plain):     {"role": "user",      "content": "text"}
          assistant turn (plain):{"role": "assistant", "content": "text"}
          assistant with tools:  {"role": "assistant", "content": [
                                    {"type": "text",     "text": "..."},
                                    {"type": "tool_use", "id": ...,
                                     "name": ..., "input": {...}}
                                 ]}
          tool results (user):   {"role": "user", "content": [
                                    {"type": "tool_result",
                                     "tool_use_id": ..., "content": "..."}
                                 ]}
          Consecutive tool-result messages are batched into ONE user turn.

        Args:
            messages: Agent's full history list.

        Returns:
            Tuple of (system_prompt_string, anthropic_messages_list).
        """
        system = messages[0].content if messages else ""
        remaining = messages[2:] if len(messages) > 2 else []

        result = []
        i = 0
        while i < len(remaining):
            msg = remaining[i]

            if msg.role == "tool":
                # Batch ALL consecutive tool results into ONE user turn
                tool_results = []
                while i < len(remaining) and remaining[i].role == "tool":
                    m = remaining[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})

            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content})
                i += 1

            elif msg.role == "assistant":
                result.append({"role": "assistant", "content": msg.content})
                i += 1

            else:
                # user, system, or any unknown role → send as user
                result.append({"role": "user", "content": msg.content})
                i += 1

        return system, result

    def _build_anthropic_tools(self, tool_specs: list[dict]) -> list[dict]:
        """
        Convert PAI tool specs (JSON Schema function format) to Anthropic's
        tool format.

        PAI format:    {"name": ..., "description": ..., "parameters": {...}}
        Anthropic fmt: {"name": ..., "description": ..., "input_schema": {...}}

        The only difference is the key name: "parameters" → "input_schema".
        """
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
            for spec in tool_specs
        ]

    def _parse_response(self, response) -> LLMResponse:
        """
        Convert an Anthropic Messages response to an LLMResponse.

        Iterates over response.content (a list of typed blocks):
          type="text"     → accumulated into the text response
          type="tool_use" → converted to a ToolCall

        Args:
            response: anthropic.types.Message object.

        Returns:
            LLMResponse with content and optional tool_calls.
        """
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,  # already a dict, not JSON string
                ))

        return LLMResponse(
            content=" ".join(text_parts),
            tool_calls=tool_calls,
            metadata={"model": self.model_name},
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, messages: list[Message]) -> LLMResponse:
        """
        Generate a plain text response from Claude (no tool calling).

        Args:
            messages: Internal conversation history.

        Returns:
            LLMResponse with content and model metadata.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError:  On any other API failure.
        """
        system, anthropic_msgs = self._extract_system_and_messages(messages)
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.MAX_TOKENS,
                system=system,
                messages=anthropic_msgs,
            )
            result = self._parse_response(response)
            logger.debug(f"AnthropicProvider responded: {result.content[:100]}")
            return result
        except _anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit: {e}")
        except _anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}")

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """
        Generate a response with Anthropic tool-use support.

        Args:
            messages: Full conversation history including tool results.
            tools:    PAI tool specs (JSON-Schema function format).

        Returns:
            LLMResponse; tool_calls populated if Claude chose a tool.

        Raises:
            RateLimitError / ProviderError (same as generate()).
        """
        system, anthropic_msgs = self._extract_system_and_messages(messages)
        anthropic_tools = self._build_anthropic_tools(tools) if tools else []
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.MAX_TOKENS,
                "system": system,
                "messages": anthropic_msgs,
            }
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = self._client.messages.create(**kwargs)
            result = self._parse_response(response)
            logger.debug(
                f"AnthropicProvider tool response: "
                f"{len(result.tool_calls)} tool call(s)"
            )
            return result
        except _anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit: {e}")
        except _anthropic.APIError as e:
            raise ProviderError(f"Anthropic tool call failed: {e}")

    def health_check(self) -> bool:
        """
        Verify the Anthropic API key is valid by sending a minimal request.

        Returns:
            True on success.

        Raises:
            ProviderError if the check fails.
        """
        try:
            self._client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            logger.debug("AnthropicProvider health check passed")
            return True
        except _anthropic.RateLimitError:
            logger.debug("AnthropicProvider health check: rate limited but reachable")
            return True
        except Exception as e:
            raise ProviderError(f"Anthropic health check failed: {e}")
