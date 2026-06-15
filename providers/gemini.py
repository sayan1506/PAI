"""
PAI Gemini Provider

Implements the LLM provider interface for Google's Gemini API using the
google-generativeai SDK. Uses the gemini-2.5-flash-lite model with
single-shot generation via generate_content().
"""

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
from google.protobuf.struct_pb2 import Struct

import config
from core.exceptions import ProviderError, RateLimitError
from core.logger import logger
from providers.base import LLMProvider, LLMResponse, Message, ToolCall


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider.

    Configures the google-generativeai SDK with the API key from config
    and uses ``GenerativeModel`` for single-shot text generation and native
    function calling.

    Attributes:
        MODEL_NAME: The Gemini model identifier this provider targets.
        model: The configured ``genai.GenerativeModel`` client.
    """

    MODEL_NAME = "gemini-2.5-flash-lite"

    def __init__(self):
        """Configure the SDK and build the Gemini model client.

        Reads ``config.GEMINI_API_KEY`` to authenticate the
        google-generativeai SDK and instantiates a ``GenerativeModel`` for
        the configured model.

        Side effects:
            Calls ``genai.configure`` with the API key and emits an info log.

        Attributes set:
            model: The configured ``genai.GenerativeModel`` instance.
        """
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.MODEL_NAME)
        logger.info(f"GeminiProvider initialised with model: {self.MODEL_NAME}")

    @property
    def model_name(self) -> str:
        """Return the name of the configured Gemini model."""
        return self.MODEL_NAME

    def _build_contents(self, messages: list[Message]) -> list[dict]:
        """Convert a Message list to Gemini's contents format.

        Handles four message types:
        - user/system messages         → role "user" with text part
        - plain assistant messages     → role "model" with text part
        - assistant with tool_calls    → role "model" with function_call parts
        - tool result messages         → role "user" with function_response parts
                                         (consecutive tool messages are batched into
                                         ONE user turn as Gemini requires)

        Args:
            messages: The conversation history as internal Message objects.

        Returns:
            A list of Gemini content dicts ready to pass to
            ``generate_content``.
        """
        contents = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "assistant" and msg.tool_calls:
                # Model requested tool calls — emit as function_call parts
                parts = []
                if msg.content:
                    parts.append(msg.content)
                for tc in msg.tool_calls:
                    parts.append(genai.protos.Part(
                        function_call=genai.protos.FunctionCall(
                            name=tc.name,
                            args=tc.arguments,
                        )
                    ))
                contents.append({"role": "model", "parts": parts})
                i += 1

            elif msg.role == "tool":
                # Batch ALL consecutive tool messages into ONE user turn.
                # Gemini requires every function_response for a single model turn
                # to arrive in a single user Content block.
                tool_parts = []
                while i < len(messages) and messages[i].role == "tool":
                    m = messages[i]
                    response_struct = Struct()
                    response_struct.update({"result": m.content})
                    tool_parts.append(genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=m.tool_call_id,
                            response=response_struct,
                        )
                    ))
                    i += 1
                contents.append({"role": "user", "parts": tool_parts})

            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [msg.content]})
                i += 1

            else:
                # user or system messages
                contents.append({"role": "user", "parts": [msg.content]})
                i += 1

        return contents

    def _build_gemini_tools(self, tool_specs: list[dict]) -> list[GeminiTool]:
        """Convert tool specs (JSON Schema format) to Gemini Tool objects.

        Args:
            tool_specs: PAI tool specs, each with ``name``, ``description``,
                and ``parameters`` keys.

        Returns:
            A single-element list holding a ``GeminiTool`` that wraps all
            function declarations.
        """
        declarations = [
            FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
            for spec in tool_specs
        ]
        return [GeminiTool(function_declarations=declarations)]

    def _parse_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool calls from a Gemini response.

        Gemini does not issue distinct call IDs, so the function name is
        reused as the ``ToolCall.id``.

        Args:
            response: The raw ``generate_content`` result.

        Returns:
            A list of ``ToolCall`` objects parsed from the response parts.
        """
        tool_calls = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=fc.name,           # Gemini doesn't issue IDs — use name
                    name=fc.name,
                    arguments=dict(fc.args),
                ))
        return tool_calls

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate a response from Gemini given a list of messages.

        Converts the internal Message format to Gemini's contents format,
        mapping "assistant" role to "model" and all other roles to "user".

        Side effects:
            Records the request with ``core.rate_tracker`` on success.

        Args:
            messages: The conversation history as a list of Message objects.

        Returns:
            An LLMResponse containing the generated text and model metadata.

        Raises:
            RateLimitError: If the API reports a rate limit or quota error.
            ProviderError: If the Gemini API call fails for any other reason.
        """
        try:
            contents = self._build_contents(messages)

            result = self.model.generate_content(contents)
            response_text = result.text

            logger.debug(f"Gemini responded: {response_text[:100]}...")
            from core import rate_tracker
            rate_tracker.record_request("gemini")
            return LLMResponse(content=response_text, metadata={"model": self.MODEL_NAME})
        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini API error: {error_str}")
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit hit: {error_str}")
            raise ProviderError(f"Gemini failed: {error_str}")

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Generate a response using Gemini's native function calling.

        Side effects:
            Records the request with ``core.rate_tracker`` on success.

        Args:
            messages: Full conversation history including any tool results.
            tools: List of tool specs in JSON-Schema function format.

        Returns:
            LLMResponse with tool_calls populated if the LLM chose a tool.
            ``content`` may be empty when only tool calls are returned.

        Raises:
            RateLimitError: If the API reports a rate limit or quota error.
            ProviderError: If the Gemini tool call fails for any other reason.
        """
        try:
            contents = self._build_contents(messages)
            gemini_tools = self._build_gemini_tools(tools) if tools else None

            result = self.model.generate_content(
                contents,
                tools=gemini_tools,
            )

            tool_calls = self._parse_tool_calls(result)
            from core import rate_tracker
            rate_tracker.record_request("gemini")

            # Extract text (may be empty if only tool calls returned)
            try:
                text = result.text
            except Exception:
                text = ""

            return LLMResponse(
                content=text,
                tool_calls=tool_calls,
                metadata={"model": self.MODEL_NAME},
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit: {error_str}")
            raise ProviderError(f"Gemini tool call failed: {error_str}")

    def health_check(self) -> bool:
        """Verify the Gemini API key is valid by listing available models.

        Returns:
            True if at least one model is returned.

        Raises:
            ProviderError: If no models are returned or the request fails.
        """
        try:
            models = list(genai.list_models())
            if not models:
                raise ProviderError("Gemini API returned no models — check your API key.")
            logger.debug(f"Gemini health check passed ({len(models)} models available)")
            return True
        except Exception as e:
            raise ProviderError(f"Gemini health check failed: {e}")
