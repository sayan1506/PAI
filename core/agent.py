"""
PAI Agent Core

The Agent manages multi-turn conversation state, delegates to an LLM provider
for response generation, and handles history trimming and error recovery.
"""

from providers.base import LLMProvider, Message, LLMResponse
from core.logger import logger
from core.exceptions import ProviderError, RateLimitError
import config

SYSTEM_PROMPT = f"""You are {config.AGENT_NAME}, a personal AI assistant running locally on the user's computer.
You are helpful, concise, and direct. You do not add unnecessary filler phrases.
When you don't know something, say so clearly.
In later versions you will have tools to control the computer — for now, you are a conversational assistant."""


class Agent:
    """Core conversation controller that manages message history and delegates to an LLM provider."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.history: list[Message] = []
        self._init_system_prompt()
        logger.info("Agent initialised")

    def _init_system_prompt(self):
        """Add system context as a user message and a primed assistant response."""
        self.history.append(Message(role="user", content=SYSTEM_PROMPT))
        self.history.append(
            Message(
                role="assistant",
                content=f"Understood. I'm {config.AGENT_NAME}, ready to help.",
            )
        )

    def _trim_history(self):
        """Preserve the first 2 messages (system prompt pair) and trim oldest conversation messages.

        Called after appending the user message but before the provider call.
        Keeps total history within MAX_SESSION_TURNS * 2 + 2 messages
        (accounting for the assistant response that will be appended after).
        """
        system_messages = self.history[:2]
        conversation = self.history[2:]
        max_conversation = config.MAX_SESSION_TURNS * 2
        if len(conversation) > max_conversation:
            # Keep one fewer to leave room for the assistant response
            keep = max_conversation - 1
            conversation = conversation[-keep:]
            self.history = system_messages + conversation
            logger.debug(f"History trimmed to {len(self.history)} messages")

    def chat(self, user_input: str) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to history, calls the provider, appends the
        assistant response, and returns the response content. Catches ProviderError
        and returns a user-friendly error message.

        Args:
            user_input: The user's message string.

        Returns:
            The assistant's response content as a string.
        """
        user_input = user_input.strip()
        if not user_input:
            return "I didn't catch that. Could you say that again?"

        self.history.append(Message(role="user", content=user_input))
        self._trim_history()
        logger.info(f"User: {user_input}")

        try:
            response = self.provider.generate(self.history)
            reply = response.content.strip()
            self.history.append(Message(role="assistant", content=reply))
            logger.info(f"PAI: {reply[:100]}{'...' if len(reply) > 100 else ''}")
            return reply
        except RateLimitError as e:
            error_msg = "Rate limit reached. Please wait a moment before trying again."
            logger.warning(f"Rate limit: {e}")
            self.history.pop()  # Remove the failed user message
            return error_msg
        except ProviderError as e:
            error_msg = f"Something went wrong with the AI provider: {e}"
            logger.error(f"Provider error: {e}")
            self.history.pop()
            return error_msg

    def reset(self):
        """Clear conversation history and re-initialize the system prompt."""
        self.history = []
        self._init_system_prompt()
        logger.info("Conversation reset")
