"""
PAI Agent Core

The Agent manages multi-turn conversation state, delegates to an LLM provider
for response generation, executes tool calls in a loop, and handles history
trimming and error recovery.
"""

from tools import get_tools, dispatch
from providers.base import LLMProvider, Message, LLMResponse, ToolCall
from core.exceptions import ProviderError, RateLimitError, ToolError
from core.logger import logger
from memory.store import init_db
from memory.retriever import build_memory_context
import config

SYSTEM_PROMPT = f"""You are {config.AGENT_NAME}, a personal AI assistant running locally on the user's computer.
You are helpful, concise, and direct. You do not add unnecessary filler phrases.
When you don't know something, say so clearly.
You have access to tools that let you control the computer — use them when the user's request requires it.
Only call a tool when necessary. For conversational questions, respond directly without tools."""


def _build_system_prompt() -> str:
    """Build the system prompt with real directory paths for the LLM."""
    from utils.platform_utils import special_dirs
    dirs = special_dirs()
    return f"""You are {config.AGENT_NAME}, a personal AI assistant running locally on the user's computer.
You are helpful, concise, and direct. You do not add unnecessary filler phrases.
When you don't know something, say so clearly.
You have access to tools that let you control the computer — use them when the user's request requires it.
Only call a tool when necessary. For conversational questions, respond directly without tools.

IMPORTANT RULES FOR TOOL USAGE:
- The "browser" tool controls web browsing. Use it for opening URLs, searching Google, searching YouTube, scrolling web pages.
- The "app_launcher" tool opens/closes desktop applications. Use it ONLY for launching or closing apps.
- The "file_system" tool manages files. Always use FULL ABSOLUTE PATHS like {dirs['desktop']}\\filename.txt
- The "terminal" tool runs shell commands. Use it for tasks like opening a folder in VS Code (e.g. code "C:\\path\\to\\folder"), checking versions, running scripts.
- Do NOT use the browser tool for tasks that aren't web browsing.
- Do NOT use app_launcher to minimize/maximize windows — it can only launch or close apps.
- When the user says "open a folder in VS Code", use terminal with command: code "full_path_to_folder"
- When the user says "search on YouTube", use browser with operation "search_youtube".
- When the user says "search" without specifying where, use browser with operation "search" (Google).
- If you cannot do something with the available tools, say so honestly.
- When the user says "open" followed by a well-known website name (github, youtube, gmail, google, twitter/x, reddit, etc.), use the browser tool with open_url and the website's URL (e.g. https://github.com). Do NOT try to read .lnk shortcut files.
- This computer runs Windows. Use PowerShell/Windows commands in terminal (not Linux commands like df, ls, cat). Examples: Get-Volume, Get-ChildItem, ipconfig, python --version.

User's directory paths (use these exact paths when the user mentions a named location):
  Home:      {dirs['home']}
  Desktop:   {dirs['desktop']}
  Documents: {dirs['documents']}
  Downloads: {dirs['downloads']}"""


class Agent:
    """Core conversation controller that manages message history and delegates to an LLM provider."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.history: list[Message] = []
        self._init_system_prompt()
        if config.MEMORY_ENABLED:
            init_db()
            logger.info("Memory store initialised.")
        logger.info("Agent initialised")

    def _init_system_prompt(self):
        """Add system context as a user message and a primed assistant response."""
        prompt = _build_system_prompt()
        self.history.append(Message(role="user", content=prompt))
        self.history.append(
            Message(
                role="assistant",
                content=f"Understood. I'm {config.AGENT_NAME}, ready to help.",
            )
        )

    def _trim_history(self):
        """Trim history to MAX_SESSION_TURNS, preserving system prompt pair.

        Only trims at user-message boundaries to avoid splitting tool-call
        blocks (assistant + tool result pairs).
        """
        system_messages = self.history[:2]
        conversation = self.history[2:]
        max_msgs = config.MAX_SESSION_TURNS * 2

        if len(conversation) <= max_msgs:
            return

        # Walk forward from the oldest messages; drop complete turn pairs
        # (user → ... → next user boundary)
        while len(conversation) > max_msgs:
            # Find the next user message after index 0
            next_user = next(
                (i for i, m in enumerate(conversation[1:], 1) if m.role == "user"),
                None,
            )
            if next_user is None:
                break
            conversation = conversation[next_user:]

        self.history = system_messages + conversation
        logger.debug(f"History trimmed to {len(self.history)} messages")

    def _call_provider(
        self,
        messages: list[Message],
        tool_specs: list[dict],
    ) -> LLMResponse:
        """
        Call the primary provider; fall back to Ollama if Gemini rate-limits.

        Primary call:
          Uses self.provider.generate_with_tools() if tool_specs is non-empty,
          else self.provider.generate().

        Fallback (Gemini → Ollama):
          Triggered only when:
            (a) a RateLimitError is raised, AND
            (b) config.LLM_PROVIDER == "gemini"
          On fallback, creates a fresh OllamaProvider() and retries the same call.
          Logs a WARNING so the user can see the swap in pai.log.
          If Ollama also fails, raises ProviderError with a clear message.
          For all other providers, re-raises RateLimitError unchanged.

        Args:
            messages:   Current conversation history (including system prompt).
            tool_specs: Serialised tool specs; may be empty list.

        Returns:
            LLMResponse from primary or fallback provider.

        Raises:
            RateLimitError: If primary rate-limits and no fallback applies.
            ProviderError:  If the fallback also fails.
        """
        def _invoke(provider) -> LLMResponse:
            if tool_specs:
                return provider.generate_with_tools(messages, tool_specs)
            return provider.generate(messages)

        try:
            return _invoke(self.provider)
        except RateLimitError as e:
            if config.LLM_PROVIDER != "gemini":
                raise  # no fallback defined for non-Gemini providers

            logger.warning(
                f"Gemini rate limit reached ({e}). "
                f"Attempting automatic fallback to Ollama."
            )
            try:
                from providers.ollama import OllamaProvider
                fallback = OllamaProvider()
                result = _invoke(fallback)
                logger.info("Ollama fallback succeeded for this turn.")
                return result
            except Exception as fallback_err:
                logger.error(f"Ollama fallback failed: {fallback_err}")
                raise ProviderError(
                    "Gemini rate limit reached and Ollama fallback failed. "
                    "Wait a moment or switch to a different provider."
                )

    def chat(self, user_input: str) -> str:
        """Process a user message and return the assistant's response.

        Implements a tool-calling loop: sends messages to the LLM, and if the
        LLM requests tool calls, executes them, appends results, and loops
        until the LLM produces a final text response or the iteration limit
        is reached.

        Args:
            user_input: The user's message string.

        Returns:
            The assistant's response content as a string.
        """
        user_input = user_input.strip()
        if not user_input:
            return "I didn't catch that. Could you say that again?"

        # Inject per-turn memory context into system prompt
        if config.MEMORY_ENABLED:
            mem_ctx = build_memory_context(user_input)
            base_prompt = _build_system_prompt()
            self.history[0] = Message(
                role="user",
                content=base_prompt + (mem_ctx if mem_ctx else ""),
            )

        self.history.append(Message(role="user", content=user_input))
        self._trim_history()
        logger.info(f"User: {user_input}")

        tools = get_tools()
        tool_specs = [t.to_function_spec() for t in tools]

        try:
            for iteration in range(config.MAX_TOOL_ITERATIONS):
                from core.timing import Timer
                with Timer("llm"):
                    response = self._call_provider(self.history, tool_specs)

                # No tool calls → final text response
                if not response.tool_calls:
                    reply = response.content.strip()
                    self.history.append(Message(role="assistant", content=reply))
                    logger.info(f"PAI: {reply[:100]}{'...' if len(reply) > 100 else ''}")
                    return reply

                # Tool calls → execute each, append results, loop again
                logger.info(f"Tool calls requested: {[tc.name for tc in response.tool_calls]}")
                self.history.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    try:
                        result = dispatch(tool_call.name, tool_call.arguments)
                        tool_output = result.output if result.success else f"Error: {result.error}"
                    except ToolError as e:
                        tool_output = f"Tool error: {e}"
                        logger.error(f"ToolError: {e}")

                    self.history.append(Message(
                        role="tool",
                        content=tool_output,
                        tool_call_id=tool_call.id,
                    ))

            # Hit iteration limit
            reply = "I wasn't able to complete that in the available steps. Please try a simpler request."
            logger.warning("Max tool iterations reached")
            return reply

        except RateLimitError as e:
            self.history.pop()
            logger.warning(f"Rate limit (no fallback available): {e}")
            return "Rate limit reached. Please wait a moment before trying again."
        except ProviderError as e:
            self.history.pop()
            logger.error(f"Provider error: {e}")
            return f"Something went wrong with the AI provider: {e}"

    def reset(self):
        """Clear conversation history and re-initialize the system prompt."""
        self.history = []
        self._init_system_prompt()
        logger.info("Conversation reset")
