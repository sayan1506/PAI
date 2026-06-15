"""Keyword-based memory retriever for system-prompt injection.

Extracts meaningful keywords from the user's message, searches the stored
facts via :mod:`memory.store`, and formats a compact context block that the
agent appends to the system prompt before each LLM call. User-defined
shortcuts are always included since they are typically few.

Notable behavior:
    - Retrieval is gated by ``config.MEMORY_ENABLED``.
    - A frozen stopword set filters out common, low-signal tokens.
    - All output is wrapped in ``[MEMORY] ... [/MEMORY]`` delimiters.
"""

import re

import config
from memory.store import search_facts, all_shortcuts


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "and", "but", "or", "nor", "so", "yet", "not", "no", "please",
    "just", "hey", "ok", "okay", "yeah", "yes", "tell", "show", "get",
    "make", "how", "when", "where", "why", "can", "help",
})


def _extract_keywords(text: str) -> list[str]:
    """Extract distinct, meaningful keywords from free-form text.

    Tokenises on alphanumeric runs, lowercases, then drops stopwords and
    tokens of two characters or fewer.

    Args:
        text: The raw input string to tokenise.

    Returns:
        A deduplicated list of keyword tokens (order not guaranteed).
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return list({t for t in tokens if t not in _STOPWORDS and len(t) > 2})


def build_memory_context(user_input: str) -> str:
    """Build a memory context block to append to the system prompt.

    Extracts keywords from the user's message, retrieves the most relevant
    stored facts, and appends all user-defined shortcuts. The pieces are
    formatted into a single delimited block for prompt injection.

    Args:
        user_input: The raw user message for the current turn.

    Returns:
        A formatted context string wrapped in ``[MEMORY]`` delimiters, or an
        empty string if memory is disabled or nothing relevant was found.
    """
    if not config.MEMORY_ENABLED:
        return ""

    lines: list[str] = []

    # Relevant facts
    keywords = _extract_keywords(user_input)
    facts = search_facts(keywords, top_k=config.MEMORY_INJECT_TOP_K)
    if facts:
        lines.append("Relevant things you know about this user:")
        for f in facts:
            display_key = f["key"].replace("_", " ")
            lines.append(f"  - {display_key}: {f['value']}")

    # All shortcuts (always inject — typically few)
    shortcuts = all_shortcuts()
    if shortcuts:
        lines.append("User-defined shortcuts (execute these when triggered by name):")
        for s in shortcuts:
            lines.append(f'  - "{s["name"]}": {s["description"]}')

    if not lines:
        return ""

    return "\n\n[MEMORY]\n" + "\n".join(lines) + "\n[/MEMORY]"
