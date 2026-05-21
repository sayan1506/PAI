"""
memory/retriever.py

Keyword-based context retriever. Extracts meaningful words from the
user's message, searches the facts table, and formats a context block
to inject into the system prompt before each LLM call.
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
    """
    Tokenise text, lowercase, drop stopwords and single-char tokens.
    Returns deduplicated list.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return list({t for t in tokens if t not in _STOPWORDS and len(t) > 2})


def build_memory_context(user_input: str) -> str:
    """
    Return a formatted memory context block to append to the system prompt.
    Empty string if memory is disabled or nothing relevant is found.
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
