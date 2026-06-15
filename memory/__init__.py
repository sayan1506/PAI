"""Long-term memory subsystem for the PAI assistant, backed by SQLite.

This package persists facts, user-defined shortcuts, reminders, and daily
rate-limit counters across sessions and exposes them for prompt injection.

Submodules:
    store:
        Low-level CRUD over the SQLite tables (facts, shortcuts, reminders,
        rate_limits). Each call opens and closes its own connection.
    retriever:
        Keyword extraction plus a context builder that formats relevant
        facts and shortcuts for injection into the system prompt.
    shortcut_manager:
        Validation layer that delegates shortcut define/remove/lookup/list
        operations to ``store``.
"""

from memory import store, retriever, shortcut_manager

__all__ = ["store", "retriever", "shortcut_manager"]
