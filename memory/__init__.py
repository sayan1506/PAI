"""
PAI Memory Module

Long-term memory backed by SQLite.
  store             — raw CRUD: facts + shortcuts
  retriever         — keyword search + context builder for system prompt injection
  shortcut_manager  — validated shortcut define/remove/lookup/list
"""

from memory import store, retriever, shortcut_manager

__all__ = ["store", "retriever", "shortcut_manager"]
