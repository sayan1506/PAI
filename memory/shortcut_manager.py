"""Validation layer for user-defined shortcuts.

Wraps the shortcut CRUD functions in :mod:`memory.store` with input
validation (non-empty name/description, name-length limits) and returns
human-readable confirmation strings suitable for surfacing back to the user.
"""

from memory.store import upsert_shortcut, delete_shortcut, get_shortcut, all_shortcuts


def define(name: str, description: str) -> str:
    """Save or overwrite a named shortcut after validating its inputs.

    Both fields are trimmed before storage. An existing shortcut with the
    same name (case-insensitive) is overwritten.

    Args:
        name: The shortcut's trigger name. Must be non-empty and at most
            100 characters.
        description: What the shortcut does. Must be non-empty.

    Returns:
        A confirmation string on success, or an error string describing the
        validation failure.
    """
    name = name.strip()
    description = description.strip()
    if not name:
        return "Error: shortcut name cannot be empty."
    if not description:
        return "Error: shortcut description cannot be empty."
    if len(name) > 100:
        return "Error: shortcut name too long (max 100 characters)."
    upsert_shortcut(name, description)
    return f"Shortcut '{name}' saved."


def remove(name: str) -> str:
    """Delete a shortcut by name.

    Args:
        name: The shortcut name to delete (matched case-insensitively).

    Returns:
        A confirmation string if the shortcut was deleted, or a not-found
        message otherwise.
    """
    deleted = delete_shortcut(name.strip())
    if deleted:
        return f"Shortcut '{name}' deleted."
    return f"No shortcut named '{name}' found."


def lookup(name: str) -> str | None:
    """Return the description for a named shortcut.

    Args:
        name: The shortcut name to look up (matched case-insensitively).

    Returns:
        The shortcut's description string, or None if no match exists.
    """
    return get_shortcut(name.strip())


def list_all() -> list[dict]:
    """Return all stored shortcuts.

    Returns:
        A list of dicts, each with ``name`` and ``description`` keys,
        ordered alphabetically by name.
    """
    return all_shortcuts()
