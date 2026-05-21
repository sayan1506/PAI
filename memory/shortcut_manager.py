"""
memory/shortcut_manager.py

Validates and delegates shortcut CRUD to the store.
"""

from memory.store import upsert_shortcut, delete_shortcut, get_shortcut, all_shortcuts


def define(name: str, description: str) -> str:
    """Save or overwrite a named shortcut. Returns confirmation string."""
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
    """Delete a shortcut by name. Returns confirmation or not-found message."""
    deleted = delete_shortcut(name.strip())
    if deleted:
        return f"Shortcut '{name}' deleted."
    return f"No shortcut named '{name}' found."


def lookup(name: str) -> str | None:
    """Return the description for a named shortcut, or None if not found."""
    return get_shortcut(name.strip())


def list_all() -> list[dict]:
    """Return all shortcuts as a list of dicts with 'name' and 'description'."""
    return all_shortcuts()
