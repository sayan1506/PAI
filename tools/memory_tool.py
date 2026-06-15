"""
tools/memory_tool.py

MemoryTool — lets the LLM read and write PAI's long-term memory.

Operations:
  remember_fact    — store a key/value fact
  forget_fact      — delete a fact by key
  list_facts       — return all stored facts
  define_shortcut  — save a named command shortcut
  delete_shortcut  — remove a named shortcut
  list_shortcuts   — return all shortcuts
"""

import config
from tools.base import BaseTool, ToolResult
from memory import store, shortcut_manager


class MemoryTool(BaseTool):
    """Tool for reading and writing PAI's persistent long-term memory.

    Exposes two groups of operations dispatched by an ``operation`` argument:
    facts (``remember_fact``, ``forget_fact``, ``list_facts``) backed by the
    ``store`` module, and named command shortcuts (``define_shortcut``,
    ``delete_shortcut``, ``list_shortcuts``) backed by ``shortcut_manager``.
    Both persist across sessions. Gated by ``config.MEMORY_ENABLED``.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "memory"

    @property
    def description(self) -> str:
        """Return the LLM-facing description of this tool."""
        return (
            "Read and write PAI's long-term memory that persists across sessions. "
            "Use 'remember_fact' to save something the user wants you to remember "
            "(provide 'key' and 'value'). "
            "Use 'forget_fact' to delete a fact (provide 'key'). "
            "Use 'list_facts' to show everything currently stored in memory. "
            "Use 'define_shortcut' to save a named command sequence the user can trigger "
            "by name in future sessions (provide 'name' and 'description' of what to do). "
            "Use 'delete_shortcut' to remove a shortcut by 'name'. "
            "Use 'list_shortcuts' to show all saved shortcuts."
        )

    @property
    def parameters(self) -> dict:
        """Return the JSON Schema for this tool's arguments."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "remember_fact", "forget_fact", "list_facts",
                        "define_shortcut", "delete_shortcut", "list_shortcuts",
                    ],
                    "description": "The memory operation to perform.",
                },
                "key": {
                    "type": "string",
                    "description": "Fact key for remember_fact/forget_fact (e.g. 'user_name', 'favorite_editor').",
                },
                "value": {
                    "type": "string",
                    "description": "Fact value for remember_fact.",
                },
                "name": {
                    "type": "string",
                    "description": "Shortcut name for define_shortcut/delete_shortcut.",
                },
                "description": {
                    "type": "string",
                    "description": "What the shortcut does (for define_shortcut).",
                },
            },
            "required": ["operation"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Dispatch a memory operation to its handler.

        Reads the ``operation`` argument and routes to the matching
        ``_op_<operation>`` method.

        Args:
            **kwargs: Operation arguments. ``operation`` selects the action;
                remaining keys (``key``, ``value``, ``name``, ``description``)
                depend on the operation.

        Returns:
            A ``ToolResult`` from the handler, or an error if memory is
            disabled or the operation is unknown.

        Side Effects:
            Fact and shortcut operations read from and write to persistent
            storage.
        """
        if not config.MEMORY_ENABLED:
            return ToolResult(
                success=False,
                error="Memory is disabled. Set MEMORY_ENABLED=true in .env to enable it.",
            )

        operation = kwargs.get("operation")
        handler = getattr(self, f"_op_{operation}", None)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown operation '{operation}'.",
            )
        return handler(**kwargs)

    # ── Fact operations ───────────────────────────────────────────────────────

    def _op_remember_fact(self, **kwargs) -> ToolResult:
        """Store or update a key/value fact in memory.

        Args:
            **kwargs: Expects ``key`` and ``value``.

        Returns:
            A ``ToolResult`` confirming the stored fact, or an error if either
            value is missing.

        Side Effects:
            Upserts the fact in persistent storage.
        """
        key = kwargs.get("key", "").strip()
        value = kwargs.get("value", "").strip()
        if not key:
            return ToolResult(success=False, error="'key' is required.")
        if not value:
            return ToolResult(success=False, error="'value' is required.")
        store.upsert_fact(key, value)
        display_key = key.lower().replace(" ", "_")
        return ToolResult(success=True, output=f"Remembered: {display_key} = {value}")

    def _op_forget_fact(self, **kwargs) -> ToolResult:
        """Delete a stored fact by key.

        Args:
            **kwargs: Expects ``key``.

        Returns:
            A ``ToolResult`` confirming deletion, or an error if the key is
            missing or no such fact exists.

        Side Effects:
            Removes the fact from persistent storage when present.
        """
        key = kwargs.get("key", "").strip()
        if not key:
            return ToolResult(success=False, error="'key' is required.")
        deleted = store.delete_fact(key)
        if deleted:
            return ToolResult(success=True, output=f"Forgotten: {key}")
        return ToolResult(success=False, error=f"No fact with key '{key}' found in memory.")

    def _op_list_facts(self, **kwargs) -> ToolResult:
        """List all facts currently stored in memory.

        Args:
            **kwargs: Unused.

        Returns:
            A ``ToolResult`` whose ``output`` is the newline-joined facts, or
            a message noting that none are stored.
        """
        facts = store.all_facts()
        if not facts:
            return ToolResult(success=True, output="No facts stored in memory yet.")
        lines = [f"{f['key'].replace('_', ' ')}: {f['value']}" for f in facts]
        return ToolResult(success=True, output="\n".join(lines))

    # ── Shortcut operations ───────────────────────────────────────────────────

    def _op_define_shortcut(self, **kwargs) -> ToolResult:
        """Define a named command shortcut.

        Args:
            **kwargs: Expects ``name`` and ``description`` of what the
                shortcut does.

        Returns:
            A ``ToolResult`` confirming the shortcut, or an error if a value
            is missing or the manager reports an error.

        Side Effects:
            Persists the shortcut via ``shortcut_manager``.
        """
        name = kwargs.get("name", "").strip()
        description = kwargs.get("description", "").strip()
        if not name:
            return ToolResult(success=False, error="'name' is required.")
        if not description:
            return ToolResult(success=False, error="'description' is required.")
        msg = shortcut_manager.define(name, description)
        success = not msg.startswith("Error:")
        if success:
            return ToolResult(success=True, output=msg)
        return ToolResult(success=False, error=msg)

    def _op_delete_shortcut(self, **kwargs) -> ToolResult:
        """Delete a named shortcut.

        Args:
            **kwargs: Expects ``name``.

        Returns:
            A ``ToolResult`` reporting the outcome message, or an error if the
            name is missing.

        Side Effects:
            Removes the shortcut via ``shortcut_manager``.
        """
        name = kwargs.get("name", "").strip()
        if not name:
            return ToolResult(success=False, error="'name' is required.")
        msg = shortcut_manager.remove(name)
        return ToolResult(success=True, output=msg)

    def _op_list_shortcuts(self, **kwargs) -> ToolResult:
        """List all defined shortcuts.

        Args:
            **kwargs: Unused.

        Returns:
            A ``ToolResult`` whose ``output`` is the newline-joined shortcuts,
            or a message noting that none are defined.
        """
        shortcuts = shortcut_manager.list_all()
        if not shortcuts:
            return ToolResult(success=True, output="No shortcuts defined yet.")
        lines = [f'"{s["name"]}": {s["description"]}' for s in shortcuts]
        return ToolResult(success=True, output="\n".join(lines))
