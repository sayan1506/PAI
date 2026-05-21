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
    """Read and write PAI's long-term memory."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
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
        key = kwargs.get("key", "").strip()
        if not key:
            return ToolResult(success=False, error="'key' is required.")
        deleted = store.delete_fact(key)
        if deleted:
            return ToolResult(success=True, output=f"Forgotten: {key}")
        return ToolResult(success=False, error=f"No fact with key '{key}' found in memory.")

    def _op_list_facts(self, **kwargs) -> ToolResult:
        facts = store.all_facts()
        if not facts:
            return ToolResult(success=True, output="No facts stored in memory yet.")
        lines = [f"{f['key'].replace('_', ' ')}: {f['value']}" for f in facts]
        return ToolResult(success=True, output="\n".join(lines))

    # ── Shortcut operations ───────────────────────────────────────────────────

    def _op_define_shortcut(self, **kwargs) -> ToolResult:
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
        name = kwargs.get("name", "").strip()
        if not name:
            return ToolResult(success=False, error="'name' is required.")
        msg = shortcut_manager.remove(name)
        return ToolResult(success=True, output=msg)

    def _op_list_shortcuts(self, **kwargs) -> ToolResult:
        shortcuts = shortcut_manager.list_all()
        if not shortcuts:
            return ToolResult(success=True, output="No shortcuts defined yet.")
        lines = [f'"{s["name"]}": {s["description"]}' for s in shortcuts]
        return ToolResult(success=True, output="\n".join(lines))
