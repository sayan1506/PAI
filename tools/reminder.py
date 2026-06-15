"""
tools/reminder.py

ReminderTool — lets the LLM set, list, and cancel time-based reminders.

Actions:
  set     — schedule a reminder (params: when, message)
  list    — show all pending reminders
  cancel  — delete a reminder (params: id  OR  message for fuzzy match)

The LLM provides 'when' as:
  - A relative string: "10m", "2h", "30s"   (N minutes / hours / seconds)
  - An ISO 8601 UTC timestamp: "2024-01-15T14:30:00Z"

ReminderTool._parse_when() converts both forms to an absolute UTC datetime,
then stores the ISO string in SQLite. The ReminderScheduler reads this
table on each poll cycle and fires past-due reminders as notifications.
"""

import re
from datetime import datetime, timedelta, timezone

import config
from tools.base import BaseTool, ToolResult
from core.logger import logger
from memory import store


class ReminderTool(BaseTool):
    """Tool for setting, listing, and cancelling time-based reminders.

    Dispatches a single ``action`` argument to handlers for creating a
    reminder, listing pending ones, and cancelling by id or fuzzy message
    match. Reminders are stored in SQLite via the ``store`` module and fire as
    desktop notifications, persisting across PAI restarts. The ``'when'``
    value accepts relative durations (``'10m'``, ``'2h'``, ``'30s'``) or an
    ISO 8601 UTC timestamp. Gated by ``config.REMINDERS_ENABLED`` at
    registration time.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "reminder"

    @property
    def description(self) -> str:
        """Return the LLM-facing description of this tool."""
        return (
            "Set, list, and cancel time-based reminders. "
            "Reminders fire as desktop notifications at the specified time. "
            "They persist across PAI restarts. "
            "Use action='set' with 'when' and 'message' to create a reminder. "
            "  'when' must be a relative duration like '10m' (10 minutes), "
            "  '2h' (2 hours), '45s' (45 seconds), or an ISO 8601 UTC timestamp. "
            "  Convert the user's natural language time to one of these formats. "
            "Use action='list' to show all pending reminders with their IDs and times. "
            "Use action='cancel' with 'id' (integer) to cancel a specific reminder, "
            "  or with 'message' (partial text) to cancel the first matching reminder."
        )

    @property
    def parameters(self) -> dict:
        """Return the JSON Schema for this tool's arguments."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "list", "cancel"],
                    "description": "The reminder action to perform.",
                },
                "when": {
                    "type": "string",
                    "description": (
                        "For action='set': relative duration ('10m', '2h', '30s') "
                        "or ISO 8601 UTC timestamp. Required for 'set'."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "For action='set': what to remind the user about. "
                        "For action='cancel': partial message text to match."
                    ),
                },
                "id": {
                    "type": "integer",
                    "description": "For action='cancel': the integer reminder ID.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Route a reminder action to its handler.

        Dispatches on the ``action`` argument and catches all exceptions,
        returning them as failed results rather than raising.

        Args:
            **kwargs: Expects ``action`` (``'set'``, ``'list'``, or
                ``'cancel'``) plus action-specific keys (``when``,
                ``message``, ``id``).

        Returns:
            The ``ToolResult`` from the selected handler, or an error result
            for an unknown action or any raised exception.
        """
        action = kwargs.get("action", "")
        try:
            if action == "set":
                return self._set(kwargs.get("when", ""), kwargs.get("message", ""))
            elif action == "list":
                return self._list()
            elif action == "cancel":
                return self._cancel(
                    reminder_id=kwargs.get("id"),
                    message=kwargs.get("message", ""),
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action!r}. Use 'set', 'list', or 'cancel'.",
                )
        except Exception as e:
            logger.error(f"ReminderTool error ({action}): {e}")
            return ToolResult(success=False, error=str(e))

    # ── Action handlers ────────────────────────────────────────────────────

    def _set(self, when: str, message: str) -> ToolResult:
        """Schedule a new reminder.

        Parses ``when`` into an absolute UTC time, stores the reminder, and
        builds a human-readable confirmation with a relative time label.

        Args:
            when: Relative duration (``'10m'``, ``'2h'``, ``'30s'``) or ISO
                8601 UTC timestamp.
            message: What to remind the user about.

        Returns:
            A ``ToolResult`` confirming the reminder (with its id and fire
            time), or an error if ``when``/``message`` is missing or ``when``
            cannot be parsed.

        Side Effects:
            Persists the reminder via ``store.add_reminder``.
        """
        when = when.strip()
        message = message.strip()

        if not when:
            return ToolResult(
                success=False,
                error="'when' is required for action='set'.",
            )
        if not message:
            return ToolResult(
                success=False,
                error="'message' is required for action='set'.",
            )

        try:
            fire_at = self._parse_when(when)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        fire_at_iso = fire_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        reminder_id = store.add_reminder(message, fire_at_iso)

        # Human-readable time label
        now_utc = datetime.now(timezone.utc)
        delta = fire_at - now_utc
        minutes = int(delta.total_seconds() / 60)

        if minutes < 1:
            time_label = "in less than a minute"
        elif minutes == 1:
            time_label = "in 1 minute"
        elif minutes < 60:
            time_label = f"in {minutes} minutes"
        else:
            hours = minutes // 60
            time_label = f"in about {hours} hour{'s' if hours > 1 else ''}"

        local_time = fire_at.strftime("%H:%M UTC")
        output = (
            f"Reminder #{reminder_id} set: '{message}' will fire at "
            f"{local_time} ({time_label})."
        )
        logger.info(f"ReminderTool: {output}")
        return ToolResult(success=True, output=output)

    def _list(self) -> ToolResult:
        """List all pending (not-yet-fired) reminders.

        Each entry shows the reminder id, message, absolute fire time, and
        minutes remaining.

        Returns:
            A ``ToolResult`` whose ``output`` is the newline-joined reminders,
            or a message noting there are none.
        """
        reminders = store.all_reminders(include_fired=False)
        if not reminders:
            return ToolResult(success=True, output="You have no pending reminders.")

        now_utc = datetime.now(timezone.utc)
        lines = []
        for r in reminders:
            fire_at = datetime.fromisoformat(r["fire_at"].replace("Z", "+00:00"))
            delta = fire_at - now_utc
            minutes = max(0, int(delta.total_seconds() / 60))
            time_str = fire_at.strftime("%H:%M UTC")
            lines.append(
                f"#{r['id']} — '{r['message']}' — fires at {time_str}"
                f" (in {minutes} min)"
            )

        return ToolResult(success=True, output="\n".join(lines))

    def _cancel(self, reminder_id: int | None, message: str) -> ToolResult:
        """Cancel a reminder by id or by fuzzy message match.

        If ``reminder_id`` is given, that specific reminder is deleted.
        Otherwise, if ``message`` is given, the first pending reminder whose
        message contains it (case-insensitive) is deleted.

        Args:
            reminder_id: Integer id of the reminder to cancel, or None.
            message: Partial message text to match when no id is provided.

        Returns:
            A ``ToolResult`` confirming the cancellation, or an error if
            nothing matched or neither argument was supplied.

        Side Effects:
            Deletes the matched reminder via ``store.delete_reminder``.
        """
        if reminder_id is not None:
            deleted = store.delete_reminder(int(reminder_id))
            if deleted:
                return ToolResult(
                    success=True,
                    output=f"Reminder #{reminder_id} cancelled.",
                )
            return ToolResult(
                success=False,
                error=f"No reminder with ID {reminder_id} found.",
            )

        if message:
            needle = message.strip().lower()
            reminders = store.all_reminders(include_fired=False)
            for r in reminders:
                if needle in r["message"].lower():
                    store.delete_reminder(r["id"])
                    return ToolResult(
                        success=True,
                        output=f"Reminder #{r['id']} ('{r['message']}') cancelled.",
                    )
            return ToolResult(
                success=False,
                error=f"No pending reminder matching '{message}' found.",
            )

        return ToolResult(
            success=False,
            error="Provide either 'id' or 'message' to cancel a reminder.",
        )

    # ── Time parsing ───────────────────────────────────────────────────────

    def _parse_when(self, when: str) -> datetime:
        """Parse a 'when' string into an absolute UTC datetime.

        Accepted formats:
          - ``"Nm"`` → now + N minutes  (e.g. ``"10m"``, ``"5m"``)
          - ``"Nh"`` → now + N hours    (e.g. ``"2h"``, ``"1h"``)
          - ``"Ns"`` → now + N seconds  (e.g. ``"30s"``, ``"90s"``)
          - ISO 8601 UTC string ending in ``"Z"`` or ``"+00:00"`` → parsed
            directly (naive timestamps are assumed to be UTC)

        Args:
            when: The raw ``when`` string to parse.

        Returns:
            The resolved absolute UTC ``datetime``.

        Raises:
            ValueError: If the format is unrecognised or N is not a positive
                integer.
        """
        when = when.strip()
        now = datetime.now(timezone.utc)

        # Relative shorthand: e.g. "10m", "2h", "30s"
        match = re.fullmatch(r"(\d+)([mhs])", when, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            unit = match.group(2).lower()
            if n <= 0:
                raise ValueError(f"Duration must be a positive integer, got: {n}")
            if unit == "m":
                return now + timedelta(minutes=n)
            if unit == "h":
                return now + timedelta(hours=n)
            if unit == "s":
                return now + timedelta(seconds=n)

        # ISO 8601 UTC
        try:
            normalized = when.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

        raise ValueError(
            f"Cannot parse 'when' value: '{when}'. "
            "Use a relative duration like '10m', '2h', '30s', "
            "or an ISO 8601 UTC timestamp like '2024-01-15T14:30:00Z'."
        )
