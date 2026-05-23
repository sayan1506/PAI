"""
memory/store.py

SQLite-backed persistent store for PAI long-term memory.
Tables: facts, shortcuts.
All public functions open + close their own connection (no global state).
Thread-safe via a module-level write lock + WAL mode.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    db_path = Path(config.MEMORY_DB_PATH).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Called once at agent startup."""
    with _lock:
        conn = _connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT    NOT NULL,
                value      TEXT    NOT NULL,
                source     TEXT    NOT NULL DEFAULT 'user',
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_key ON facts (key);

            CREATE TABLE IF NOT EXISTS shortcuts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                description TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                message     TEXT    NOT NULL,
                fire_at     TEXT    NOT NULL,
                fired       INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reminders_fire_at
                ON reminders (fire_at) WHERE fired = 0;
        """)
        conn.commit()
        conn.close()


# ── Facts ─────────────────────────────────────────────────────────────────────

def upsert_fact(key: str, value: str, source: str = "user") -> None:
    """Insert or update a fact. Key is normalised to snake_case."""
    key = key.strip().lower().replace(" ", "_")
    now = _now()
    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO facts (key, value, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE
                SET value      = excluded.value,
                    source     = excluded.source,
                    updated_at = excluded.updated_at
            """,
            (key, value.strip(), source, now, now),
        )
        conn.commit()
        conn.close()


def delete_fact(key: str) -> bool:
    """Delete a fact by key. Returns True if a row was removed."""
    key = key.strip().lower().replace(" ", "_")
    with _lock:
        conn = _connect()
        cursor = conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


def get_fact(key: str) -> str | None:
    """Retrieve a single fact value by exact normalised key."""
    key = key.strip().lower().replace(" ", "_")
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return row["value"] if row else None


def all_facts() -> list[dict]:
    """Return all facts ordered by most recently updated."""
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT key, value, source FROM facts ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def search_facts(keywords: list[str], top_k: int = 5) -> list[dict]:
    """
    Return facts whose key or value contains any of the given keywords.
    Case-insensitive substring match. Returns at most top_k rows.
    """
    if not keywords:
        return []
    clauses = " OR ".join(["key LIKE ? OR value LIKE ?"] * len(keywords))
    params: list[str] = []
    for kw in keywords:
        like = f"%{kw.lower()}%"
        params.extend([like, like])
    with _lock:
        conn = _connect()
        rows = conn.execute(
            f"SELECT key, value FROM facts WHERE {clauses} "
            f"ORDER BY updated_at DESC LIMIT ?",
            params + [top_k],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ── Shortcuts ─────────────────────────────────────────────────────────────────

def upsert_shortcut(name: str, description: str) -> None:
    """Insert or replace a named shortcut (case-insensitive on name)."""
    now = _now()
    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO shortcuts (name, description, created_at)
                VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE
                SET description = excluded.description
            """,
            (name.strip(), description.strip(), now),
        )
        conn.commit()
        conn.close()


def delete_shortcut(name: str) -> bool:
    """Delete a shortcut by name. Returns True if removed."""
    with _lock:
        conn = _connect()
        cursor = conn.execute(
            "DELETE FROM shortcuts WHERE name = ? COLLATE NOCASE",
            (name.strip(),),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


def get_shortcut(name: str) -> str | None:
    """Return the description of a named shortcut, or None."""
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT description FROM shortcuts WHERE name = ? COLLATE NOCASE",
            (name.strip(),),
        ).fetchone()
        conn.close()
        return row["description"] if row else None


def all_shortcuts() -> list[dict]:
    """Return all shortcuts ordered alphabetically by name."""
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT name, description FROM shortcuts ORDER BY name ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ── Reminders ─────────────────────────────────────────────────────────────────

def add_reminder(message: str, fire_at_iso: str) -> int:
    """
    Insert a new pending reminder. Returns the new row id.

    Args:
        message:     The text to display when the reminder fires.
        fire_at_iso: UTC ISO 8601 timestamp string for when to fire.

    Returns:
        The integer id of the newly inserted reminder row.
    """
    now = _now()
    with _lock:
        conn = _connect()
        cursor = conn.execute(
            "INSERT INTO reminders (message, fire_at, fired, created_at) "
            "VALUES (?, ?, 0, ?)",
            (message.strip(), fire_at_iso, now),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
    return row_id


def pending_reminders() -> list[dict]:
    """
    Return all reminders that are due (fire_at <= now UTC) and not yet fired.

    The scheduler calls this every poll interval to find reminders to fire.
    Returns a list of dicts with keys: id, message, fire_at, created_at.
    """
    now = _now()
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, message, fire_at, created_at FROM reminders "
            "WHERE fired = 0 AND fire_at <= ? "
            "ORDER BY fire_at ASC",
            (now,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def all_reminders(include_fired: bool = False) -> list[dict]:
    """
    Return all reminders, ordered by fire_at ascending.

    Args:
        include_fired: If False (default) only pending reminders are returned.

    Returns:
        List of dicts with keys: id, message, fire_at, fired, created_at.
    """
    with _lock:
        conn = _connect()
        if include_fired:
            rows = conn.execute(
                "SELECT id, message, fire_at, fired, created_at FROM reminders "
                "ORDER BY fire_at ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, message, fire_at, fired, created_at FROM reminders "
                "WHERE fired = 0 ORDER BY fire_at ASC"
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def mark_reminder_fired(reminder_id: int) -> None:
    """Mark a reminder as fired so the scheduler does not fire it again."""
    with _lock:
        conn = _connect()
        conn.execute(
            "UPDATE reminders SET fired = 1 WHERE id = ?",
            (reminder_id,),
        )
        conn.commit()
        conn.close()


def delete_reminder(reminder_id: int) -> bool:
    """
    Delete a reminder by id regardless of its fired state.

    Returns:
        True if a row was deleted, False if no reminder with that id exists.
    """
    with _lock:
        conn = _connect()
        cursor = conn.execute(
            "DELETE FROM reminders WHERE id = ?", (reminder_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
    return deleted
