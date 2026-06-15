"""SQLite-backed persistent store for PAI long-term memory.

Provides low-level CRUD over four tables: ``facts``, ``shortcuts``,
``reminders``, and ``rate_limits``. There is no global connection — every
public function opens and closes its own connection, keeping the module
free of shared connection state.

Threading and I/O:
    A module-level :class:`threading.Lock` serialises all writes, and each
    connection enables SQLite WAL journaling and foreign-key enforcement.
    Connections are opened with ``check_same_thread=False`` so they may be
    used from the daemon threads that drive reminders and rate tracking.
    Timestamps are stored as UTC ISO 8601 strings.
"""

import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import config

_lock = threading.Lock()


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    """Open a configured SQLite connection to the memory database.

    Expands the configured DB path, creates the parent directory if needed,
    and enables WAL journaling and foreign-key enforcement. The returned
    connection uses :class:`sqlite3.Row` as its row factory.

    Returns:
        An open :class:`sqlite3.Connection`. The caller owns it and must
        close it.

    Raises:
        sqlite3.Error: If the database cannot be opened.
    """
    db_path = Path(config.MEMORY_DB_PATH).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all memory tables and indexes if they do not yet exist.

    Idempotent; safe to call on every startup. Acquires the write lock and
    creates the ``facts``, ``shortcuts``, ``reminders``, and ``rate_limits``
    tables along with their supporting indexes.

    Side Effects:
        Writes schema to the SQLite database file.
    """
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

            CREATE TABLE IF NOT EXISTS rate_limits (
                provider   TEXT    NOT NULL,
                date       TEXT    NOT NULL,
                count      INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT    NOT NULL,
                PRIMARY KEY (provider, date)
            );
        """)
        conn.commit()
        conn.close()


# ── Facts ─────────────────────────────────────────────────────────────────────

def upsert_fact(key: str, value: str, source: str = "user") -> None:
    """Insert a fact or update the existing one with the same key.

    The key is normalised to lowercase snake_case (spaces become
    underscores) so lookups are stable. On conflict the value, source, and
    ``updated_at`` timestamp are overwritten; ``created_at`` is preserved.

    Args:
        key: The fact identifier; normalised before storage.
        value: The fact's value; surrounding whitespace is stripped.
        source: Where the fact came from. Defaults to ``"user"``.

    Side Effects:
        Commits a write to the ``facts`` table under the write lock.
    """
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
    """Delete a fact by key.

    Args:
        key: The fact identifier; normalised to snake_case before matching.

    Returns:
        True if a row was removed, False if no matching key existed.

    Side Effects:
        Commits a delete to the ``facts`` table under the write lock.
    """
    key = key.strip().lower().replace(" ", "_")
    with _lock:
        conn = _connect()
        cursor = conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


def get_fact(key: str) -> str | None:
    """Retrieve a single fact value by its exact normalised key.

    Args:
        key: The fact identifier; normalised to snake_case before matching.

    Returns:
        The stored value string, or None if the key is not present.
    """
    key = key.strip().lower().replace(" ", "_")
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return row["value"] if row else None


def all_facts() -> list[dict]:
    """Return every stored fact, most recently updated first.

    Returns:
        A list of dicts, each with ``key``, ``value``, and ``source`` keys.
    """
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT key, value, source FROM facts ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def search_facts(keywords: list[str], top_k: int = 5) -> list[dict]:
    """Return facts whose key or value matches any of the given keywords.

    Performs a case-insensitive substring (``LIKE``) match against both the
    key and value columns, ordered by most recently updated.

    Args:
        keywords: Keyword strings to match. An empty list yields no results.
        top_k: Maximum number of rows to return. Defaults to 5.

    Returns:
        A list of dicts (at most ``top_k``), each with ``key`` and ``value``
        keys. Empty if ``keywords`` is empty or nothing matched.
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
    """Insert a shortcut or update the existing one with the same name.

    Names are matched case-insensitively (``COLLATE NOCASE``). On conflict
    only the description is updated.

    Args:
        name: The shortcut's trigger name; whitespace is stripped.
        description: What the shortcut does; whitespace is stripped.

    Side Effects:
        Commits a write to the ``shortcuts`` table under the write lock.
    """
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
    """Delete a shortcut by name.

    Args:
        name: The shortcut name to delete (matched case-insensitively).

    Returns:
        True if a row was removed, False if no matching name existed.

    Side Effects:
        Commits a delete to the ``shortcuts`` table under the write lock.
    """
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
    """Return the description of a named shortcut.

    Args:
        name: The shortcut name to look up (matched case-insensitively).

    Returns:
        The description string, or None if no matching shortcut exists.
    """
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT description FROM shortcuts WHERE name = ? COLLATE NOCASE",
            (name.strip(),),
        ).fetchone()
        conn.close()
        return row["description"] if row else None


def all_shortcuts() -> list[dict]:
    """Return every stored shortcut, ordered alphabetically by name.

    Returns:
        A list of dicts, each with ``name`` and ``description`` keys.
    """
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


# ── Rate Limits ───────────────────────────────────────────────────────────────

def get_daily_rate_count(provider: str) -> int:
    """Return today's request count for the given provider.

    Args:
        provider: The provider identifier whose count to read.

    Returns:
        Today's count for the provider, or 0 if no row exists for today
        (UTC-derived calendar date).
    """
    today = date.today().isoformat()
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT count FROM rate_limits WHERE provider = ? AND date = ?",
            (provider, today),
        ).fetchone()
        conn.close()
    return row["count"] if row else 0


def increment_daily_rate_count(provider: str) -> int:
    """Increment today's request count for the given provider by one.

    Inserts a row for ``(provider, today)`` if none exists, otherwise bumps
    the existing count.

    Args:
        provider: The provider identifier whose count to increment.

    Returns:
        The provider's new total count for today.

    Side Effects:
        Commits a write to the ``rate_limits`` table under the write lock.
    """
    today = date.today().isoformat()
    now   = _now()
    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO rate_limits (provider, date, count, updated_at)
                VALUES (?, ?, 1, ?)
            ON CONFLICT(provider, date) DO UPDATE
                SET count      = count + 1,
                    updated_at = excluded.updated_at
            """,
            (provider, today, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count FROM rate_limits WHERE provider = ? AND date = ?",
            (provider, today),
        ).fetchone()
        conn.close()
    return row["count"] if row else 1
