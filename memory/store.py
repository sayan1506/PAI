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
