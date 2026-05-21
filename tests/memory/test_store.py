"""Tests for memory/store.py"""
import pytest
import config
from memory.store import (
    init_db, upsert_fact, delete_fact, get_fact, all_facts, search_facts,
    upsert_shortcut, delete_shortcut, get_shortcut, all_shortcuts,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test_memory.db")
    monkeypatch.setattr(config, "MEMORY_DB_PATH", db_path)
    init_db()


class TestInitDb:
    def test_creates_tables(self, tmp_path, monkeypatch):
        import sqlite3
        db_path = config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "facts" in table_names
        assert "shortcuts" in table_names


class TestFacts:
    def test_upsert_inserts_new(self):
        upsert_fact("user_name", "Sayan")
        assert get_fact("user_name") == "Sayan"

    def test_upsert_updates_existing(self):
        upsert_fact("user_name", "Sayan")
        upsert_fact("user_name", "Sayan Ghosh")
        assert get_fact("user_name") == "Sayan Ghosh"

    def test_normalises_key(self):
        upsert_fact("User Name", "Sayan")
        assert get_fact("user_name") == "Sayan"

    def test_delete_removes_row(self):
        upsert_fact("temp_key", "temp_val")
        assert delete_fact("temp_key") is True
        assert get_fact("temp_key") is None

    def test_delete_returns_false_when_missing(self):
        assert delete_fact("nonexistent") is False

    def test_all_facts_returns_list(self):
        upsert_fact("a", "1")
        upsert_fact("b", "2")
        facts = all_facts()
        assert len(facts) >= 2
        assert all("key" in f and "value" in f for f in facts)

    def test_search_matches_by_value(self):
        upsert_fact("favorite_editor", "VS Code")
        results = search_facts(["code"])
        assert any(f["key"] == "favorite_editor" for f in results)

    def test_search_returns_empty_for_no_match(self):
        assert search_facts(["zxqwerty"]) == []

    def test_search_respects_top_k(self):
        for i in range(10):
            upsert_fact(f"test_key_{i}", f"test_value_{i}")
        results = search_facts(["test"], top_k=3)
        assert len(results) == 3


class TestShortcuts:
    def test_upsert_and_get(self):
        upsert_shortcut("work mode", "open VS Code and Spotify")
        assert get_shortcut("work mode") == "open VS Code and Spotify"

    def test_get_case_insensitive(self):
        upsert_shortcut("Work Mode", "open VS Code")
        assert get_shortcut("work mode") is not None

    def test_delete_returns_true(self):
        upsert_shortcut("temp", "desc")
        assert delete_shortcut("temp") is True

    def test_delete_returns_false_when_missing(self):
        assert delete_shortcut("nonexistent") is False

    def test_all_shortcuts_returns_list(self):
        upsert_shortcut("a", "desc a")
        upsert_shortcut("b", "desc b")
        shortcuts = all_shortcuts()
        assert len(shortcuts) >= 2
        assert all("name" in s and "description" in s for s in shortcuts)
