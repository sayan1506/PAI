"""Tests for memory/retriever.py"""
from unittest.mock import patch
import pytest
import config
from memory.retriever import _extract_keywords, build_memory_context


class TestExtractKeywords:
    def test_strips_stopwords(self):
        keywords = _extract_keywords("what is my name")
        assert "what" not in keywords
        assert "is" not in keywords
        assert "my" not in keywords
        assert "name" in keywords

    def test_deduplicates(self):
        keywords = _extract_keywords("name name name")
        assert keywords.count("name") == 1


class TestBuildMemoryContext:
    def test_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", False)
        assert build_memory_context("anything") == ""

    @patch("memory.retriever.all_shortcuts", return_value=[])
    @patch("memory.retriever.search_facts", return_value=[])
    def test_returns_empty_when_no_matches(self, mock_search, mock_shortcuts, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", True)
        assert build_memory_context("hello") == ""

    @patch("memory.retriever.all_shortcuts", return_value=[])
    @patch("memory.retriever.search_facts", return_value=[{"key": "user_name", "value": "Sayan"}])
    def test_includes_matching_facts(self, mock_search, mock_shortcuts, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", True)
        monkeypatch.setattr(config, "MEMORY_INJECT_TOP_K", 5)
        result = build_memory_context("what is my name")
        assert "user name: Sayan" in result

    @patch("memory.retriever.all_shortcuts", return_value=[{"name": "work mode", "description": "open VS Code"}])
    @patch("memory.retriever.search_facts", return_value=[])
    def test_includes_shortcuts(self, mock_search, mock_shortcuts, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", True)
        monkeypatch.setattr(config, "MEMORY_INJECT_TOP_K", 5)
        result = build_memory_context("hello")
        assert '"work mode": open VS Code' in result

    @patch("memory.retriever.all_shortcuts", return_value=[])
    @patch("memory.retriever.search_facts", return_value=[{"key": "k", "value": "v"}])
    def test_wraps_in_memory_tags(self, mock_search, mock_shortcuts, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", True)
        monkeypatch.setattr(config, "MEMORY_INJECT_TOP_K", 5)
        result = build_memory_context("test")
        assert "[MEMORY]" in result
        assert "[/MEMORY]" in result
