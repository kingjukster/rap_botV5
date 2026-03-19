"""Tests for evo_rhyme.db.execute_readonly_sql."""

from __future__ import annotations

import sys
import types

import pytest

import evo_rhyme.db as db


class _SqlCursor:
    """Cursor stub with fetchmany and column_names for execute_readonly_sql."""

    def __init__(self, rows: list[dict], column_names: tuple[str, ...] = ("x",)):
        self._rows = rows
        self._column_names = column_names
        self.closed = False
        self.executed = []

    @property
    def column_names(self):
        return self._column_names

    def execute(self, query: str, params: tuple | None = None):
        self.executed.append((query, params))

    def fetchmany(self, size: int | None = None):
        if size is None:
            return self._rows
        return self._rows[:size]

    def close(self):
        self.closed = True


class _SqlConn:
    def __init__(self, cursor_obj: _SqlCursor):
        self._cursor_obj = cursor_obj
        self.closed = False

    def cursor(self, dictionary: bool = False):
        return self._cursor_obj

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def _install_mysql_stub(monkeypatch, cursor_obj: _SqlCursor):
    connector_mod = types.ModuleType("mysql.connector")

    def _connect(**kwargs):
        return _SqlConn(cursor_obj)

    connector_mod.connect = _connect  # type: ignore[attr-defined]
    sys.modules["mysql"] = types.ModuleType("mysql")
    sys.modules["mysql"].connector = connector_mod  # type: ignore[attr-defined]
    sys.modules["mysql.connector"] = connector_mod


def test_reject_non_select():
    """Reject DELETE, INSERT, UPDATE, DROP, etc."""
    result = db.execute_readonly_sql("DELETE FROM runs")
    assert "error" in result
    assert "Only SELECT" in result["error"]

    result = db.execute_readonly_sql("DROP TABLE runs")
    assert "error" in result

    result = db.execute_readonly_sql("INSERT INTO runs VALUES (1)")
    assert "error" in result


def test_reject_multi_statement():
    """Reject queries containing multiple statements."""
    result = db.execute_readonly_sql("SELECT 1; DROP TABLE runs")
    assert "error" in result
    assert "Multiple statements" in result["error"]

    result = db.execute_readonly_sql("SELECT 1; SELECT 2")
    assert "error" in result


def test_valid_select_returns_rows(monkeypatch):
    """Valid SELECT returns columns, rows, row_count."""
    cursor = _SqlCursor(
        rows=[{"1": 1}],
        column_names=("1",),
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        result = db.execute_readonly_sql("SELECT 1")
        assert "rows" in result
        assert "columns" in result
        assert "row_count" in result
        assert result["columns"] == ["1"]
        assert result["rows"] == [{"1": 1}]
        assert result["row_count"] == 1
    finally:
        db._DB_CONFIG = None


def test_limit_enforcement(monkeypatch):
    """fetchmany limits returned rows to max_rows."""
    cursor = _SqlCursor(
        rows=[{"x": i} for i in range(10)],
        column_names=("x",),
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        result = db.execute_readonly_sql("SELECT * FROM runs", max_rows=5)
        assert "rows" in result
        assert len(result["rows"]) <= 5
        assert result["row_count"] <= 5
    finally:
        db._DB_CONFIG = None


def test_empty_result(monkeypatch):
    """Empty result returns row_count 0 and empty rows."""
    cursor = _SqlCursor(rows=[], column_names=("id",))
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        result = db.execute_readonly_sql("SELECT * FROM runs WHERE 1=0")
        assert result["row_count"] == 0
        assert result["rows"] == []
        assert result["columns"] == ["id"]
    finally:
        db._DB_CONFIG = None


def test_db_disabled_returns_error(monkeypatch):
    """When DB is disabled, return error dict."""
    monkeypatch.setenv("RAPBOT_USE_DB", "0")
    db._DB_CONFIG = None
    result = db.execute_readonly_sql("SELECT 1")
    assert "error" in result
    assert "Database operation failed" in result["error"]
