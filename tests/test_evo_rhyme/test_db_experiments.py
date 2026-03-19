"""Tests for evo_rhyme.db experiment/arm APIs (with stubbed MySQL)."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

import evo_rhyme.db as db


class _DummyCursor:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None, row: Optional[Dict[str, Any]] = None):
        self._rows = rows or []
        self._row = row
        self.closed = False
        self.executed = []
        self.lastrowid = 1

    def execute(self, query: str, params: tuple | None = None):
        self.executed.append((query, params))

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class _DummyConn:
    def __init__(self, cursor_obj: _DummyCursor):
        self._cursor_obj = cursor_obj
        self.closed = False

    def cursor(self, dictionary: bool = False):
        return self._cursor_obj

    def commit(self):
        pass

    def close(self):
        self.closed = True


def _install_stub(monkeypatch, cursor: _DummyCursor):
    connector_mod = types.ModuleType("mysql.connector")
    connector_mod.connect = lambda **kwargs: _DummyConn(cursor)
    sys.modules["mysql"] = types.ModuleType("mysql")
    sys.modules["mysql"].connector = connector_mod
    sys.modules["mysql.connector"] = connector_mod


def test_insert_experiment_returns_id(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 7
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        eid = db.insert_experiment("test_exp", "desc", "single_control")
        assert eid == 7
        assert any("experiments" in str(e[0]) for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_insert_experiment_arm_returns_id(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 11
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        aid = db.insert_experiment_arm(7, "arm_a", {"elites": 5})
        assert aid == 11
        assert any("experiment_arms" in str(e[0]) for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_list_experiments_returns_list(monkeypatch):
    cursor = _DummyCursor(rows=[{"experiment_id": 1, "name": "e1", "description": None, "mode": None, "created_at": "2024-01-01"}])
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        out = db.list_experiments(limit=10, offset=0)
        assert len(out) == 1
        assert out[0]["name"] == "e1"
    finally:
        db._DB_CONFIG = None


def test_get_experiment_returns_one(monkeypatch):
    cursor = _DummyCursor(row={"experiment_id": 1, "name": "e1", "description": "d", "mode": "m", "created_at": "2024-01-01"})
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        out = db.get_experiment(1)
        assert out is not None
        assert out["name"] == "e1"
    finally:
        db._DB_CONFIG = None


def test_list_experiment_arms_parses_control_snapshot(monkeypatch):
    import json
    cursor = _DummyCursor(rows=[{"arm_id": 1, "experiment_id": 7, "arm_name": "a1", "control_snapshot": json.dumps({"elites": 5}), "created_at": "2024-01-01"}])
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        out = db.list_experiment_arms(7)
        assert len(out) == 1
        assert out[0]["control_snapshot"] == {"elites": 5}
    finally:
        db._DB_CONFIG = None


def test_insert_run_accepts_experiment_id_arm_id(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 100
    _install_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        run_id = db.insert_run("script", "theme", {"a": 1}, experiment_id=7, arm_id=11)
        assert run_id == 100
        assert any("experiment_id" in str(e[0]) or "arm_id" in str(e[0]) for e in cursor.executed)
    finally:
        db._DB_CONFIG = None
