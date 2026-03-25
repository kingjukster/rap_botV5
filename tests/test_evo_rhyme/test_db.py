import sys
import types
from typing import Any, Dict, List, Optional

import pytest

import evo_rhyme.db as db


@pytest.fixture(autouse=True)
def _reset_archive_schema_normalization_flag():
    """Each test gets a fresh archive_cells normalization gate (ALTER DROP may run once)."""
    db._ARCHIVE_CELLS_NORMALIZED = False
    yield
    db._ARCHIVE_CELLS_NORMALIZED = False


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
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self, dictionary: bool = False):
        return self._cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _install_mysql_stub(monkeypatch, cursor_obj: _DummyCursor):
    connector_mod = types.ModuleType("mysql.connector")

    def _connect(**kwargs):
        return _DummyConn(cursor_obj)

    connector_mod.connect = _connect  # type: ignore[attr-defined]

    mysql_mod = types.ModuleType("mysql")
    mysql_mod.connector = connector_mod  # type: ignore[attr-defined]

    sys.modules["mysql"] = mysql_mod
    sys.modules["mysql.connector"] = connector_mod


def test_db_disabled_returns_defaults(monkeypatch):
    monkeypatch.setenv("RAPBOT_USE_DB", "0")
    db._DB_CONFIG = None

    assert db.db_enabled() is False

    with db.connection() as conn:
        assert conn is None

    assert db.get_connection() is None
    assert db.score_cache_get("h", "type", "scheme") is None
    assert db.insert_run("script", "theme", {"a": 1}) == -1
    assert db.list_runs() == []
    assert db.count_runs() == 0
    # Ensure later tests see DB disabled configuration.
    db._DB_CONFIG = None


def test_db_enabled_uses_mysql_stub_for_simple_ops(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {
                "run_id": 1,
                "script_name": "script.py",
                "theme_keywords": "theme",
                "config_json": '{"foo": "bar"}',
                "status": "completed",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            }
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)

    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None

    conn = db.get_connection()
    assert isinstance(conn, _DummyConn)

    res = db.list_runs(limit=10, offset=0, status_filter=None)
    assert res and res[0]["config_json"] == {"foo": "bar"}
    # Reset so subsequent tests don't use cached enabled config.
    db._DB_CONFIG = None


def test_score_cache_get_and_put_with_stub(monkeypatch):
    row = {"scores_json": '{"score": 0.5}'}
    cursor = _DummyCursor(row=row)
    _install_mysql_stub(monkeypatch, cursor)

    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None

    # First ensure put executes and commits without error.
    db.score_cache_put("hash", "type", "scheme", {"score": 0.5})

    # Then exercise get and JSON decoding logic.
    result = db.score_cache_get("hash", "type", "scheme")
    assert result == {"score": 0.5}
    # Reset so score_couplet in other tests does not hit cached config.
    db._DB_CONFIG = None


# ---------------------------------------------------------------------------
# Area 4: DB – insert_run, update_run_status, insert_generation, insert_candidate,
# insert_lineage, upsert_archive_cell, upsert_archive_cells_batch, get_run,
# list_generations, list_candidates, count_runs, load_archive_cells, JSON/error paths.
# ---------------------------------------------------------------------------


def test_insert_run_returns_run_id_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 42
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        run_id = db.insert_run("script.py", "theme", {"key": "value"})
        assert run_id == 42
    finally:
        db._DB_CONFIG = None


def test_update_run_status_with_stub(monkeypatch):
    cursor = _DummyCursor()
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.update_run_status(1, "completed")
        assert len(cursor.executed) >= 1
    finally:
        db._DB_CONFIG = None


def test_insert_generation_with_stub(monkeypatch):
    cursor = _DummyCursor()
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.insert_generation(1, 0, 0.9, 0.5, 0.3, 0.2, {"best_raw": 0.95})
        assert any("INSERT INTO generations" in str(e[0]) for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_insert_candidate_returns_id_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 10
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        cid = db.insert_candidate(1, 0, "couplet", "COUPLET", ["line1", "line2"], 0.8, {"end_rhyme": 0.9})
        assert cid == 10
    finally:
        db._DB_CONFIG = None


def test_insert_lineage_with_stub(monkeypatch):
    cursor = _DummyCursor()
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.insert_lineage(2, 1, "mutation", 0)
        assert any("lineage" in str(e[0]).lower() for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_upsert_archive_cell_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 1
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.upsert_archive_cell(1, "0_0_0", [["a", "b"]], 0.7, None)
        assert len(cursor.executed) >= 2  # INSERT candidates then INSERT/UPDATE archive_cells
    finally:
        db._DB_CONFIG = None


def test_upsert_archive_cells_batch_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 1
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.upsert_archive_cells_batch(1, [("0_0_0", [["x"]], 0.5, None)])
        assert len(cursor.executed) >= 2
    finally:
        db._DB_CONFIG = None


def test_get_run_returns_none_when_missing(monkeypatch):
    cursor = _DummyCursor(row=None)
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        assert db.get_run(999) is None
    finally:
        db._DB_CONFIG = None


def test_get_run_returns_run_with_decoded_config(monkeypatch):
    cursor = _DummyCursor(
        row={
            "run_id": 1,
            "script_name": "s",
            "theme_keywords": "t",
            "config_json": '{"a": 1}',
            "status": "completed",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        run = db.get_run(1)
        assert run is not None
        assert run["config_json"] == {"a": 1}
    finally:
        db._DB_CONFIG = None


def test_get_run_config_json_decode_error_returns_empty_dict(monkeypatch):
    cursor = _DummyCursor(
        row={
            "run_id": 1,
            "script_name": "s",
            "theme_keywords": "t",
            "config_json": "not valid json {",
            "status": "completed",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        run = db.get_run(1)
        assert run is not None
        assert run["config_json"] == {}
    finally:
        db._DB_CONFIG = None


def test_list_generations_with_stub(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {
                "gen": 0,
                "best_fitness": 0.9,
                "avg_fitness": 0.5,
                "diversity": 0.3,
                "acceptance_rate": 0.2,
                "extra_json": '{"x": 1}',
                "created_at": "2024-01-01",
            }
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        gens = db.list_generations(1)
        assert len(gens) == 1
        assert gens[0]["extra_json"] == {"x": 1}
    finally:
        db._DB_CONFIG = None


def test_list_candidates_with_gen_filter(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {
                "candidate_id": 1,
                "gen": 0,
                "candidate_type": "couplet",
                "scheme": "COUPLET",
                "lines_json": '["line1", "line2"]',
                "fitness": 0.8,
                "scores_json": '{"e": 0.9}',
                "created_at": "2024-01-01",
            }
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        cands = db.list_candidates(1, gen=0)
        assert len(cands) == 1
        assert cands[0]["lines"] == ["line1", "line2"]
        assert cands[0]["scores"] == {"e": 0.9}
    finally:
        db._DB_CONFIG = None


def test_count_runs_with_and_without_filter(monkeypatch):
    cursor = _DummyCursor()
    cursor._row = (7,)  # COUNT(*) result
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        n = db.count_runs()
        assert n == 7
        n2 = db.count_runs(status_filter="completed")
        assert n2 == 7
    finally:
        db._DB_CONFIG = None


def test_load_archive_cells_with_stub(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {
                "cell_key": "0_0_0",
                "candidate_id": 1,
                "lines_json": '[["a","b"]]',
                "fitness": 0.8,
                "scores_json": None,
            }
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        cells = db.load_archive_cells(1)
        assert len(cells) == 1
        assert cells[0]["cell_key"] == "0_0_0"
        assert cells[0]["lines"] == [["a", "b"]]
        assert cells[0]["fitness"] == 0.8
    finally:
        db._DB_CONFIG = None


def test_list_runs_json_decode_error_sets_empty_config(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {
                "run_id": 1,
                "script_name": "s",
                "theme_keywords": "t",
                "config_json": "invalid",
                "status": "completed",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            }
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        runs = db.list_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["config_json"] == {}
    finally:
        db._DB_CONFIG = None


def test_refresh_run_derived_with_stub(monkeypatch):
    cursor = _DummyCursor()
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        db.refresh_run_derived(5)
        assert any("run_derived" in str(e[0]).lower() for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_insert_operator_event_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor.lastrowid = 99
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        oid = db.insert_operator_event(1, 0, "mutation", candidate_id=2, parents=[10], meta={"emitter": "x"})
        assert oid == 99
        assert any("operator_events" in str(e[0]).lower() for e in cursor.executed)
    finally:
        db._DB_CONFIG = None


def test_list_operator_mix_global_with_stub(monkeypatch):
    cursor = _DummyCursor(
        rows=[
            {"operator": "mutation", "cnt": 5},
            {"operator": "crossover", "cnt": 3},
        ]
    )
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        mix = db.list_operator_mix_global(limit_ops=10)
        assert mix == [{"operator": "mutation", "count": 5}, {"operator": "crossover", "count": 3}]
    finally:
        db._DB_CONFIG = None


def test_count_archive_cells_with_stub(monkeypatch):
    cursor = _DummyCursor()
    cursor._row = (42,)
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        assert db.count_archive_cells(7) == 42
    finally:
        db._DB_CONFIG = None


def test_execute_rollback_on_exception(monkeypatch):
    """_execute returns default and rollback is called when callback raises."""
    class FailingCursor(_DummyCursor):
        def execute(self, query, params=None):
            raise RuntimeError("db error")
    cursor = FailingCursor()
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        result = db.get_run(1)
        assert result is None
    finally:
        db._DB_CONFIG = None


def test_load_top_candidates_cross_run(monkeypatch):
    """load_top_candidates_cross_run returns parsed candidate dicts."""
    import json
    rows = [
        {
            "candidate_id": 10,
            "run_id": 5,
            "lines_json": json.dumps(["line a", "line b", "line c", "line d"]),
            "fitness": 0.85,
            "scores_json": json.dumps({"coherence": 0.9}),
        },
    ]
    cursor = _DummyCursor(rows=rows)
    _install_mysql_stub(monkeypatch, cursor)
    monkeypatch.setenv("RAPBOT_USE_DB", "1")
    db._DB_CONFIG = None
    try:
        result = db.load_top_candidates_cross_run(limit=10, min_fitness=0.3)
        assert len(result) == 1
        assert result[0]["lines"] == ["line a", "line b", "line c", "line d"]
        assert result[0]["fitness"] == 0.85
        assert result[0]["scores"]["coherence"] == 0.9
        assert result[0]["run_id"] == 5
    finally:
        db._DB_CONFIG = None

