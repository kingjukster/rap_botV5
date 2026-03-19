"""Tests for webapp.services.run_service: list_runs, get_run, get_run_summary, etc. Area 13."""

import pytest

from webapp.services import run_service
from webapp.services.run_service import (
    get_run,
    get_run_archive,
    get_run_candidates,
    get_run_top_candidates,
    get_run_generations,
    get_run_summary,
    get_candidate,
    get_candidate_lineage,
    get_run_lineage,
    get_run_seeds,
    get_score_cache_recent,
    list_runs,
)


def test_list_runs_db_disabled_returns_empty():
    # With RAPBOT_USE_DB=0 (conftest), service should return empty
    result = list_runs(limit=10, offset=0, status_filter=None)
    assert result == {"runs": [], "total": 0}


def test_list_runs_no_db_available_returns_empty(monkeypatch):
    """When DB_AVAILABLE is False (import failed), list_runs returns empty."""
    monkeypatch.setattr(run_service, "DB_AVAILABLE", False)
    result = list_runs(limit=10, offset=0, status_filter=None)
    assert result == {"runs": [], "total": 0}


def test_get_run_db_disabled_returns_none():
    assert get_run(1) is None


def test_get_run_generations_db_disabled_returns_empty():
    assert get_run_generations(1) == []


def test_get_run_candidates_db_disabled_returns_empty():
    assert get_run_candidates(1, gen=None) == []
    assert get_run_candidates(1, gen=0) == []


def test_get_run_archive_db_disabled_returns_empty():
    assert get_run_archive(1) == []


def test_get_run_summary_db_disabled_returns_none():
    assert get_run_summary(1) is None


def test_candidate_and_lineage_and_seeds_and_cache_db_disabled_returns_empty():
    assert get_candidate(1) is None
    assert get_candidate_lineage(1) == {"parents": [], "children": []}
    assert get_run_lineage(1) == []
    assert get_run_seeds(1) == []
    assert get_score_cache_recent() == []
    assert get_run_top_candidates(1) == []


def test_list_runs_db_enabled_uses_db(monkeypatch):
    monkeypatch.setattr(run_service, "db", run_service.db)
    monkeypatch.setattr(run_service.db, "db_enabled", lambda: True)
    monkeypatch.setattr(run_service.db, "list_runs", lambda limit, offset, status_filter: [{"run_id": 1}])
    monkeypatch.setattr(run_service.db, "count_runs", lambda status_filter: 1)
    result = list_runs(limit=10, offset=0, status_filter=None)
    assert result["runs"] == [{"run_id": 1}]
    assert result["total"] == 1


def test_get_run_db_enabled_returns_run(monkeypatch):
    monkeypatch.setattr(run_service.db, "db_enabled", lambda: True)
    monkeypatch.setattr(run_service.db, "get_run", lambda run_id: {"run_id": run_id, "status": "completed"})
    run = get_run(1)
    assert run is not None
    assert run["run_id"] == 1
    assert run["status"] == "completed"


def test_get_run_summary_db_enabled_includes_generations(monkeypatch):
    monkeypatch.setattr(run_service.db, "db_enabled", lambda: True)
    monkeypatch.setattr(run_service.db, "get_run", lambda run_id: {"run_id": run_id, "status": "completed"})
    monkeypatch.setattr(run_service.db, "list_generations", lambda run_id: [{"gen": 0, "best_fitness": 0.9}])
    summary = get_run_summary(1)
    assert summary is not None
    assert summary.get("last_best_fitness") == 0.9
    assert summary.get("num_generations") == 1


def test_get_run_summary_db_enabled_no_generations(monkeypatch):
    """get_run_summary when run exists but has no generations (else branch)."""
    monkeypatch.setattr(run_service.db, "db_enabled", lambda: True)
    monkeypatch.setattr(run_service.db, "get_run", lambda run_id: {"run_id": run_id, "status": "completed"})
    monkeypatch.setattr(run_service.db, "list_generations", lambda run_id: [])
    summary = get_run_summary(1)
    assert summary is not None
    assert summary.get("last_best_fitness") is None
    assert summary.get("last_avg_fitness") is None
    assert summary.get("num_generations") == 0


def test_get_run_archive_db_enabled_returns_cells(monkeypatch):
    monkeypatch.setattr(run_service.db, "db_enabled", lambda: True)
    monkeypatch.setattr(run_service.db, "load_archive_cells", lambda run_id: [{"cell_key": "0_0", "fitness": 0.8}])
    cells = get_run_archive(1)
    assert len(cells) == 1
    assert cells[0]["cell_key"] == "0_0"
