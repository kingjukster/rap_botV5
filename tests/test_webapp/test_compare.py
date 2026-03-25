"""Tests for compare service and API endpoint."""

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from webapp.api.routes import router as api_router


def _patch_compare(monkeypatch, result):
    """Patch the compare_runs function on the routes module."""
    import webapp.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "svc_compare_runs", lambda ids: result)


def _make_app():
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    return app


def test_compare_api_requires_two_ids(monkeypatch):
    _patch_compare(monkeypatch, {"runs": [], "generation_series": {}})

    import webapp.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "get_analysis_data", lambda: {"runs": {}, "fitness_trend": [], "config_stats": [], "best_fitness": None, "stagnation_runs": 0, "operator_mix": []})

    app = _make_app()
    client = TestClient(app)

    resp = client.get("/api/runs/compare?ids=1")
    assert resp.status_code == 400
    assert "at least 2" in resp.json()["detail"].lower()


def test_compare_api_returns_data(monkeypatch):
    payload = {
        "runs": [
            {"run_id": 1, "status": "completed"},
            {"run_id": 2, "status": "completed"},
        ],
        "generation_series": {
            "1": [{"gen": 0, "best_fitness": 0.5}],
            "2": [{"gen": 0, "best_fitness": 0.6}],
        },
    }
    _patch_compare(monkeypatch, payload)

    import webapp.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "get_analysis_data", lambda: {"runs": {}, "fitness_trend": [], "config_stats": [], "best_fitness": None, "stagnation_runs": 0, "operator_mix": []})

    app = _make_app()
    client = TestClient(app)

    resp = client.get("/api/runs/compare?ids=1,2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 2
    assert "1" in data["generation_series"]


def test_compare_service_unit():
    from webapp.services.compare_service import compare_runs

    # With DB off, returns empty runs
    result = compare_runs([1, 2])
    assert result["runs"] == []
    assert result["generation_series"] == {}
