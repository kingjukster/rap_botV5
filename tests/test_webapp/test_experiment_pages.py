"""Tests for experiment list and detail pages + summary API."""

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from webapp.api.routes import router as api_router
from webapp.routes.pages import router as pages_router


def _patch_routes(monkeypatch):
    """Patch service functions to avoid needing a real DB."""
    import webapp.api.routes as routes_mod
    import webapp.routes.pages as pages_mod

    monkeypatch.setattr(routes_mod, "list_experiments", lambda limit=50, offset=0: [
        {"experiment_id": 1, "name": "test_exp", "description": "desc", "mode": "grid", "created_at": "2025-01-01"},
    ])
    monkeypatch.setattr(routes_mod, "get_experiment", lambda eid: (
        {"experiment_id": eid, "name": "test_exp", "description": "desc", "mode": "grid", "created_at": "2025-01-01"}
        if eid != 404 else None
    ))
    monkeypatch.setattr(routes_mod, "list_experiment_arms", lambda eid: [
        {"arm_id": 1, "experiment_id": eid, "arm_name": "baseline"},
    ])
    monkeypatch.setattr(routes_mod, "get_experiment_summary", lambda eid: {
        "arms": [{"arm_id": 1, "arm_name": "baseline", "run_count": 5, "mean_fitness": 0.75, "median_fitness": 0.76,
                   "std_fitness": 0.05, "q1": 0.70, "q3": 0.80, "min_fitness": 0.60, "max_fitness": 0.90,
                   "fitnesses": [0.6, 0.7, 0.76, 0.8, 0.9], "is_best": True}]
    })
    monkeypatch.setattr(routes_mod, "get_analysis_data", lambda: {
        "runs": {}, "fitness_trend": [], "config_stats": [],
        "best_fitness": None, "stagnation_runs": 0, "operator_mix": [],
    })

    monkeypatch.setattr(pages_mod, "list_experiments", lambda limit=100, offset=0: [
        {"experiment_id": 1, "name": "test_exp", "description": "desc", "mode": "grid", "created_at": "2025-01-01"},
    ])
    monkeypatch.setattr(pages_mod, "get_experiment", lambda eid: (
        {"experiment_id": eid, "name": "test_exp", "description": "desc", "mode": "grid", "created_at": "2025-01-01"}
        if eid != 404 else None
    ))
    monkeypatch.setattr(pages_mod, "list_experiment_arms", lambda eid: [
        {"arm_id": 1, "experiment_id": eid, "arm_name": "baseline"},
    ])


def test_experiments_list_api(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    client = TestClient(app)

    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    assert resp.json()[0]["name"] == "test_exp"


def test_experiment_summary_api(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    client = TestClient(app)

    resp = client.get("/api/experiments/1/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["arms"]) == 1
    assert data["arms"][0]["is_best"] is True


def test_experiment_detail_404(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    client = TestClient(app)

    resp = client.get("/api/experiments/404")
    assert resp.status_code == 404


def test_experiments_page_renders(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(pages_router)
    client = TestClient(app)

    resp = client.get("/experiments")
    assert resp.status_code == 200
    assert "test_exp" in resp.text


def test_experiment_detail_page_renders(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(pages_router)
    client = TestClient(app)

    resp = client.get("/experiments/1")
    assert resp.status_code == 200
    assert "test_exp" in resp.text


def test_experiment_detail_page_missing_redirects(monkeypatch):
    _patch_routes(monkeypatch)
    app = FastAPI()
    app.include_router(pages_router)
    client = TestClient(app, follow_redirects=False)

    resp = client.get("/experiments/404")
    assert resp.status_code == 302
