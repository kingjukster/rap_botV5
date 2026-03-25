"""Tests for the SSE run stream endpoint."""

import pytest
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from webapp.api.routes import router as api_router


def _patch_for_sse(monkeypatch, run_data, gens_sequence):
    """
    Patch get_run and get_run_generations on the routes module.
    gens_sequence is a list: first call returns gens_sequence[0], second returns gens_sequence[1], etc.
    """
    import webapp.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "get_run", lambda rid: run_data)

    call_count = {"n": 0}
    def _fake_gens(rid):
        idx = min(call_count["n"], len(gens_sequence) - 1)
        call_count["n"] += 1
        return gens_sequence[idx]

    monkeypatch.setattr(routes_mod, "get_run_generations", _fake_gens)

    monkeypatch.setattr(routes_mod, "get_analysis_data", lambda: {
        "runs": {}, "fitness_trend": [], "config_stats": [],
        "best_fitness": None, "stagnation_runs": 0, "operator_mix": [],
    })


def test_sse_stream_not_found(monkeypatch):
    import webapp.api.routes as routes_mod
    monkeypatch.setattr(routes_mod, "get_run", lambda rid: None)
    monkeypatch.setattr(routes_mod, "get_run_generations", lambda rid: [])
    monkeypatch.setattr(routes_mod, "get_analysis_data", lambda: {
        "runs": {}, "fitness_trend": [], "config_stats": [],
        "best_fitness": None, "stagnation_runs": 0, "operator_mix": [],
    })

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    client = TestClient(app)

    try:
        with client.stream("GET", "/api/runs/999/stream") as resp:
            assert resp.status_code == 200
            text = ""
            for chunk in resp.iter_text():
                text += chunk
                if "error" in text:
                    break
            assert "Run not found" in text
    except ImportError:
        pytest.skip("sse-starlette not installed")


def test_sse_stream_completed_run(monkeypatch):
    run_data = {"run_id": 1, "status": "completed", "config_json": {}}
    gens = [{"gen": 0, "best_fitness": 0.5, "avg_fitness": 0.3}]
    _patch_for_sse(monkeypatch, run_data, [gens])

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    client = TestClient(app)

    try:
        with client.stream("GET", "/api/runs/1/stream") as resp:
            assert resp.status_code == 200
            text = ""
            for chunk in resp.iter_text():
                text += chunk
                if "done" in text:
                    break
            assert "generation" in text
            assert "done" in text
    except ImportError:
        pytest.skip("sse-starlette not installed")
