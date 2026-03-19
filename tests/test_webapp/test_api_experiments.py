"""Tests for webapp API experiment endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from webapp.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_experiments_list_returns_200(client):
    with patch("webapp.api.routes.list_experiments") as m:
        m.return_value = [{"experiment_id": 1, "name": "e1", "description": None, "mode": None, "created_at": "2024-01-01"}]
        r = client.get("/api/experiments?limit=10&offset=0")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "experiment_id" in data[0]
        assert "name" in data[0]


def test_experiments_get_404(client):
    with patch("webapp.api.routes.get_experiment") as m:
        m.return_value = None
        r = client.get("/api/experiments/999")
    assert r.status_code == 404


def test_experiments_get_200(client):
    with patch("webapp.api.routes.get_experiment") as m:
        m.return_value = {"experiment_id": 1, "name": "e1", "description": "d", "mode": "m", "created_at": "2024-01-01"}
        r = client.get("/api/experiments/1")
    assert r.status_code == 200
    assert r.json()["name"] == "e1"


def test_experiments_arms_returns_list(client):
    with patch("webapp.api.routes.list_experiment_arms") as m:
        m.return_value = [{"arm_id": 1, "arm_name": "arm_a", "control_snapshot": {"elites": 5}}]
        r = client.get("/api/experiments/1/arms")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_experiments_impact_returns_report(client):
    with patch("webapp.api.routes.get_control_impact_report") as m:
        m.return_value = {"runs": 5, "by_control": {}, "correlations": {}}
        r = client.get("/api/experiments/1/impact")
    assert r.status_code == 200
    assert r.json()["runs"] == 5
