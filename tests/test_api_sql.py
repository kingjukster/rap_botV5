"""Tests for /api/sql endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from webapp.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_sql_valid_select_returns_200(client):
    """Valid SELECT returns 200 with columns, rows, row_count."""
    with patch("webapp.api.routes.execute_sql") as m:
        m.return_value = {
            "columns": ["run_id", "status"],
            "rows": [{"run_id": 1, "status": "completed"}],
            "row_count": 1,
        }
        r = client.post("/api/sql", json={"sql": "SELECT * FROM runs LIMIT 1"})
    assert r.status_code == 200
    data = r.json()
    assert "columns" in data
    assert "rows" in data
    assert "row_count" in data
    assert data["row_count"] == 1
    assert data["columns"] == ["run_id", "status"]


def test_sql_rejected_query_returns_400(client):
    """Rejected query (non-SELECT, multi-statement) returns 400 with error."""
    with patch("webapp.api.routes.execute_sql") as m:
        m.return_value = {"error": "Only SELECT queries are allowed."}
        r = client.post("/api/sql", json={"sql": "DROP TABLE runs"})
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert "SELECT" in data["error"]


def test_sql_db_unavailable_returns_503(client):
    """When DB is unavailable, returns 503."""
    with patch("webapp.api.routes.execute_sql") as m:
        m.return_value = {"error": "Database is not available. Set RAPBOT_USE_DB=1."}
        r = client.post("/api/sql", json={"sql": "SELECT 1"})
    assert r.status_code == 503
    data = r.json()
    assert "error" in data
    assert "Database" in data["error"]


def test_sql_multi_statement_rejected(client):
    """Multi-statement query returns 400."""
    with patch("webapp.api.routes.execute_sql") as m:
        m.return_value = {"error": "Multiple statements are not allowed."}
        r = client.post("/api/sql", json={"sql": "SELECT 1; DROP TABLE runs"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_sql_accepts_max_rows(client):
    """API passes max_rows to service."""
    with patch("webapp.api.routes.execute_sql") as m:
        m.return_value = {"columns": [], "rows": [], "row_count": 0}
        r = client.post("/api/sql", json={"sql": "SELECT 1", "max_rows": 500})
    assert r.status_code == 200
    m.assert_called_once_with(sql="SELECT 1", max_rows=500)
