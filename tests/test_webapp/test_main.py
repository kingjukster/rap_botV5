import os

from fastapi.testclient import TestClient

from webapp.main import app


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_db_returns_503_when_db_unavailable():
    """When DB is disabled or unreachable, /health/db returns 503."""
    client = TestClient(app)
    resp = client.get("/health/db")
    # In test env RAPBOT_USE_DB is typically 0, so DB is unavailable
    assert resp.status_code == 503
    data = resp.json()
    assert data.get("db") == "unavailable"
    assert data.get("status") == "degraded"


def test_health_db_returns_200_when_db_connected(monkeypatch):
    """When DB check succeeds, /health/db returns 200."""
    from webapp.main import check_db_connection

    monkeypatch.setattr("webapp.main.check_db_connection", lambda: True)
    client = TestClient(app)
    resp = client.get("/health/db")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "connected"}


def test_global_exception_handler_returns_error_page(monkeypatch):
    """Unhandled exceptions result in 500 and the error template."""
    from webapp.routes import pages as pages_mod

    def _raise(limit=None, offset=None, status_filter=None):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pages_mod, "list_runs", _raise)
    # Do not re-raise server exceptions so we get the 500 response body from our handler
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/")
    assert resp.status_code == 500
    assert "Something went wrong" in resp.text
    assert "Back to Dashboard" in resp.text


def test_app_includes_api_and_pages_routes():
    client = TestClient(app)
    # These routes are provided by the included routers; a 200/404 distinction
    # is less important than verifying FastAPI resolves them without error.
    resp_dashboard = client.get("/")
    assert resp_dashboard.status_code in (200, 302)

    resp_api = client.get("/api/runs")
    # When DB is disabled the service layer may return an empty payload,
    # but the route itself should still exist.
    assert resp_api.status_code != 404

