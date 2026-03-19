from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates
from datetime import datetime

from webapp.routes.pages import router as pages_router


class _RunServiceStub:
    def __init__(self):
        self.list_runs_called = False

    def list_runs(self, limit: int, offset: int, status_filter=None):
        self.list_runs_called = True
        return {
            "runs": [
                {"run_id": 1, "status": "completed"},
                {"run_id": 2, "status": "running"},
            ],
            "total": 2,
        }

    def get_run_summary(self, run_id: int):
        if run_id == 404:
            return None
        return {
            "run_id": run_id,
            "status": "completed",
            "last_best_fitness": 0.9,
            "num_generations": 3,
            "config_json": {"theme": "pressure", "population": 100} if run_id == 1 else (None if run_id == 2 else {}),
        }

    def get_run_generations(self, run_id: int):
        return [
            {
                "run_id": run_id,
                "gen": 0,
                "best_fitness": 0.7,
                "avg_fitness": 0.5,
                "diversity": 0.3,
                "acceptance_rate": 0.12,
                "extra_json": {"occupied_niches": 10},
                "created_at": datetime(2024, 1, 1, 0, 0, 0),
            },
            {
                "run_id": run_id,
                "gen": 1,
                "best_fitness": 0.85,
                "avg_fitness": 0.6,
                "diversity": 0.33,
                "acceptance_rate": 0.2,
                "extra_json": {"occupied_niches": 25},
                "created_at": datetime(2024, 1, 1, 0, 1, 0),
            },
        ]

    def get_run_candidates(self, run_id: int, gen=None):
        return [
            {
                "run_id": run_id,
                "gen": 0,
                "candidate_id": 1,
                "candidate_type": "mutation",
                "scheme": "AABB",
                "fitness": 0.9,
                "scores": {"flow": 0.8, "rhyme": 0.95},
                "lines": ["line one", "line two"],
                "created_at": datetime(2024, 1, 1, 0, 0, 30),
            },
        ]

    def get_run_top_candidates(self, run_id: int, *, limit: int = 20, candidate_type=None, scheme=None, gen=None):
        return [
            {
                "run_id": run_id,
                "gen": 1,
                "candidate_id": 999,
                "candidate_type": "verse",
                "scheme": "AABB",
                "fitness": 0.99,
                "scores": {"flow": 0.9},
                "lines": ["top line one", "top line two"],
                "created_at": datetime(2024, 1, 1, 0, 2, 0),
            },
        ][:limit]

    def get_run_archive(self, run_id: int):
        return [
            {"run_id": run_id, "cell_key": "1_2_0", "fitness": 0.5, "lines": ["a", "b"], "scores": {"flow": 0.6, "rhyme": 0.8}},
        ]

    def get_candidate(self, candidate_id: int):
        if candidate_id == 404:
            return None
        return {"candidate_id": candidate_id, "run_id": 1, "gen": 0, "candidate_type": "mutation", "scheme": "AABB", "fitness": 0.9, "scores": {"flow": 0.8}, "lines": ["x"]}

    def get_candidate_lineage(self, candidate_id: int):
        return {"parents": [{"parent_id": 10, "operation": "seed", "gen": 0}], "children": []}

    def get_run_lineage(self, run_id: int, *, gen=None, limit=2000, offset=0):
        return [{"child_id": 1, "parent_id": 2, "operation": "mutate", "gen": 0, "child_fitness": 0.9, "parent_fitness": 0.7, "created_at": datetime(2024, 1, 1, 0, 0, 0)}]

    def get_run_seeds(self, run_id: int, *, limit=200, offset=0):
        return [{"id": 1, "run_id": run_id, "seed_key": "k", "seed_data": {"foo": "bar"}, "created_at": datetime(2024, 1, 1, 0, 0, 0)}]

    def get_score_cache_recent(self, *, limit=200, offset=0):
        return [{"text_hash": "abc", "candidate_type": "verse4", "scheme": "AABB", "updated_at": datetime(2024, 1, 1, 0, 0, 0)}]


def _create_app_with_stub(monkeypatch):
    app = FastAPI()
    app.include_router(pages_router)

    from webapp import routes as routes_pkg
    from webapp.routes import pages

    stub = _RunServiceStub()
    monkeypatch.setattr(pages, "list_runs", stub.list_runs)
    monkeypatch.setattr(pages, "get_run_summary", stub.get_run_summary)
    monkeypatch.setattr(pages, "get_run_generations", stub.get_run_generations)
    monkeypatch.setattr(pages, "get_run_candidates", stub.get_run_candidates)
    monkeypatch.setattr(pages, "get_run_top_candidates", stub.get_run_top_candidates)
    monkeypatch.setattr(pages, "get_run_archive", stub.get_run_archive)
    monkeypatch.setattr(pages, "get_candidate", stub.get_candidate)
    monkeypatch.setattr(pages, "get_candidate_lineage", stub.get_candidate_lineage)
    monkeypatch.setattr(pages, "get_run_lineage", stub.get_run_lineage)
    monkeypatch.setattr(pages, "get_run_seeds", stub.get_run_seeds)
    monkeypatch.setattr(pages, "get_score_cache_recent", stub.get_score_cache_recent)

    return app, stub


def test_dashboard_renders_template_with_runs(monkeypatch):
    app, stub = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    # Jinja2TemplateResponse renders HTML; just assert basic structure.
    assert "<html" in resp.text.lower()
    assert "runs" in resp.text.lower()


def test_dashboard_includes_summary_stats(monkeypatch):
    """Dashboard shows summary cards (total and by status) when runs exist."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Total runs" in resp.text
    assert "Running" in resp.text
    assert "Completed" in resp.text


def test_run_detail_redirects_when_missing(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp = client.get("/runs/404", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"


def test_archive_view_renders_when_run_exists(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp = client.get("/runs/1/archive")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_archive_view_redirects_when_run_missing(monkeypatch):
    """archive_view redirects to / when run_id has no summary."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/404/archive", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"


def test_about_page_returns_200(monkeypatch):
    """About page returns 200 and contains expected copy."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/about")
    assert resp.status_code == 200
    assert "Rap Bot" in resp.text
    assert "RAPBOT_USE_DB" in resp.text
    assert "run_verse_qd" in resp.text or "evolution" in resp.text.lower()


def test_run_detail_includes_config_section(monkeypatch):
    """Run detail shows Run config when config_json is present."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1")
    assert resp.status_code == 200
    assert "Run config" in resp.text or "config" in resp.text.lower()
    assert "theme" in resp.text or "population" in resp.text


def test_run_detail_handles_null_config_json(monkeypatch):
    """Run detail renders even when config_json is null."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/2")
    assert resp.status_code == 200
    assert "Run config" not in resp.text


def test_run_detail_includes_generation_metadata(monkeypatch):
    """Run detail shows generation fitness and Gen."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1")
    assert resp.status_code == 200
    assert "Gen" in resp.text
    assert "0.7" in resp.text or "0.85" in resp.text


def test_run_detail_candidates_show_scores_when_present(monkeypatch):
    """Run detail shows score breakdown when candidate has scores."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1")
    assert resp.status_code == 200
    assert "Score breakdown" in resp.text or "flow" in resp.text or "rhyme" in resp.text


def test_run_generations_page_returns_200_when_run_exists(monkeypatch):
    """Generations page returns 200 and shows generation data."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/generations")
    assert resp.status_code == 200
    assert "Generations" in resp.text
    assert "Gen" in resp.text


def test_run_generations_page_redirects_when_run_missing(monkeypatch):
    """Generations page redirects to / when run has no summary."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/404/generations", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"


def test_archive_page_includes_score_breakdown_for_cells(monkeypatch):
    """Archive page shows Score breakdown when cells have scores."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/archive")
    assert resp.status_code == 200
    assert "Score breakdown" in resp.text
    assert "flow" in resp.text or "rhyme" in resp.text


def test_archive_page_includes_heatmap_or_dimension_controls(monkeypatch):
    """Archive page includes heatmap dimension controls."""
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/archive")
    assert resp.status_code == 200
    assert "Heatmap" in resp.text or "Dimension" in resp.text or "Dim " in resp.text


def test_lineage_page_returns_200(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/lineage")
    assert resp.status_code == 200
    assert "Lineage" in resp.text


def test_candidate_detail_page_returns_200(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/candidates/1")
    assert resp.status_code == 200
    assert "Candidate" in resp.text


def test_seeds_page_returns_200(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/runs/1/seeds")
    assert resp.status_code == 200
    assert "Seed bank" in resp.text


def test_score_cache_page_returns_200(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/score-cache")
    assert resp.status_code == 200
    assert "Score cache" in resp.text

