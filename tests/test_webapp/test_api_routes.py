from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from webapp.api.routes import router as api_router


class _SvcStub:
    def __init__(self):
        self.list_runs_called_with: Dict[str, Any] = {}
        self.get_run_called_with: List[int] = []
        self.get_generations_called_with: List[int] = []
        self.get_candidates_called_with: List[Dict[str, Any]] = []
        self.get_archive_called_with: List[int] = []

    def list_runs(self, limit: int, offset: int, status_filter: Optional[str]):
        self.list_runs_called_with = {
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
        }
        return {"runs": [], "total": 0}

    def get_run(self, run_id: int):
        self.get_run_called_with.append(run_id)
        if run_id == 404:
            return None
        return {"run_id": run_id, "status": "completed"}

    def get_run_generations(self, run_id: int):
        self.get_generations_called_with.append(run_id)
        return [{"run_id": run_id, "gen": 0}]

    def get_run_candidates(self, run_id: int, gen: Optional[int]):
        self.get_candidates_called_with.append({"run_id": run_id, "gen": gen})
        return [{"run_id": run_id, "gen": gen, "candidate_id": 1}]

    def get_run_top_candidates(self, run_id: int, *, limit: int = 20, candidate_type=None, scheme=None, gen=None):
        return [{"run_id": run_id, "candidate_id": 9, "fitness": 0.99}][:limit]

    def get_run_archive(self, run_id: int):
        self.get_archive_called_with.append(run_id)
        return [{"run_id": run_id, "cell_key": "0"}]

    def get_candidate(self, candidate_id: int):
        if candidate_id == 404:
            return None
        return {"candidate_id": candidate_id, "run_id": 1, "gen": 2}

    def get_candidate_lineage(self, candidate_id: int):
        return {"parents": [{"parent_id": 10, "gen": 1}], "children": [{"child_id": 11, "gen": 3}]}

    def get_run_lineage(self, run_id: int, *, gen=None, limit=2000, offset=0):
        return [{"child_id": 1, "parent_id": 2, "operation": "mutate", "gen": gen or 0}]

    def get_run_seeds(self, run_id: int, *, limit=200, offset=0):
        return [{"id": 1, "run_id": run_id, "seed_key": "k"}]

    def get_score_cache_recent(self, *, limit=200, offset=0):
        return [{"text_hash": "abc", "candidate_type": "verse4", "scheme": "AABB"}]


def _create_app_with_stub(monkeypatch: Any) -> tuple[FastAPI, _SvcStub]:
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    stub = _SvcStub()

    # The router imported the service functions into module-level names,
    # so we patch those symbols directly.
    import webapp.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "svc_list_runs", stub.list_runs)
    monkeypatch.setattr(routes_mod, "get_run", stub.get_run)
    monkeypatch.setattr(routes_mod, "get_run_generations", stub.get_run_generations)
    monkeypatch.setattr(routes_mod, "get_run_candidates", stub.get_run_candidates)
    monkeypatch.setattr(routes_mod, "get_run_top_candidates", stub.get_run_top_candidates)
    monkeypatch.setattr(routes_mod, "get_run_archive", stub.get_run_archive)
    monkeypatch.setattr(routes_mod, "get_candidate", stub.get_candidate)
    monkeypatch.setattr(routes_mod, "get_candidate_lineage", stub.get_candidate_lineage)
    monkeypatch.setattr(routes_mod, "get_run_lineage", stub.get_run_lineage)
    monkeypatch.setattr(routes_mod, "get_run_seeds", stub.get_run_seeds)
    monkeypatch.setattr(routes_mod, "get_score_cache_recent", stub.get_score_cache_recent)

    return app, stub


def test_api_list_runs_uses_service_and_returns_payload(monkeypatch):
    app, stub = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp = client.get("/api/runs?limit=10&offset=5&status=completed")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"runs": [], "total": 0}
    assert stub.list_runs_called_with == {
        "limit": 10,
        "offset": 5,
        "status_filter": "completed",
    }


def test_api_get_run_returns_404_when_missing(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp = client.get("/api/runs/404")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


def test_api_get_run_generations_and_candidates_and_archive(monkeypatch):
    app, stub = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp_gens = client.get("/api/runs/1/generations")
    assert resp_gens.status_code == 200
    assert resp_gens.json() == [{"run_id": 1, "gen": 0}]

    resp_cands = client.get("/api/runs/1/candidates?gen=2")
    assert resp_cands.status_code == 200
    assert resp_cands.json() == [{"run_id": 1, "gen": 2, "candidate_id": 1}]

    resp_archive = client.get("/api/runs/1/archive")
    assert resp_archive.status_code == 200
    assert resp_archive.json() == [{"run_id": 1, "cell_key": "0"}]

    assert stub.get_generations_called_with == [1]
    assert stub.get_candidates_called_with == [{"run_id": 1, "gen": 2}]
    assert stub.get_archive_called_with == [1]

    resp_top = client.get("/api/runs/1/top-candidates?limit=5")
    assert resp_top.status_code == 200
    assert resp_top.json()[0]["candidate_id"] == 9


def test_api_candidate_and_lineage_and_seeds_and_score_cache(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    resp_candidate = client.get("/api/candidates/1")
    assert resp_candidate.status_code == 200
    assert resp_candidate.json()["candidate_id"] == 1

    resp_missing = client.get("/api/candidates/404")
    assert resp_missing.status_code == 404
    assert resp_missing.json()["detail"] == "Candidate not found"

    resp_lineage = client.get("/api/candidates/1/lineage")
    assert resp_lineage.status_code == 200
    data = resp_lineage.json()
    assert "parents" in data and "children" in data

    resp_run_lineage = client.get("/api/runs/1/lineage?gen=3&limit=10&offset=0")
    assert resp_run_lineage.status_code == 200
    assert isinstance(resp_run_lineage.json(), list)

    resp_seeds = client.get("/api/runs/1/seeds")
    assert resp_seeds.status_code == 200
    assert resp_seeds.json()[0]["run_id"] == 1

    resp_cache = client.get("/api/score-cache/recent")
    assert resp_cache.status_code == 200
    assert resp_cache.json()[0]["text_hash"] == "abc"


def test_api_get_run_progress_returns_status_and_generations(monkeypatch):
    import webapp.api.routes as routes_mod

    app, stub = _create_app_with_stub(monkeypatch)
    client = TestClient(app)

    def get_run_with_config(run_id: int):
        if run_id == 1:
            return {"run_id": 1, "status": "running", "config_json": {"generations": 60}}
        return stub.get_run(run_id)

    def get_gens_count(run_id: int):
        if run_id == 1:
            return [{"gen": i} for i in range(5)]
        return stub.get_run_generations(run_id)

    monkeypatch.setattr(routes_mod, "get_run", get_run_with_config)
    monkeypatch.setattr(routes_mod, "get_run_generations", get_gens_count)

    resp = client.get("/api/runs/1/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert data["status"] == "running"
    assert data["current_generation"] == 5
    assert data["total_generations"] == 60


def test_api_get_run_progress_returns_404_when_run_missing(monkeypatch):
    app, _ = _create_app_with_stub(monkeypatch)
    client = TestClient(app)
    resp = client.get("/api/runs/404/progress")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"

