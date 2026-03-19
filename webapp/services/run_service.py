"""Run listing and detail service."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

db = None
DB_AVAILABLE = False

try:
    from evo_rhyme import db as _db
    db = _db
    DB_AVAILABLE = True
except ImportError:
    # Lightweight env (e.g. Docker): load only evo_rhyme.db to avoid pulling numpy/pandas/etc.
    _db_path = Path(__file__).resolve().parents[2] / "evo_rhyme" / "db.py"
    if _db_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("evo_rhyme.db", _db_path)
            db = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db)
            DB_AVAILABLE = True
        except Exception:
            pass


def list_runs(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """List runs from DB. Returns {runs: [...], total: int}."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return {"runs": [], "total": 0}

    runs = db.list_runs(limit=limit, offset=offset, status_filter=status_filter)
    total = db.count_runs(status_filter=status_filter)
    return {"runs": runs, "total": total}


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """Get run detail by id."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return None
    return db.get_run(run_id)


def get_run_generations(run_id: int) -> List[Dict[str, Any]]:
    """Get generations for a run."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_generations(run_id)


def get_run_candidates(run_id: int, gen: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get candidates for a run, optionally by generation."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_candidates(run_id, gen=gen)


def get_run_top_candidates(
    run_id: int,
    *,
    limit: int = 20,
    candidate_type: Optional[str] = None,
    scheme: Optional[str] = None,
    gen: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get top candidates for a run by fitness."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_top_candidates(run_id, limit=limit, candidate_type=candidate_type, scheme=scheme, gen=gen)


def get_run_archive(run_id: int) -> List[Dict[str, Any]]:
    """Get MAP-Elites archive cells for a run."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.load_archive_cells(run_id)


def get_candidate(candidate_id: int) -> Optional[Dict[str, Any]]:
    """Get a candidate by candidate_id."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return None
    return db.get_candidate(candidate_id)


def get_run_lineage(
    run_id: int,
    *,
    gen: Optional[int] = None,
    limit: int = 2000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get lineage edges for a run."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_lineage_edges(run_id, gen=gen, limit=limit, offset=offset)


def get_candidate_lineage(candidate_id: int) -> Dict[str, Any]:
    """Get immediate parent/child lineage edges for a candidate."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return {"parents": [], "children": []}
    return {
        "parents": db.list_candidate_parents(candidate_id),
        "children": db.list_candidate_children(candidate_id),
    }


def get_run_seeds(run_id: int, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """Get seed bank entries for a run."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_seed_bank(run_id, limit=limit, offset=offset)


def get_score_cache_recent(*, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """List recent score cache keys."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return []
    return db.list_score_cache_recent(limit=limit, offset=offset)


def check_db_connection() -> bool:
    """Return True if the database is enabled and a simple query succeeds."""
    if not DB_AVAILABLE or not db or not db.db_enabled():
        return False

    def _ping(conn):
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return True

    return db._execute(_ping, default=False) is True


def get_run_summary(run_id: int) -> Optional[Dict[str, Any]]:
    """Get run with last-gen best fitness and candidate count."""
    run = get_run(run_id)
    if not run:
        return None
    gens = get_run_generations(run_id)
    if gens:
        last = gens[-1]
        run["last_best_fitness"] = last.get("best_fitness")
        run["last_avg_fitness"] = last.get("avg_fitness")
        run["num_generations"] = len(gens)
    else:
        run["last_best_fitness"] = None
        run["last_avg_fitness"] = None
        run["num_generations"] = 0
    return run
