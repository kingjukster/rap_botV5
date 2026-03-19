"""Run listing and detail service."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

db = None
DB_AVAILABLE = False
logger = logging.getLogger(__name__)

try:
    from evo_rhyme import db as _db
    db = _db
    DB_AVAILABLE = True
except ImportError as exc:
    logger.debug("Primary evo_rhyme.db import failed; trying lightweight fallback: %s", exc)
    # Lightweight env (e.g. Docker): load only evo_rhyme.db to avoid pulling numpy/pandas/etc.
    _db_path = Path(__file__).resolve().parents[2] / "evo_rhyme" / "db.py"
    if _db_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("evo_rhyme.db", _db_path)
            db = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db)
            DB_AVAILABLE = True
        except Exception as fallback_exc:
            logger.warning("Fallback evo_rhyme.db import failed: %s", fallback_exc)


def _db_ready() -> bool:
    """
    Determine DB readiness dynamically.

    Some lightweight/container imports can leave DB_AVAILABLE stale even when
    a working db module is loaded; check capabilities on `db` directly.
    """
    if db is None or not hasattr(db, "db_enabled"):
        return False
    try:
        return bool(db.db_enabled())
    except Exception:
        return False


def list_runs(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """List runs from DB. Returns {runs: [...], total: int}."""
    if not _db_ready():
        return {"runs": [], "total": 0}

    runs = db.list_runs(limit=limit, offset=offset, status_filter=status_filter)
    total = db.count_runs(status_filter=status_filter)
    return {"runs": runs, "total": total}


def get_run_counts_by_status() -> Dict[str, Any]:
    """Return total run count and counts per status (running, completed, failed) across all runs."""
    if not _db_ready():
        return {"total": 0, "by_status": {"running": 0, "completed": 0, "failed": 0}}
    fn = getattr(db, "count_runs_by_status", None)
    if fn is None:
        total = db.count_runs()
        return {"total": total, "by_status": {"running": 0, "completed": 0, "failed": 0}}
    return fn()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """Get run detail by id."""
    if not _db_ready():
        return None
    return db.get_run(run_id)


def get_run_generations(run_id: int) -> List[Dict[str, Any]]:
    """Get generations for a run."""
    if not _db_ready():
        return []
    return db.list_generations(run_id)


def get_run_candidates(run_id: int, gen: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get candidates for a run, optionally by generation."""
    if not _db_ready():
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
    if not _db_ready():
        return []
    return db.list_top_candidates(run_id, limit=limit, candidate_type=candidate_type, scheme=scheme, gen=gen)


def get_run_archive(run_id: int) -> List[Dict[str, Any]]:
    """Get MAP-Elites archive cells for a run."""
    if not _db_ready():
        return []
    loader = getattr(db, "load_archive_cells", None)
    if loader is None:
        return []
    return loader(run_id)


def get_candidate(candidate_id: int) -> Optional[Dict[str, Any]]:
    """Get a candidate by candidate_id."""
    if not _db_ready():
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
    if not _db_ready():
        return []
    fn = getattr(db, "list_lineage_edges", None)
    if fn is None:
        return []
    return fn(run_id, gen=gen, limit=limit, offset=offset)


def get_candidate_lineage(candidate_id: int) -> Dict[str, Any]:
    """Get immediate parent/child lineage edges for a candidate."""
    if not _db_ready():
        return {"parents": [], "children": []}
    list_parents = getattr(db, "list_candidate_parents", None)
    list_children = getattr(db, "list_candidate_children", None)
    return {
        "parents": list_parents(candidate_id) if list_parents else [],
        "children": list_children(candidate_id) if list_children else [],
    }


def get_run_seeds(run_id: int, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """Get seed bank entries for a run."""
    if not _db_ready():
        return []
    fn = getattr(db, "list_seed_bank", None)
    if fn is None:
        return []
    return fn(run_id, limit=limit, offset=offset)


def get_score_cache_recent(*, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """List recent score cache keys."""
    if not _db_ready():
        return []
    fn = getattr(db, "list_score_cache_recent", None)
    if fn is None:
        return []
    return fn(limit=limit, offset=offset)


def execute_sql(sql: str, max_rows: int = 1000) -> Dict[str, Any]:
    """Execute a read-only SQL query. Returns {columns, rows, row_count} or {error}."""
    if not _db_ready():
        return {"error": "Database is not available. Set RAPBOT_USE_DB=1."}
    fn = getattr(db, "execute_readonly_sql", None)
    if fn is None:
        return {"error": "SQL execution is not available."}
    return fn(sql, max_rows=max_rows)


def mark_stale_runs_failed(minutes_idle: int = 30) -> int:
    """Mark runs as failed if they have been 'running' with no activity for minutes_idle. Returns count updated."""
    if not _db_ready():
        return 0
    fn = getattr(db, "mark_stale_runs_failed", None)
    if fn is None:
        return 0
    return fn(minutes_idle=minutes_idle)


def check_db_connection() -> bool:
    """Return True if the database is enabled and a simple query succeeds."""
    if not _db_ready() or not hasattr(db, "_execute"):
        return False

    def _ping(conn):
        cur = conn.cursor()
        cur.execute("SELECT 1")
        # mysql-connector requires consuming result sets before cursor close.
        cur.fetchone()
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


# ---------------------------------------------------------------------------
# Experiments (control experiment runner)
# ---------------------------------------------------------------------------


def list_experiments(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List experiments, most recent first."""
    if not _db_ready():
        return []
    return db.list_experiments(limit=limit, offset=offset)


def get_experiment(experiment_id: int) -> Optional[Dict[str, Any]]:
    """Get experiment by id."""
    if not _db_ready():
        return None
    return db.get_experiment(experiment_id)


def list_experiment_arms(experiment_id: int) -> List[Dict[str, Any]]:
    """List arms for an experiment."""
    if not _db_ready():
        return []
    return db.list_experiment_arms(experiment_id)


def list_experiment_runs(
    experiment_id: int,
    arm_id: Optional[int] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List runs for an experiment, optionally filtered by arm."""
    if not _db_ready():
        return []
    return db.list_runs_for_experiment(experiment_id, arm_id=arm_id, limit=limit, offset=offset)


def get_control_impact_report(
    experiment_id: int,
    arm_id: Optional[int] = None,
    aggregation_mode: str = "best",
    top_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Build control impact report for an experiment (mean diffs, CI, correlations)."""
    if not _db_ready():
        return None
    run_list = db.list_runs_for_experiment(experiment_id, arm_id=arm_id, limit=500)
    if not run_list:
        return {"runs": 0, "by_control": {}, "correlations": {}}
    run_ids = [r["run_id"] for r in run_list]
    try:
        from evo_rhyme.experiment_metrics import aggregate_outcome
    except ImportError:
        return {"runs": len(run_ids), "error": "experiment_metrics not available"}
    rows = []
    for run_id in run_ids:
        run = db.get_run(run_id)
        if not run:
            continue
        candidates = db.list_candidates(run_id, gen=None, limit=500)
        cands = [{"fitness": c.get("fitness"), "scores": c.get("scores") or c.get("scores_json")} for c in candidates]
        agg = aggregate_outcome(cands, mode=aggregation_mode, top_k=top_k, score_key="fitness", scores_key="scores")
        config = run.get("config_json") or {}
        if isinstance(config, str):
            try:
                import json
                config = json.loads(config)
            except Exception:
                config = {}
        rows.append({"run_id": run_id, "controls": config, "fitness": agg["fitness"], "fitness_vector": agg["fitness_vector"]})
    try:
        from evo_rhyme.experiment_analysis import analyze_control_impact
        return analyze_control_impact(rows, bootstrap_n=200)
    except ImportError:
        return {"runs": len(rows), "error": "experiment_analysis not available"}
