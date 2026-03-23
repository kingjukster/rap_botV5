"""Evolution analysis service: run stats, fitness trend, config dominance, stagnation."""

from __future__ import annotations

import importlib.util
import datetime as _dt
import decimal
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

db = None
try:
    from evo_rhyme import db as _db
    db = _db
except ImportError as exc:
    logger.debug("Primary evo_rhyme.db import failed; trying lightweight fallback: %s", exc)
    _db_path = Path(__file__).resolve().parents[2] / "evo_rhyme" / "db.py"
    if _db_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("evo_rhyme.db", _db_path)
            _db_mod = importlib.util.module_from_spec(spec)
            # Ensure evo_rhyme parent exists so db.py dataclasses can resolve __module__
            import sys
            if "evo_rhyme" not in sys.modules:
                import types
                sys.modules["evo_rhyme"] = types.ModuleType("evo_rhyme")
            sys.modules["evo_rhyme.db"] = _db_mod
            spec.loader.exec_module(_db_mod)
            db = _db_mod
        except Exception as fallback_exc:
            logger.warning("Fallback evo_rhyme.db import failed: %s", fallback_exc)


def _db_ready() -> bool:
    if db is None or not hasattr(db, "db_enabled"):
        return False
    try:
        return bool(db.db_enabled())
    except Exception:
        return False


def _make_row_safe(val: Any) -> Any:
    """Convert DB values to JSON-serializable types."""
    if val is None:
        return None
    if isinstance(val, (_dt.datetime, _dt.date)):
        return val.isoformat()
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return val


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _make_row_safe(v) for k, v in row.items()}


def _empty_analysis() -> Dict[str, Any]:
    return {
        "runs": {"total": 0, "completed": 0, "failed": 0, "running": 0},
        "fitness_trend": [],
        "config_stats": [],
        "best_fitness": None,
        "stagnation_runs": 0,
    }


def get_analysis_data() -> Dict[str, Any]:
    """
    Return evolution analysis: run stats, fitness trend, config dominance, stagnation.
    Uses real DB data when available; returns empty structure when DB disabled.
    """
    if not _db_ready():
        return _empty_analysis()

    run_stats = _fetch_run_stats()
    fitness_trend = _fetch_fitness_trend()
    config_stats = _fetch_config_stats()
    best_fitness, stagnation_runs = _compute_stagnation()

    return {
        "runs": run_stats,
        "fitness_trend": fitness_trend,
        "config_stats": config_stats,
        "best_fitness": best_fitness,
        "stagnation_runs": stagnation_runs,
    }


def _fetch_run_stats() -> Dict[str, Any]:
    def _run(conn: Any) -> Dict[str, Any]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(status = 'completed'), 0) AS completed,
                    COALESCE(SUM(status = 'failed'), 0) AS failed,
                    COALESCE(SUM(status = 'running'), 0) AS running
                FROM runs
                """
            )
            row = cur.fetchone()
            if not row:
                return {"total": 0, "completed": 0, "failed": 0, "running": 0}
            return {
                "total": int(row.get("total") or 0),
                "completed": int(row.get("completed") or 0),
                "failed": int(row.get("failed") or 0),
                "running": int(row.get("running") or 0),
            }
        finally:
            cur.close()

    result = db._execute(_run, default={"total": 0, "completed": 0, "failed": 0, "running": 0})
    return result


def _fetch_fitness_trend() -> List[Dict[str, Any]]:
    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT r.run_id, r.created_at, g.best_fitness
                FROM runs r
                JOIN (
                    SELECT run_id, MAX(gen) AS max_gen
                    FROM generations
                    GROUP BY run_id
                ) mg ON r.run_id = mg.run_id
                JOIN generations g ON g.run_id = mg.run_id AND g.gen = mg.max_gen
                ORDER BY r.created_at DESC
                LIMIT 100
                """
            )
            rows = cur.fetchall()
            return [_sanitize_row(dict(r)) for r in rows]
        finally:
            cur.close()

    return db._execute(_run, default=[])


def _fetch_config_stats() -> List[Dict[str, Any]]:
    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT
                    JSON_UNQUOTE(JSON_EXTRACT(r.config_json, '$.arm')) AS arm,
                    COUNT(*) AS count,
                    AVG(g.best_fitness) AS avg_fitness
                FROM runs r
                JOIN (
                    SELECT run_id, MAX(best_fitness) AS best_fitness
                    FROM generations
                    GROUP BY run_id
                ) g ON r.run_id = g.run_id
                WHERE JSON_EXTRACT(r.config_json, '$.arm') IS NOT NULL
                GROUP BY JSON_UNQUOTE(JSON_EXTRACT(r.config_json, '$.arm'))
                ORDER BY count DESC
                LIMIT 10
                """
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                d = _sanitize_row(dict(r))
                if d.get("avg_fitness") is not None:
                    d["avg_fitness"] = float(d["avg_fitness"])
                d["count"] = int(d.get("count") or 0)
                out.append(d)
            return out
        finally:
            cur.close()

    return db._execute(_run, default=[])


def _compute_stagnation() -> tuple[float | None, int]:
    def _run(conn: Any) -> tuple[float | None, int]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT MAX(best_fitness) AS best FROM generations"
            )
            row = cur.fetchone()
            best_overall = row.get("best") if row else None
            if best_overall is not None:
                best_overall = float(best_overall)

            cur.execute(
                """
                SELECT r.run_id, r.created_at, mg.best
                FROM runs r
                JOIN (
                    SELECT run_id, MAX(best_fitness) AS best
                    FROM generations
                    GROUP BY run_id
                ) mg ON r.run_id = mg.run_id
                ORDER BY r.created_at DESC
                LIMIT 50
                """
            )
            recent = cur.fetchall()

            stagnation_runs = 0
            for r in recent:
                best = r.get("best")
                if best is not None:
                    best = float(best)
                if best_overall is None:
                    break
                if best is None or best < best_overall:
                    stagnation_runs += 1
                else:
                    break

            return best_overall, stagnation_runs
        finally:
            cur.close()

    result = db._execute(_run, default=(None, 0))
    if result is None:
        return None, 0
    return result
