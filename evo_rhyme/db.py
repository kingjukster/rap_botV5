"""
MySQL database module for evo_rhyme run logging, score cache, archive, and lineage.

Loads config from os.environ; uses python-dotenv if available.
Uses connection pooling when enabled. All DB functions catch exceptions, log,
and return None or -1 on failure.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DB_CONFIG = None
_POOL_NAME = "rapbot_pool"
_POOL_SIZE = 5


def _get_config() -> Dict[str, Any]:
    global _DB_CONFIG
    if _DB_CONFIG is None:
        use_db = os.environ.get("RAPBOT_USE_DB", "").strip().lower()
        _DB_CONFIG = {
            "enabled": use_db in ("1", "true", "yes"),
            "host": os.environ.get("RAPBOT_DB_HOST", "localhost"),
            "port": int(os.environ.get("RAPBOT_DB_PORT", "3306")),
            "user": os.environ.get("RAPBOT_DB_USER", "root"),
            "password": os.environ.get("RAPBOT_DB_PASSWORD", ""),
            "database": os.environ.get("RAPBOT_DB_NAME", "rapbot"),
        }
    return _DB_CONFIG


def db_enabled() -> bool:
    """Return True if database logging is enabled via RAPBOT_USE_DB."""
    return _get_config()["enabled"]


@contextmanager
def connection():
    """
    Context manager yielding a MySQL connection. Uses connection pooling when enabled.
    Connection is returned to the pool (or closed) on exit.
    """
    conn = None
    if not db_enabled():
        yield None
        return
    try:
        import mysql.connector

        cfg = _get_config()
        conn = mysql.connector.connect(
            pool_name=_POOL_NAME,
            pool_size=_POOL_SIZE,
            pool_reset_session=True,
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset="utf8mb4",
        )
        yield conn
    except Exception as e:
        logger.warning("DB connection failed: %s", e)
        yield None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _execute(
    callback: Callable[[Any], Any],
    default: Any = None,
    *,
    commit: bool = False,
) -> Any:
    """Run callback(conn) with a connection. Handles errors and returns default on failure."""
    with connection() as conn:
        if conn is None:
            return default
        try:
            result = callback(conn)
            if commit:
                conn.commit()
            return result
        except Exception as e:
            logger.warning("DB operation failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass
            return default


def get_connection():
    """
    Return a MySQL connection or None if DB disabled or connection fails.
    Uses connection pooling. Caller must close() when done (returns to pool).
    Prefer using connection() context manager for automatic cleanup.
    """
    if not db_enabled():
        return None
    try:
        import mysql.connector

        cfg = _get_config()
        return mysql.connector.connect(
            pool_name=_POOL_NAME,
            pool_size=_POOL_SIZE,
            pool_reset_session=True,
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset="utf8mb4",
        )
    except Exception as e:
        logger.warning("DB connection failed: %s", e)
        return None


def score_cache_get(text_hash: str, candidate_type: str, scheme: str) -> Optional[Dict[str, Any]]:
    """
    Look up cached scores by text_hash, candidate_type, scheme.
    Returns dict with scores (or similar) or None.
    """

    def _run(conn: Any) -> Optional[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT scores_json FROM score_cache WHERE text_hash = %s AND candidate_type = %s AND scheme = %s",
                (text_hash, candidate_type, scheme),
            )
            row = cur.fetchone()
            if row and row.get("scores_json"):
                return json.loads(row["scores_json"])
            return None
        finally:
            cur.close()

    return _execute(_run, default=None)


def score_cache_put(
    text_hash: str, candidate_type: str, scheme: str, scores: Dict[str, Any]
) -> None:
    """Insert or replace cached scores."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO score_cache (text_hash, candidate_type, scheme, scores_json)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE scores_json = VALUES(scores_json)
                """,
                (text_hash, candidate_type, scheme, json.dumps(scores)),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def insert_run(script_name: str, theme_keywords: str, config_json: Dict[str, Any]) -> int:
    """
    Insert a new run and return run_id, or -1 on failure.
    theme_keywords: comma-separated or JSON string.
    config_json: dict (will be json.dumps'd).
    """

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO runs (script_name, theme_keywords, config_json, status) VALUES (%s, %s, %s, 'running')",
                (script_name, theme_keywords, json.dumps(config_json)),
            )
            run_id = cur.lastrowid
            return run_id if run_id else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def update_run_status(run_id: int, status: str) -> None:
    """Update run status (e.g. 'running', 'completed', 'failed')."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute("UPDATE runs SET status = %s WHERE run_id = %s", (status, run_id))
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def insert_generation(
    run_id: int,
    gen: int,
    best_fitness: float,
    avg_fitness: float,
    diversity: Optional[float],
    acceptance_rate: Optional[float],
    extra_json: Optional[Dict[str, Any]],
) -> None:
    """Insert a generation record for a run."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO generations (run_id, gen, best_fitness, avg_fitness, diversity, acceptance_rate, extra_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    gen,
                    best_fitness,
                    avg_fitness,
                    diversity,
                    acceptance_rate,
                    json.dumps(extra_json) if extra_json else None,
                ),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def insert_candidate(
    run_id: int,
    gen: int,
    candidate_type: str,
    scheme: str,
    lines_json: List[Any],
    fitness: float,
    scores_json: Optional[Dict[str, Any]],
) -> int:
    """
    Insert a candidate and return candidate_id, or -1 on failure.
    """

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO candidates (run_id, gen, candidate_type, scheme, lines_json, fitness, scores_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    gen,
                    candidate_type,
                    scheme,
                    json.dumps(lines_json),
                    fitness,
                    json.dumps(scores_json) if scores_json else None,
                ),
            )
            cid = cur.lastrowid
            return cid if cid else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def insert_lineage(child_id: int, parent_id: int, operation: str, gen: int) -> None:
    """Insert a lineage edge (child evolved from parent)."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO lineage (child_id, parent_id, operation, gen) VALUES (%s, %s, %s, %s)",
                (child_id, parent_id, operation, gen),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def upsert_archive_cell(
    run_id: int,
    cell_key: str,
    lines_json: List[Any],
    fitness: float,
    scores_json: Optional[Dict[str, Any]],
) -> None:
    """Insert or update an archive cell. Creates a candidate row if needed."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO candidates (run_id, gen, candidate_type, scheme, lines_json, fitness, scores_json)
                VALUES (%s, 0, 'verse4', 'AABB', %s, %s, %s)
                """,
                (
                    run_id,
                    json.dumps(lines_json),
                    fitness,
                    json.dumps(scores_json) if scores_json else None,
                ),
            )
            cid = cur.lastrowid
            if not cid:
                return
            cur.execute(
                """
                INSERT INTO archive_cells (run_id, cell_key, candidate_id, lines_json, fitness, scores_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE candidate_id=VALUES(candidate_id), lines_json=VALUES(lines_json),
                fitness=VALUES(fitness), scores_json=VALUES(scores_json)
                """,
                (
                    run_id,
                    cell_key,
                    cid,
                    json.dumps(lines_json),
                    fitness,
                    json.dumps(scores_json) if scores_json else None,
                ),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def upsert_archive_cells_batch(
    run_id: int,
    cells: List[Tuple[str, List[Any], float, Optional[Dict[str, Any]]]],
) -> None:
    """
    Batch insert/update archive cells in a single transaction.
    cells: list of (cell_key, lines_json, fitness, scores_json).
    Much more efficient than calling upsert_archive_cell for each cell.
    """

    if not cells:
        return

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            for cell_key, lines_json, fitness, scores_json in cells:
                cur.execute(
                    """
                    INSERT INTO candidates (run_id, gen, candidate_type, scheme, lines_json, fitness, scores_json)
                    VALUES (%s, 0, 'verse4', 'AABB', %s, %s, %s)
                    """,
                    (
                        run_id,
                        json.dumps(lines_json),
                        fitness,
                        json.dumps(scores_json) if scores_json else None,
                    ),
                )
                cid = cur.lastrowid
                if cid:
                    cur.execute(
                        """
                        INSERT INTO archive_cells (run_id, cell_key, candidate_id, lines_json, fitness, scores_json)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE candidate_id=VALUES(candidate_id), lines_json=VALUES(lines_json),
                        fitness=VALUES(fitness), scores_json=VALUES(scores_json)
                        """,
                        (
                            run_id,
                            cell_key,
                            cid,
                            json.dumps(lines_json),
                            fitness,
                            json.dumps(scores_json) if scores_json else None,
                        ),
                    )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def load_archive_cells(run_id: int) -> List[Dict[str, Any]]:
    """Load all archive cells for a run as list of dicts."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT cell_key, candidate_id, lines_json, fitness, scores_json FROM archive_cells WHERE run_id = %s",
                (run_id,),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d: Dict[str, Any] = {
                    "cell_key": r["cell_key"],
                    "candidate_id": r["candidate_id"],
                    "lines": json.loads(r["lines_json"]) if r.get("lines_json") else [],
                    "fitness": float(r["fitness"]) if r.get("fitness") is not None else 0.0,
                    "scores": json.loads(r["scores_json"]) if r.get("scores_json") else None,
                }
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])
