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
from dataclasses import dataclass
import gzip
import hashlib
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


def list_runs(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List runs with metadata. Returns list of dicts with run_id, script_name, theme_keywords, config_json, status, created_at, updated_at."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            if status_filter:
                cur.execute(
                    """
                    SELECT run_id, script_name, theme_keywords, config_json, status, created_at, updated_at
                    FROM runs
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (status_filter, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT run_id, script_name, theme_keywords, config_json, status, created_at, updated_at
                    FROM runs
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d: Dict[str, Any] = dict(r)
                if d.get("config_json"):
                    try:
                        d["config_json"] = json.loads(d["config_json"]) if isinstance(d["config_json"], str) else d["config_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["config_json"] = {}
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """Get a single run by run_id."""

    def _run(conn: Any) -> Optional[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT run_id, script_name, theme_keywords, config_json, status, created_at, updated_at FROM runs WHERE run_id = %s",
                (run_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            d = dict(r)
            if d.get("config_json"):
                try:
                    d["config_json"] = json.loads(d["config_json"]) if isinstance(d["config_json"], str) else d["config_json"]
                except (json.JSONDecodeError, TypeError):
                    d["config_json"] = {}
            return d
        finally:
            cur.close()

    return _execute(_run, default=None)


def list_generations(run_id: int) -> List[Dict[str, Any]]:
    """List generations for a run, ordered by gen ascending."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT gen, best_fitness, avg_fitness, diversity, acceptance_rate, extra_json, created_at
                FROM generations
                WHERE run_id = %s
                ORDER BY gen ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("extra_json"):
                    try:
                        d["extra_json"] = json.loads(d["extra_json"]) if isinstance(d["extra_json"], str) else d["extra_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["extra_json"] = None
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_candidates(
    run_id: int,
    gen: Optional[int] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """List candidates for a run, optionally filtered by gen."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            if gen is not None:
                cur.execute(
                    """
                    SELECT candidate_id, gen, candidate_type, scheme, lines_json, fitness, scores_json, created_at
                    FROM candidates
                    WHERE run_id = %s AND gen = %s
                    ORDER BY fitness DESC
                    LIMIT %s
                    """,
                    (run_id, gen, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT candidate_id, gen, candidate_type, scheme, lines_json, fitness, scores_json, created_at
                    FROM candidates
                    WHERE run_id = %s
                    ORDER BY gen DESC, fitness DESC
                    LIMIT %s
                    """,
                    (run_id, limit),
                )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("lines_json"):
                    try:
                        d["lines"] = json.loads(d["lines_json"]) if isinstance(d["lines_json"], str) else d["lines_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["lines"] = []
                else:
                    d["lines"] = []
                if d.get("scores_json"):
                    try:
                        d["scores"] = json.loads(d["scores_json"]) if isinstance(d["scores_json"], str) else d["scores_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["scores"] = None
                else:
                    d["scores"] = None
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_top_candidates(
    run_id: int,
    *,
    limit: int = 20,
    candidate_type: Optional[str] = None,
    scheme: Optional[str] = None,
    gen: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List top candidates by fitness for a run, optionally filtered."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            where = ["run_id = %s"]
            params: list[Any] = [run_id]
            if gen is not None:
                where.append("gen = %s")
                params.append(gen)
            if candidate_type is not None:
                where.append("candidate_type = %s")
                params.append(candidate_type)
            if scheme is not None:
                where.append("scheme = %s")
                params.append(scheme)

            sql = f"""
                SELECT candidate_id, gen, candidate_type, scheme, lines_json, fitness, scores_json, created_at
                FROM candidates
                WHERE {' AND '.join(where)}
                ORDER BY fitness DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("lines_json"):
                    try:
                        d["lines"] = json.loads(d["lines_json"]) if isinstance(d["lines_json"], str) else d["lines_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["lines"] = []
                else:
                    d["lines"] = []
                if d.get("scores_json"):
                    try:
                        d["scores"] = json.loads(d["scores_json"]) if isinstance(d["scores_json"], str) else d["scores_json"]
                    except (json.JSONDecodeError, TypeError):
                        d["scores"] = None
                else:
                    d["scores"] = None
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def count_runs(status_filter: Optional[str] = None) -> int:
    """Return total count of runs, optionally filtered by status."""

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            if status_filter:
                cur.execute("SELECT COUNT(*) FROM runs WHERE status = %s", (status_filter,))
            else:
                cur.execute("SELECT COUNT(*) FROM runs")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()

    return _execute(_run, default=0)


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: int
    sha256: str
    kind: str
    run_id: Optional[int]
    run_tag: Optional[str]
    rel_path: Optional[str]
    content_len: int
    created_at: Any


def insert_artifact(
    *,
    kind: str,
    content_bytes: bytes,
    sha256: Optional[str] = None,
    run_id: Optional[int] = None,
    run_tag: Optional[str] = None,
    rel_path: Optional[str] = None,
) -> int:
    """
    Insert a gzipped artifact blob into the DB.
    Returns artifact_id (or existing artifact_id if sha256 already present), or -1 on failure.
    """
    if sha256 is None:
        sha256 = hashlib.sha256(content_bytes).hexdigest()
    content_gzip = gzip.compress(content_bytes, compresslevel=9)
    content_len = len(content_bytes)

    def _run(conn: Any) -> int:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("SELECT artifact_id FROM artifacts WHERE sha256 = %s", (sha256,))
            row = cur.fetchone()
            if row and row.get("artifact_id"):
                return int(row["artifact_id"])

            cur.execute(
                """
                INSERT INTO artifacts (run_id, run_tag, kind, rel_path, sha256, content_gzip, content_len)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, run_tag, kind, rel_path, sha256, content_gzip, content_len),
            )
            aid = cur.lastrowid
            return int(aid) if aid else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def get_artifact_bytes_by_sha256(sha256: str) -> Optional[bytes]:
    """Fetch and gunzip artifact content by sha256."""

    def _run(conn: Any) -> Optional[bytes]:
        cur = conn.cursor()
        try:
            cur.execute("SELECT content_gzip FROM artifacts WHERE sha256 = %s", (sha256,))
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            try:
                return gzip.decompress(row[0])
            except Exception:
                return None
        finally:
            cur.close()

    return _execute(_run, default=None)


def list_artifacts(*, limit: int = 200, offset: int = 0, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """List artifact metadata (no blob payload)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            if kind:
                cur.execute(
                    """
                    SELECT artifact_id, sha256, kind, run_id, run_tag, rel_path, content_len, created_at
                    FROM artifacts
                    WHERE kind = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (kind, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT artifact_id, sha256, kind, run_id, run_tag, rel_path, content_len, created_at
                    FROM artifacts
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])


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


def get_candidate(candidate_id: int) -> Optional[Dict[str, Any]]:
    """Get a single candidate by candidate_id."""

    def _run(conn: Any) -> Optional[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT candidate_id, run_id, gen, candidate_type, scheme, lines_json, fitness, scores_json, created_at
                FROM candidates
                WHERE candidate_id = %s
                """,
                (candidate_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            d = dict(r)
            if d.get("lines_json"):
                try:
                    d["lines"] = json.loads(d["lines_json"]) if isinstance(d["lines_json"], str) else d["lines_json"]
                except (json.JSONDecodeError, TypeError):
                    d["lines"] = []
            else:
                d["lines"] = []
            if d.get("scores_json"):
                try:
                    d["scores"] = json.loads(d["scores_json"]) if isinstance(d["scores_json"], str) else d["scores_json"]
                except (json.JSONDecodeError, TypeError):
                    d["scores"] = None
            else:
                d["scores"] = None
            return d
        finally:
            cur.close()

    return _execute(_run, default=None)


def list_lineage_edges(
    run_id: int,
    *,
    gen: Optional[int] = None,
    limit: int = 2000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List lineage edges for a run. Joins to candidates to filter by run_id.

    Returns rows like:
      {child_id, parent_id, operation, gen, created_at, child_fitness, parent_fitness}
    """

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            if gen is None:
                cur.execute(
                    """
                    SELECT
                        l.child_id, l.parent_id, l.operation, l.gen, l.created_at,
                        cc.fitness AS child_fitness,
                        pc.fitness AS parent_fitness
                    FROM lineage l
                    JOIN candidates cc ON cc.candidate_id = l.child_id
                    JOIN candidates pc ON pc.candidate_id = l.parent_id
                    WHERE cc.run_id = %s
                    ORDER BY l.gen DESC, l.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (run_id, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        l.child_id, l.parent_id, l.operation, l.gen, l.created_at,
                        cc.fitness AS child_fitness,
                        pc.fitness AS parent_fitness
                    FROM lineage l
                    JOIN candidates cc ON cc.candidate_id = l.child_id
                    JOIN candidates pc ON pc.candidate_id = l.parent_id
                    WHERE cc.run_id = %s AND l.gen = %s
                    ORDER BY l.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (run_id, gen, limit, offset),
                )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_candidate_parents(candidate_id: int, *, limit: int = 200) -> List[Dict[str, Any]]:
    """List parent edges for a candidate (incoming lineage)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT parent_id, operation, gen, created_at
                FROM lineage
                WHERE child_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (candidate_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_candidate_children(candidate_id: int, *, limit: int = 200) -> List[Dict[str, Any]]:
    """List child edges for a candidate (outgoing lineage)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT child_id, operation, gen, created_at
                FROM lineage
                WHERE parent_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (candidate_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_seed_bank(run_id: Optional[int] = None, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """List seed bank entries, optionally filtered to a run_id (or NULL when run_id is None)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            if run_id is None:
                cur.execute(
                    """
                    SELECT id, run_id, seed_key, seed_data, created_at
                    FROM seed_bank
                    WHERE run_id IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT id, run_id, seed_key, seed_data, created_at
                    FROM seed_bank
                    WHERE run_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (run_id, limit, offset),
                )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("seed_data"):
                    try:
                        d["seed_data"] = json.loads(d["seed_data"]) if isinstance(d["seed_data"], str) else d["seed_data"]
                    except (json.JSONDecodeError, TypeError):
                        d["seed_data"] = None
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_score_cache_recent(*, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """List most recently updated score_cache rows (for debugging/visibility)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT text_hash, candidate_type, scheme, created_at, updated_at
                FROM score_cache
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])
