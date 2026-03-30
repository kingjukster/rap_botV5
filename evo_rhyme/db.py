"""
MySQL database module for evo_rhyme run logging, score cache, archive, and lineage.

Loads config from os.environ; uses python-dotenv if available.
Uses connection pooling when enabled. All DB functions catch exceptions, log,
and return None or -1 on failure.
"""

from __future__ import annotations

import decimal
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


_ARCHIVE_CELLS_NORMALIZED = False


def _ensure_archive_cells_normalized(conn: Any) -> None:
    """
    Drop legacy archive_cells payload columns if present (lines_json, fitness, scores_json).
    Idempotent; safe across concurrent processes (unknown-column errors ignored).
    """
    global _ARCHIVE_CELLS_NORMALIZED
    if _ARCHIVE_CELLS_NORMALIZED:
        return
    cur = conn.cursor()
    try:
        for col in ("lines_json", "fitness", "scores_json"):
            try:
                cur.execute(f"ALTER TABLE archive_cells DROP COLUMN `{col}`")
            except Exception as e:
                err = str(e).lower()
                if "unknown column" in err or "check that column" in err or "1091" in err:
                    pass
                else:
                    raise
    finally:
        cur.close()
    _ARCHIVE_CELLS_NORMALIZED = True


def _ensure_run_derived_table(conn: Any) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS run_derived (
                run_id INT NOT NULL PRIMARY KEY,
                final_best_fitness DOUBLE DEFAULT NULL,
                final_avg_fitness DOUBLE DEFAULT NULL,
                max_diversity_or_coverage DOUBLE DEFAULT NULL,
                total_generations INT DEFAULT NULL,
                total_candidates INT DEFAULT NULL,
                last_occupied_niches INT DEFAULT NULL,
                archive_cell_count INT DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    finally:
        cur.close()


def _ensure_operator_events_table(conn: Any) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id INT NOT NULL,
                gen INT NOT NULL,
                candidate_id INT DEFAULT NULL,
                operator VARCHAR(64) NOT NULL,
                parents_json JSON DEFAULT NULL,
                meta_json JSON DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                INDEX idx_run_gen (run_id, gen),
                INDEX idx_run_operator (run_id, operator)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    finally:
        cur.close()


def ensure_experiment_schema() -> None:
    """
    Ensure experiment tables/columns exist (idempotent).
    Safe to call repeatedly; used by experiment runner paths.
    """

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description VARCHAR(1024) DEFAULT NULL,
                    mode VARCHAR(64) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_name (name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_arms (
                    arm_id INT AUTO_INCREMENT PRIMARY KEY,
                    experiment_id INT NOT NULL,
                    arm_name VARCHAR(255) NOT NULL,
                    control_snapshot JSON DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    INDEX idx_experiment_id (experiment_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            # runs.experiment_id / runs.arm_id (ignore duplicate-column errors)
            for col in ("experiment_id", "arm_id"):
                try:
                    cur.execute(f"ALTER TABLE runs ADD COLUMN {col} INT NULL DEFAULT NULL")
                except Exception as e:
                    if "Duplicate column" in str(e):
                        pass
                    else:
                        raise
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


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


def execute_readonly_sql(sql: str, max_rows: int = 1000) -> Dict[str, Any]:
    """
    Execute a read-only SQL query. Only SELECT is allowed.
    Returns {columns, rows, row_count} or {error}.
    """
    import datetime as _dt

    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return {"error": "Multiple statements are not allowed."}
    if not stripped.lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed."}

    logger.info(
        "SQL executed: %s",
        (stripped[:200] + "...") if len(stripped) > 200 else stripped,
    )

    def _make_row_safe(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (_dt.datetime, _dt.date)):
            return val.isoformat()
        if isinstance(val, (decimal.Decimal,)):
            return float(val)
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8", errors="replace")
        return val

    def _run(conn: Any) -> Dict[str, Any]:
        cur = conn.cursor(dictionary=True)
        try:
            try:
                cur.execute("SET SESSION max_execution_time = 5000")
            except Exception:
                pass
            cur.execute(stripped)
            columns = list(cur.column_names) if cur.column_names else []
            rows_raw = cur.fetchmany(max_rows)
            rows = []
            for r in rows_raw:
                rows.append({k: _make_row_safe(v) for k, v in r.items()})
            return {"columns": columns, "rows": rows, "row_count": len(rows)}
        finally:
            cur.close()

    result = _execute(_run, default=None)
    if result is None:
        return {"error": "Database operation failed. Check logs for details."}
    return result


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


def insert_run(
    script_name: str,
    theme_keywords: str,
    config_json: Dict[str, Any],
    *,
    experiment_id: Optional[int] = None,
    arm_id: Optional[int] = None,
) -> int:
    """
    Insert a new run and return run_id, or -1 on failure.
    theme_keywords: comma-separated or JSON string.
    config_json: dict (will be json.dumps'd). Include full control_snapshot for experiment analysis.
    experiment_id, arm_id: optional linkage to experiment/arm when run by control experiment runner.
    """

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            # Prefer extended schema (experiment_id, arm_id) when columns exist
            try:
                cur.execute(
                    """
                    INSERT INTO runs (script_name, theme_keywords, config_json, status, experiment_id, arm_id)
                    VALUES (%s, %s, %s, 'running', %s, %s)
                    """,
                    (script_name, theme_keywords, json.dumps(config_json), experiment_id, arm_id),
                )
            except Exception as e:
                if "Unknown column" in str(e):
                    cur.execute(
                        "INSERT INTO runs (script_name, theme_keywords, config_json, status) VALUES (%s, %s, %s, 'running')",
                        (script_name, theme_keywords, json.dumps(config_json)),
                    )
                else:
                    raise
            run_id = cur.lastrowid
            return run_id if run_id else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def insert_experiment(name: str, description: Optional[str] = None, mode: Optional[str] = None) -> int:
    """Insert an experiment and return experiment_id, or -1 on failure."""
    ensure_experiment_schema()

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO experiments (name, description, mode) VALUES (%s, %s, %s)",
                (name, description or None, mode or None),
            )
            eid = cur.lastrowid
            return int(eid) if eid else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def insert_experiment_arm(
    experiment_id: int,
    arm_name: str,
    control_snapshot: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert an experiment arm and return arm_id, or -1 on failure."""
    ensure_experiment_schema()

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO experiment_arms (experiment_id, arm_name, control_snapshot) VALUES (%s, %s, %s)",
                (experiment_id, arm_name, json.dumps(control_snapshot) if control_snapshot else None),
            )
            aid = cur.lastrowid
            return int(aid) if aid else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def list_experiments(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List experiments, most recent first."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT experiment_id, name, description, mode, created_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])


def get_experiment(experiment_id: int) -> Optional[Dict[str, Any]]:
    """Get a single experiment by id."""

    def _run(conn: Any) -> Optional[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT experiment_id, name, description, mode, created_at FROM experiments WHERE experiment_id = %s",
                (experiment_id,),
            )
            r = cur.fetchone()
            return dict(r) if r else None
        finally:
            cur.close()

    return _execute(_run, default=None)


def list_experiment_arms(experiment_id: int) -> List[Dict[str, Any]]:
    """List arms for an experiment."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT arm_id, experiment_id, arm_name, control_snapshot, created_at
                FROM experiment_arms
                WHERE experiment_id = %s
                ORDER BY arm_id ASC
                """,
                (experiment_id,),
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                d = dict(r)
                if d.get("control_snapshot") and isinstance(d["control_snapshot"], str):
                    try:
                        d["control_snapshot"] = json.loads(d["control_snapshot"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_runs_for_experiment(
    experiment_id: int,
    arm_id: Optional[int] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List runs for an experiment, optionally filtered by arm_id. Requires runs.experiment_id column."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            try:
                if arm_id is not None:
                    cur.execute(
                        """
                        SELECT run_id, script_name, theme_keywords, config_json, status, experiment_id, arm_id, created_at, updated_at
                        FROM runs
                        WHERE experiment_id = %s AND arm_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (experiment_id, arm_id, limit, offset),
                    )
                else:
                    cur.execute(
                        """
                        SELECT run_id, script_name, theme_keywords, config_json, status, experiment_id, arm_id, created_at, updated_at
                        FROM runs
                        WHERE experiment_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (experiment_id, limit, offset),
                    )
            except Exception as e:
                if "Unknown column" in str(e):
                    # Fallback when experiment_id/arm_id columns don't exist
                    return []
                raise
            rows = cur.fetchall()
            out = []
            for r in rows:
                d = dict(r)
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


def refresh_run_derived(run_id: int) -> None:
    """Recompute denormalized aggregates from generations / candidates / archive_cells."""

    def _run(conn: Any) -> None:
        _ensure_run_derived_table(conn)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO run_derived (
                    run_id, final_best_fitness, final_avg_fitness, max_diversity_or_coverage,
                    total_generations, total_candidates, last_occupied_niches, archive_cell_count
                )
                SELECT
                    %s AS run_id,
                    (SELECT MAX(g.best_fitness) FROM generations g WHERE g.run_id = %s),
                    (SELECT g2.avg_fitness FROM generations g2 WHERE g2.run_id = %s ORDER BY g2.gen DESC LIMIT 1),
                    (SELECT MAX(g3.diversity) FROM generations g3 WHERE g3.run_id = %s),
                    (SELECT COUNT(*) FROM generations gx WHERE gx.run_id = %s),
                    (SELECT COUNT(*) FROM candidates cx WHERE cx.run_id = %s),
                    (
                        SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(g4.extra_json, '$.occupied_niches')) AS SIGNED)
                        FROM generations g4
                        WHERE g4.run_id = %s
                        ORDER BY g4.gen DESC LIMIT 1
                    ),
                    (SELECT COUNT(*) FROM archive_cells ac WHERE ac.run_id = %s)
                ON DUPLICATE KEY UPDATE
                    final_best_fitness = VALUES(final_best_fitness),
                    final_avg_fitness = VALUES(final_avg_fitness),
                    max_diversity_or_coverage = VALUES(max_diversity_or_coverage),
                    total_generations = VALUES(total_generations),
                    total_candidates = VALUES(total_candidates),
                    last_occupied_niches = VALUES(last_occupied_niches),
                    archive_cell_count = VALUES(archive_cell_count)
                """,
                (run_id, run_id, run_id, run_id, run_id, run_id, run_id, run_id),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def get_run_derived(run_id: int) -> Optional[Dict[str, Any]]:
    """Return run_derived row or None."""

    def _run(conn: Any) -> Optional[Dict[str, Any]]:
        _ensure_run_derived_table(conn)
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT run_id, final_best_fitness, final_avg_fitness, max_diversity_or_coverage,
                       total_generations, total_candidates, last_occupied_niches, archive_cell_count, updated_at
                FROM run_derived WHERE run_id = %s
                """,
                (run_id,),
            )
            r = cur.fetchone()
            return dict(r) if r else None
        finally:
            cur.close()

    return _execute(_run, default=None)


def count_archive_cells(run_id: int) -> int:
    """Return number of MAP-Elites cells stored for a run."""

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) AS c FROM archive_cells WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            cur.close()

    return _execute(_run, default=0) or 0


def insert_operator_event(
    run_id: int,
    gen: int,
    operator: str,
    *,
    candidate_id: Optional[int] = None,
    parents: Optional[List[int]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Append one operator trace row. Returns insert id or -1 on failure."""

    def _run(conn: Any) -> int:
        _ensure_operator_events_table(conn)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO operator_events (run_id, gen, candidate_id, operator, parents_json, meta_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    gen,
                    candidate_id,
                    operator,
                    json.dumps(parents) if parents else None,
                    json.dumps(meta) if meta else None,
                ),
            )
            rid = cur.lastrowid
            return int(rid) if rid else -1
        finally:
            cur.close()

    return _execute(_run, default=-1, commit=True)


def list_operator_events_for_run(
    run_id: int,
    *,
    limit: int = 2000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List operator events for a run (most recent first)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        _ensure_operator_events_table(conn)
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT id, run_id, gen, candidate_id, operator, parents_json, meta_json, created_at
                FROM operator_events
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s OFFSET %s
                """,
                (run_id, limit, offset),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                for key in ("parents_json", "meta_json"):
                    if d.get(key) and isinstance(d[key], str):
                        try:
                            d[key] = json.loads(d[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                out.append(d)
            return out
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_operator_mix_global(*, limit_ops: int = 15) -> List[Dict[str, Any]]:
    """Counts of operator labels across all events (for dashboard)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        _ensure_operator_events_table(conn)
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT operator, COUNT(*) AS cnt
                FROM operator_events
                GROUP BY operator
                ORDER BY cnt DESC
                LIMIT %s
                """,
                (limit_ops,),
            )
            return [
                {"operator": r["operator"], "count": int(r["cnt"] or 0)}
                for r in cur.fetchall()
            ]
        finally:
            cur.close()

    return _execute(_run, default=[])


def update_run_status(run_id: int, status: str, failure_reason: Optional[str] = None) -> None:
    """Update run status (e.g. 'running', 'completed', 'failed'). Optionally store failure_reason."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            _ensure_failure_reason_column(conn)
            if failure_reason is not None and status == "failed":
                cur.execute(
                    "UPDATE runs SET status = %s, failure_reason = %s WHERE run_id = %s",
                    (status, failure_reason[:4096] if len(failure_reason) > 4096 else failure_reason, run_id),
                )
            else:
                cur.execute("UPDATE runs SET status = %s WHERE run_id = %s", (status, run_id))
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)
    if status in ("completed", "failed"):
        try:
            refresh_run_derived(run_id)
        except Exception as e:
            logger.warning("refresh_run_derived after status=%s failed: %s", status, e)


def update_run_config(run_id: int, config_json: Dict[str, Any]) -> None:
    """Update config_json for a run."""

    def _run(conn: Any) -> None:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE runs SET config_json = %s WHERE run_id = %s",
                (json.dumps(config_json), run_id),
            )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def _ensure_failure_reason_column(conn: Any) -> None:
    """Ensure runs.failure_reason column exists (idempotent)."""
    cur = conn.cursor()
    try:
        cur.execute(
            "ALTER TABLE runs ADD COLUMN failure_reason VARCHAR(4096) DEFAULT NULL"
        )
    except Exception as e:
        if "Duplicate column" in str(e):
            pass
        else:
            raise
    finally:
        cur.close()


def mark_stale_runs_failed(minutes_idle: int = 30) -> int:
    """
    Mark runs as 'failed' if they have status='running' and no activity for at least minutes_idle.
    Activity = run.created_at (if no generations) or max(generations.created_at).
    Returns the number of runs updated.
    """

    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE runs r
                LEFT JOIN (
                    SELECT run_id, MAX(created_at) AS last_gen_at
                    FROM generations
                    GROUP BY run_id
                ) g ON r.run_id = g.run_id
                SET r.status = 'failed'
                WHERE r.status = 'running'
                AND COALESCE(g.last_gen_at, r.created_at) < DATE_SUB(NOW(), INTERVAL %s MINUTE)
                """,
                (minutes_idle,),
            )
            return cur.rowcount
        finally:
            cur.close()

    return _execute(_run, default=0, commit=True) or 0


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
    """Insert or update an archive cell. Creates a candidate row; cell references candidate (no duplicate JSON on archive_cells)."""

    def _run(conn: Any) -> None:
        _ensure_archive_cells_normalized(conn)
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
                INSERT INTO archive_cells (run_id, cell_key, candidate_id)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE candidate_id=VALUES(candidate_id)
                """,
                (run_id, cell_key, cid),
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
        _ensure_archive_cells_normalized(conn)
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
                        INSERT INTO archive_cells (run_id, cell_key, candidate_id)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE candidate_id=VALUES(candidate_id)
                        """,
                        (run_id, cell_key, cid),
                    )
        finally:
            cur.close()

    _execute(_run, default=None, commit=True)


def list_runs(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List runs with metadata. Returns list of dicts with run_id, script_name, theme_keywords, config_json, status, created_at, updated_at, failure_reason."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        _ensure_failure_reason_column(conn)
        _ensure_run_derived_table(conn)
        cur = conn.cursor(dictionary=True)
        try:
            derived_cols = """
                , d.final_best_fitness AS drv_final_best_fitness,
                  d.final_avg_fitness AS drv_final_avg_fitness,
                  d.max_diversity_or_coverage AS drv_max_diversity_or_coverage,
                  d.total_generations AS drv_total_generations,
                  d.total_candidates AS drv_total_candidates,
                  d.last_occupied_niches AS drv_last_occupied_niches,
                  d.archive_cell_count AS drv_archive_cell_count
            """
            if status_filter:
                cur.execute(
                    f"""
                    SELECT r.run_id, r.script_name, r.theme_keywords, r.config_json, r.status,
                           r.created_at, r.updated_at, r.failure_reason
                           {derived_cols}
                    FROM runs r
                    LEFT JOIN run_derived d ON d.run_id = r.run_id
                    WHERE r.status = %s
                    ORDER BY r.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (status_filter, limit, offset),
                )
            else:
                cur.execute(
                    f"""
                    SELECT r.run_id, r.script_name, r.theme_keywords, r.config_json, r.status,
                           r.created_at, r.updated_at, r.failure_reason
                           {derived_cols}
                    FROM runs r
                    LEFT JOIN run_derived d ON d.run_id = r.run_id
                    ORDER BY r.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d: Dict[str, Any] = dict(r)
                derived: Dict[str, Any] = {}
                for key in list(d.keys()):
                    if key.startswith("drv_"):
                        derived[key[4:]] = d.pop(key)
                if any(v is not None for v in derived.values()):
                    d["derived"] = derived
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
        _ensure_failure_reason_column(conn)
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT run_id, script_name, theme_keywords, config_json, status, created_at, updated_at, failure_reason FROM runs WHERE run_id = %s",
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


def count_runs_by_status() -> Dict[str, Any]:
    """Return total count and counts per status (running, completed, failed) across all runs."""

    def _run(conn: Any) -> Dict[str, Any]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT status, COUNT(*) AS cnt FROM runs GROUP BY status
                """
            )
            rows = cur.fetchall()
            by_status = {"running": 0, "completed": 0, "failed": 0}
            total = 0
            for r in rows:
                s = (r.get("status") or "").lower()
                cnt = int(r.get("cnt") or 0)
                total += cnt
                if s in by_status:
                    by_status[s] = cnt
            return {"total": total, "by_status": by_status}
        finally:
            cur.close()

    return _execute(_run, default={"total": 0, "by_status": {"running": 0, "completed": 0, "failed": 0}})


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
    """Load all archive cells for a run as list of dicts (lines/scores from candidates)."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        _ensure_archive_cells_normalized(conn)
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                """
                SELECT ac.cell_key, ac.candidate_id,
                       c.lines_json, c.fitness, c.scores_json
                FROM archive_cells ac
                INNER JOIN candidates c ON c.candidate_id = ac.candidate_id
                WHERE ac.run_id = %s
                """,
                (run_id,),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                raw_lines = r.get("lines_json")
                if isinstance(raw_lines, str):
                    try:
                        lines_parsed = json.loads(raw_lines) if raw_lines else []
                    except (json.JSONDecodeError, TypeError):
                        lines_parsed = []
                elif raw_lines is not None:
                    lines_parsed = raw_lines
                else:
                    lines_parsed = []
                raw_scores = r.get("scores_json")
                scores_parsed = None
                if raw_scores:
                    if isinstance(raw_scores, str):
                        try:
                            scores_parsed = json.loads(raw_scores)
                        except (json.JSONDecodeError, TypeError):
                            scores_parsed = None
                    else:
                        scores_parsed = raw_scores
                d: Dict[str, Any] = {
                    "cell_key": r["cell_key"],
                    "candidate_id": r["candidate_id"],
                    "lines": lines_parsed,
                    "fitness": float(r["fitness"]) if r.get("fitness") is not None else 0.0,
                    "scores": scores_parsed,
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


def insert_seed_bank(
    seed_key: str,
    seed_data: Dict[str, Any],
    run_id: Optional[int] = None,
) -> int:
    """Insert a seed bank entry. Returns row id or -1 on failure."""
    def _run(conn: Any) -> int:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO seed_bank (run_id, seed_key, seed_data)
                VALUES (%s, %s, %s)
                """,
                (run_id, seed_key, json.dumps(seed_data) if isinstance(seed_data, dict) else seed_data),
            )
            return cur.lastrowid or -1
        finally:
            cur.close()
    return _execute(_run, default=-1)


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


def list_song_artists() -> List[str]:
    """List distinct artist names from songs table. Returns [] if DB disabled or empty."""

    def _run(conn: Any) -> List[str]:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT DISTINCT artist FROM songs ORDER BY artist",
            )
            return [row[0] for row in cur.fetchall() if row and row[0]]
        finally:
            cur.close()

    return _execute(_run, default=[])


def list_songs_for_artist(artist: str) -> List[Dict[str, Any]]:
    """List songs for one artist. Returns [{\"song_id\": \"...\", \"title\": \"...\"}, ...]."""

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT song_id, title FROM songs WHERE artist = %s ORDER BY title",
                (artist,),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()

    return _execute(_run, default=[])


def get_song_lines(song_id: str) -> List[str]:
    """Load ordered lyric lines for a song. Returns [] if song not found or DB disabled."""

    def _run(conn: Any) -> List[str]:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT line_text FROM song_lines WHERE song_id = %s ORDER BY line_index",
                (song_id,),
            )
            return [row[0] for row in cur.fetchall() if row and row[0]]
        finally:
            cur.close()

    return _execute(_run, default=[])


def _diversify_cross_run_candidates(
    rows: List[Dict[str, Any]],
    limit: int,
    *,
    max_per_source_run: int = 2,
) -> List[Dict[str, Any]]:
    """Greedy diversity: cap seeds per source run and skip duplicate line payloads."""
    if not rows or limit <= 0:
        return []
    per_run: Dict[int, int] = {}
    seen_fp: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if len(out) >= limit:
            break
        rid = int(r.get("run_id") or 0)
        if rid <= 0:
            continue
        if per_run.get(rid, 0) >= max_per_source_run:
            continue
        lines = r.get("lines") or []
        try:
            fp = json.dumps(lines, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            fp = str(lines)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        per_run[rid] = per_run.get(rid, 0) + 1
        out.append(r)
    if len(out) < limit:
        for r in rows:
            if len(out) >= limit:
                break
            if r in out:
                continue
            rid = int(r.get("run_id") or 0)
            if rid <= 0:
                continue
            if per_run.get(rid, 0) >= max_per_source_run:
                continue
            lines = r.get("lines") or []
            try:
                fp = json.dumps(lines, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                fp = str(lines)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            per_run[rid] = per_run.get(rid, 0) + 1
            out.append(r)
    return out[:limit]


def load_top_candidates_cross_run(
    *,
    limit: int = 50,
    min_fitness: float = 0.0,
    candidate_type: Optional[str] = "verse4",
    exclude_run_ids: Optional[List[int]] = None,
    recent_runs: int = 200,
    diversify: bool = True,
    max_seeds_per_source_run: int = 2,
) -> List[Dict[str, Any]]:
    """Load top candidates across recent completed runs for cross-run seeding.

    Returns list of dicts with lines, fitness, scores, run_id, sorted by fitness desc.
    When diversify=True, over-samples from SQL then applies per-source-run caps and
    duplicate-line filtering so seeding does not collapse to one prior run.
    """

    def _run(conn: Any) -> List[Dict[str, Any]]:
        cur = conn.cursor(dictionary=True)
        try:
            run_filter = ""
            params: list[Any] = []
            if exclude_run_ids:
                placeholders = ",".join(["%s"] * len(exclude_run_ids))
                run_filter = f" AND r.run_id NOT IN ({placeholders})"
                params.extend(exclude_run_ids)
            sql_limit = int(limit)
            if diversify:
                sql_limit = min(500, max(sql_limit * 5, sql_limit))
            cur.execute(
                f"""
                SELECT c.candidate_id, c.run_id, c.lines_json, c.fitness, c.scores_json
                FROM candidates c
                JOIN runs r ON r.run_id = c.run_id
                WHERE r.status = 'completed'
                  AND c.fitness >= %s
                  {"AND c.candidate_type = %s" if candidate_type else ""}
                  {run_filter}
                ORDER BY c.fitness DESC
                LIMIT %s
                """,
                tuple(
                    [min_fitness]
                    + ([candidate_type] if candidate_type else [])
                    + params
                    + [sql_limit]
                ),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d: Dict[str, Any] = {
                    "candidate_id": r["candidate_id"],
                    "run_id": r["run_id"],
                    "fitness": float(r["fitness"]) if r.get("fitness") is not None else 0.0,
                }
                raw_lines = r.get("lines_json")
                if isinstance(raw_lines, str):
                    try:
                        d["lines"] = json.loads(raw_lines)
                    except (json.JSONDecodeError, TypeError):
                        d["lines"] = []
                elif raw_lines is not None:
                    d["lines"] = raw_lines
                else:
                    d["lines"] = []
                raw_scores = r.get("scores_json")
                if raw_scores:
                    if isinstance(raw_scores, str):
                        try:
                            d["scores"] = json.loads(raw_scores)
                        except (json.JSONDecodeError, TypeError):
                            d["scores"] = None
                    else:
                        d["scores"] = raw_scores
                else:
                    d["scores"] = None
                out.append(d)
            if diversify and out:
                out = _diversify_cross_run_candidates(
                    out, limit, max_per_source_run=max_seeds_per_source_run
                )
            return out[:limit]
        finally:
            cur.close()

    return _execute(_run, default=[])


def operator_event_counts_recent_runs(max_runs: int = 25) -> Dict[str, int]:
    """Count operator_events by operator label over the last N completed runs (by run_id)."""

    def _run(conn: Any) -> Dict[str, int]:
        cur = conn.cursor(dictionary=True)
        try:
            # JOIN avoids LIMIT-in-subquery issues with IN (...) on some MySQL setups.
            cur.execute(
                """
                SELECT oe.operator AS op, COUNT(*) AS c
                FROM operator_events oe
                INNER JOIN (
                    SELECT run_id FROM runs
                    WHERE status = 'completed'
                    ORDER BY run_id DESC
                    LIMIT %s
                ) r ON r.run_id = oe.run_id
                GROUP BY oe.operator
                """,
                (max(1, int(max_runs)),),
            )
            rows = cur.fetchall() or []
            return {str(r["op"]): int(r["c"]) for r in rows if r.get("op")}
        finally:
            cur.close()

    return _execute(_run, default={}) or {}


def run_status_counts() -> Dict[str, int]:
    """Return {\"completed\": n, \"running\": n, ...} for all runs."""

    def _run(conn: Any) -> Dict[str, int]:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT status, COUNT(*) AS c FROM runs GROUP BY status"
            )
            rows = cur.fetchall() or []
            return {str(r["status"]): int(r["c"]) for r in rows if r.get("status")}
        finally:
            cur.close()

    return _execute(_run, default={}) or {}
