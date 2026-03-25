#!/usr/bin/env python3
"""
Initialize the rapbot MySQL database and create schema.

Uses same env vars as evo_rhyme.db: RAPBOT_USE_DB, RAPBOT_DB_HOST, RAPBOT_DB_PORT,
RAPBOT_DB_USER, RAPBOT_DB_PASSWORD, RAPBOT_DB_NAME.
Creates database 'rapbot' if not exists; uses utf8mb4.

Run: python scripts/init_db.py
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import mysql.connector
except ImportError:
    print("Install mysql-connector-python: pip install mysql-connector-python", file=sys.stderr)
    sys.exit(1)

DB_NAME = os.environ.get("RAPBOT_DB_NAME", "rapbot")
DB_HOST = os.environ.get("RAPBOT_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("RAPBOT_DB_PORT", "3306"))
DB_USER = os.environ.get("RAPBOT_DB_USER", "root")
DB_PASSWORD = os.environ.get("RAPBOT_DB_PASSWORD", "")

# Individual CREATE TABLE statements - robust against semicolons in strings/comments
SCHEMA_STATEMENTS = [
    # runs: evolution run metadata
    """
CREATE TABLE IF NOT EXISTS runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    script_name VARCHAR(255) NOT NULL,
    theme_keywords VARCHAR(1024) DEFAULT NULL,
    config_json JSON DEFAULT NULL,
    status VARCHAR(64) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # generations: per-generation stats
    """
CREATE TABLE IF NOT EXISTS generations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    gen INT NOT NULL,
    best_fitness DOUBLE DEFAULT NULL,
    avg_fitness DOUBLE DEFAULT NULL,
    diversity DOUBLE DEFAULT NULL,
    acceptance_rate DOUBLE DEFAULT NULL,
    extra_json JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    INDEX idx_run_gen (run_id, gen)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # candidates: evolved candidates per run/generation
    """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    gen INT NOT NULL,
    candidate_type VARCHAR(64) NOT NULL,
    scheme VARCHAR(32) DEFAULT NULL,
    lines_json JSON NOT NULL,
    fitness DOUBLE DEFAULT NULL,
    scores_json JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    INDEX idx_run_gen (run_id, gen),
    INDEX idx_candidate_type (candidate_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # lineage: parent-child relationships between candidates
    """
CREATE TABLE IF NOT EXISTS lineage (
    child_id INT NOT NULL,
    parent_id INT NOT NULL,
    operation VARCHAR(64) NOT NULL,
    gen INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (child_id, parent_id),
    FOREIGN KEY (child_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    INDEX idx_gen (gen)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # score_cache: cache scored results by text hash
    """
CREATE TABLE IF NOT EXISTS score_cache (
    text_hash VARCHAR(128) NOT NULL,
    candidate_type VARCHAR(64) NOT NULL,
    scheme VARCHAR(32) NOT NULL,
    scores_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (text_hash, candidate_type, scheme)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # archive_cells: MAP-Elites cell -> candidate_id (lines/scores live on candidates only)
    """
CREATE TABLE IF NOT EXISTS archive_cells (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    cell_key VARCHAR(256) NOT NULL,
    candidate_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    UNIQUE KEY uk_run_cell (run_id, cell_key),
    INDEX idx_run_id (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # run_derived: aggregates for fast dashboard / analysis (refreshed at run completion)
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # operator_events: mutation/crossover trail (optional; mirrors JSONL tracer when enabled)
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # seed_bank: seed lines/blocks for future use
    """
CREATE TABLE IF NOT EXISTS seed_bank (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT DEFAULT NULL,
    seed_key VARCHAR(128) DEFAULT NULL,
    seed_data JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_run (run_id),
    INDEX idx_seed_key (seed_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # artifacts: store large run/result JSON blobs (gzipped)
    """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id INT DEFAULT NULL,
    run_tag VARCHAR(128) DEFAULT NULL,
    kind VARCHAR(64) NOT NULL,
    rel_path VARCHAR(768) DEFAULT NULL,
    sha256 CHAR(64) NOT NULL,
    content_gzip LONGBLOB NOT NULL,
    content_len BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL,
    UNIQUE KEY uk_sha256 (sha256),
    INDEX idx_run_id (run_id),
    INDEX idx_run_tag (run_tag),
    INDEX idx_kind (kind)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # experiments: experiment metadata for causal control runs
    """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(1024) DEFAULT NULL,
    mode VARCHAR(64) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # experiment_arms: one row per arm (control configuration)
    """
CREATE TABLE IF NOT EXISTS experiment_arms (
    arm_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    arm_name VARCHAR(255) NOT NULL,
    control_snapshot JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    INDEX idx_experiment_id (experiment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # songs: artist/song catalog for seed population
    """
CREATE TABLE IF NOT EXISTS songs (
    song_id VARCHAR(255) PRIMARY KEY,
    artist VARCHAR(512) NOT NULL,
    title VARCHAR(512) NOT NULL,
    song_score DOUBLE DEFAULT NULL,
    num_bars INT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_artist (artist)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
    # song_lines: lyric lines per song for seed generation
    """
CREATE TABLE IF NOT EXISTS song_lines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    song_id VARCHAR(255) NOT NULL,
    line_index INT NOT NULL,
    line_text TEXT NOT NULL,
    INDEX idx_song_id (song_id),
    FOREIGN KEY (song_id) REFERENCES songs(song_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""",
]


def _add_runs_experiment_columns(cur) -> None:
    """Add experiment_id and arm_id to runs if not present (idempotent)."""
    for col in ("experiment_id", "arm_id"):
        try:
            cur.execute(
                f"ALTER TABLE runs ADD COLUMN {col} INT NULL DEFAULT NULL",
            )
        except mysql.connector.Error as e:
            if "Duplicate column" in str(e):
                pass
            else:
                raise


def _add_runs_failure_reason(cur) -> None:
    """Add failure_reason to runs if not present (idempotent)."""
    try:
        cur.execute(
            "ALTER TABLE runs ADD COLUMN failure_reason VARCHAR(4096) DEFAULT NULL",
        )
    except mysql.connector.Error as e:
        if "Duplicate column" in str(e):
            pass
        else:
            raise


def _normalize_archive_cells_drop_payload_columns(cur) -> None:
    """
    Older schemas duplicated lines/fitness/scores on archive_cells; occupants now join candidates.
    Drops redundant columns if they exist (idempotent per column).
    """
    for col in ("lines_json", "fitness", "scores_json"):
        try:
            cur.execute(f"ALTER TABLE archive_cells DROP COLUMN `{col}`")
        except mysql.connector.Error as e:
            err = str(e).lower()
            if "unknown column" in err or "check that column" in err or "1091" in err:
                pass
            else:
                raise


def _ensure_run_derived_and_operator_tables(cur) -> None:
    """Create run_derived and operator_events if missing (idempotent)."""
    stmts = [
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
        """,
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
        """,
    ]
    for stmt in stmts:
        cur.execute(stmt.strip())


def main() -> int:
    conn = None
    try:
        # Connect without database to create it
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            charset="utf8mb4",
        )
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cur.close()
        conn.close()

        # Reconnect with database
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4",
        )
        cur = conn.cursor()

        for stmt in SCHEMA_STATEMENTS:
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)

        _add_runs_experiment_columns(cur)
        _add_runs_failure_reason(cur)
        _normalize_archive_cells_drop_payload_columns(cur)
        _ensure_run_derived_and_operator_tables(cur)

        conn.commit()
        cur.close()
        conn.close()
        print(f"Database '{DB_NAME}' initialized successfully.")
        return 0
    except mysql.connector.Error as e:
        print(f"MySQL error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
