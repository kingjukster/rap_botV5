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
    # archive_cells: MAP-Elites archive cells per run
    """
CREATE TABLE IF NOT EXISTS archive_cells (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    cell_key VARCHAR(256) NOT NULL,
    candidate_id INT NOT NULL,
    lines_json JSON NOT NULL,
    fitness DOUBLE DEFAULT NULL,
    scores_json JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    UNIQUE KEY uk_run_cell (run_id, cell_key),
    INDEX idx_run_id (run_id)
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
]


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
