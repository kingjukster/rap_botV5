#!/usr/bin/env python3
"""
Import song catalog from elite_songs_lines_clean.csv into the rapbot database.

Uses same env vars as evo_rhyme.db: RAPBOT_USE_DB, RAPBOT_DB_HOST, RAPBOT_DB_PORT,
RAPBOT_DB_USER, RAPBOT_DB_PASSWORD, RAPBOT_DB_NAME.

Run:
    python scripts/import_song_catalog.py
    python scripts/import_song_catalog.py --replace
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_csv_path(configured: str | None) -> Path:
    """Resolve CSV path from env or default."""
    if configured and configured.strip():
        p = Path(configured.strip())
        return p if p.is_absolute() else (ROOT / p)
    return ROOT / "data" / "elite_songs_lines_clean.csv"


def _get_db_config() -> dict:
    """Same config as evo_rhyme.db."""
    return {
        "host": os.environ.get("RAPBOT_DB_HOST", "localhost"),
        "port": int(os.environ.get("RAPBOT_DB_PORT", "3306")),
        "user": os.environ.get("RAPBOT_DB_USER", "root"),
        "password": os.environ.get("RAPBOT_DB_PASSWORD", ""),
        "database": os.environ.get("RAPBOT_DB_NAME", "rapbot"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import song catalog from CSV into rapbot database.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to elite CSV (default: data/elite_songs_lines_clean.csv or RAPBOT_SONG_CATALOG_CSV)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Truncate songs and song_lines before import (full reload)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    csv_path = _get_csv_path(args.csv or os.environ.get("RAPBOT_SONG_CATALOG_CSV"))
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return 1

    try:
        import mysql.connector
    except ImportError:
        logger.error("Install mysql-connector-python: pip install mysql-connector-python")
        return 1

    cfg = _get_db_config()
    try:
        conn = mysql.connector.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset="utf8mb4",
        )
    except Exception as e:
        logger.error("DB connection failed: %s", e)
        return 1

    cur = conn.cursor()

    if args.replace:
        logger.info("Truncating song_lines and songs...")
        try:
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            cur.execute("TRUNCATE TABLE song_lines")
            cur.execute("TRUNCATE TABLE songs")
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            conn.commit()
        except Exception as e:
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            conn.rollback()
            logger.error("Truncate failed: %s", e)
            cur.close()
            conn.close()
            return 1

    # First pass: collect distinct songs
    songs_seen: set[tuple[str, str, str]] = set()
    songs_to_insert: list[tuple[str, str, str, float | None]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            song_id = str(row.get("song_id") or "").strip()
            artist = str(row.get("artist") or "").strip()
            title = str(row.get("title") or "").strip()
            if not song_id or not artist or not title:
                continue
            key = (song_id, artist, title)
            if key in songs_seen:
                continue
            songs_seen.add(key)
            try:
                score = float(row.get("song_score") or 0)
            except (TypeError, ValueError):
                score = None
            songs_to_insert.append((song_id, artist, title, score))

    logger.info("Inserting %d songs...", len(songs_to_insert))
    inserted_song_ids: set[str] = set()
    for song_id, artist, title, score in songs_to_insert:
        try:
            if args.replace:
                cur.execute(
                    "INSERT INTO songs (song_id, artist, title, song_score) VALUES (%s, %s, %s, %s)",
                    (song_id, artist, title, score),
                )
            else:
                cur.execute(
                    """
                    INSERT IGNORE INTO songs (song_id, artist, title, song_score)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (song_id, artist, title, score),
                )
            if cur.rowcount > 0:
                inserted_song_ids.add(song_id)
        except Exception as e:
            logger.warning("Insert song %s failed: %s", song_id, e)
    conn.commit()
    logger.info("Inserted %d songs", len(inserted_song_ids))

    # Valid song_ids: when --replace, all we inserted; else only newly inserted
    valid_song_ids = (
        {s[0] for s in songs_to_insert}
        if args.replace
        else inserted_song_ids
    )

    # Second pass: insert song lines (only for songs that exist in DB)
    lines_to_insert: list[tuple[str, int, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            song_id = str(row.get("song_id") or "").strip()
            line_text = str(row.get("line_text") or "").strip()
            if not song_id or not line_text:
                continue
            if song_id not in valid_song_ids:
                continue
            try:
                line_idx = int(str(row.get("line_index") or "0").strip())
            except (TypeError, ValueError):
                line_idx = len(lines_to_insert)
            lines_to_insert.append((song_id, line_idx, line_text))

    logger.info("Inserting %d song lines...", len(lines_to_insert))
    batch_size = 500
    inserted_lines = 0
    for i in range(0, len(lines_to_insert), batch_size):
        batch = lines_to_insert[i : i + batch_size]
        try:
            cur.executemany(
                "INSERT INTO song_lines (song_id, line_index, line_text) VALUES (%s, %s, %s)",
                batch,
            )
            inserted_lines += cur.rowcount
        except Exception as e:
            logger.warning("Batch insert failed: %s", e)
            for song_id, line_idx, line_text in batch:
                try:
                    cur.execute(
                        "INSERT INTO song_lines (song_id, line_index, line_text) VALUES (%s, %s, %s)",
                        (song_id, line_idx, line_text),
                    )
                    inserted_lines += cur.rowcount
                except Exception as e2:
                    logger.debug("Insert line failed for %s: %s", song_id, e2)
    conn.commit()
    logger.info("Inserted %d song lines", inserted_lines)

    # Update num_bars
    logger.info("Updating num_bars...")
    try:
        cur.execute(
            """
            UPDATE songs s
            SET num_bars = (SELECT COUNT(*) FROM song_lines sl WHERE sl.song_id = s.song_id)
            """
        )
        conn.commit()
        logger.info("Updated num_bars for %d songs", cur.rowcount)
    except Exception as e:
        logger.warning("Update num_bars failed: %s", e)
        conn.rollback()

    cur.close()
    conn.close()
    logger.info("Import complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
