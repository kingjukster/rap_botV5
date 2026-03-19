#!/usr/bin/env python3
"""
Mark runs that have been 'running' with no activity for a while as 'failed'.
Use when runs were terminated (e.g. Ctrl+C, container stop) and never updated.

Uses same env vars as evo_rhyme.db.

Run: python scripts/mark_stale_runs.py [--minutes 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from evo_rhyme import db
except ImportError as e:
    print(f"Could not import evo_rhyme.db: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark stale 'running' runs as failed")
    parser.add_argument(
        "--minutes",
        type=int,
        default=30,
        help="Consider runs stale if no activity for this many minutes (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without making changes",
    )
    args = parser.parse_args()

    if not db.db_enabled():
        print("Database is not enabled (RAPBOT_USE_DB=1 required)", file=sys.stderr)
        return 1

    if args.dry_run:
        # Count runs that would be updated (read-only)
        def _count(conn):
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM runs r
                    LEFT JOIN (
                        SELECT run_id, MAX(created_at) AS last_gen_at
                        FROM generations GROUP BY run_id
                    ) g ON r.run_id = g.run_id
                    WHERE r.status = 'running'
                    AND COALESCE(g.last_gen_at, r.created_at) < DATE_SUB(NOW(), INTERVAL %s MINUTE)
                    """,
                    (args.minutes,),
                )
                return cur.fetchone()[0]
            finally:
                cur.close()

        n = db._execute(_count, default=0)
        print(f"Would mark {n} run(s) as failed (no activity for {args.minutes}+ min)")
        return 0

    n = db.mark_stale_runs_failed(minutes_idle=args.minutes)
    print(f"Marked {n} run(s) as failed (had been running with no activity for {args.minutes}+ min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
