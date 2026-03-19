#!/usr/bin/env python3
"""List the past 20-40 runs from DB and/or file system."""
import json
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from evo_rhyme.db import list_runs, count_runs, db_enabled


def main():
    limit = 40
    if db_enabled():
        runs = list_runs(limit=limit, offset=0)
        total = count_runs()
        print(f"Total runs in DB: {total}\n")
        print(f"{'#':>3} {'run_id':>8} {'status':>10} {'script':<30} {'created_at'}")
        print("-" * 80)
        for i, r in enumerate(runs, 1):
            script = (r.get("script_name") or "")[:28]
            created = str(r.get("created_at") or "")[:19]
            print(f"{i:3} {r.get('run_id', ''):>8} {(r.get('status') or ''):>10} {script:<30} {created}")
    else:
        print("DB not enabled. Showing file-system runs from data/evo_rhyme/runs/")
        runs_dir = root / "data" / "evo_rhyme" / "runs"
        if not runs_dir.exists():
            print("No runs directory found.")
            return
        dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
        dirs = dirs[:limit]
        print(f"\n{'#':>3} {'run_name'}")
        print("-" * 50)
        for i, d in enumerate(dirs, 1):
            print(f"{i:3} {d.name}")


if __name__ == "__main__":
    main()
