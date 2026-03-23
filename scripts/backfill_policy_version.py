#!/usr/bin/env python3
"""
Backfill policy_version in config_json for all runs that don't have it.

Adds policy_version="static" to runs without policy_version, so the runs_statistics
notebook's policy progression section can group and display them.

Usage:
  python scripts/backfill_policy_version.py
  python scripts/backfill_policy_version.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_POLICY_VERSION = "static"


def _has_policy_version(cfg: dict) -> bool:
    if cfg.get("policy_version") is not None:
        return True
    cs = cfg.get("control_snapshot")
    if isinstance(cs, dict) and cs.get("policy_version") is not None:
        return True
    return False


def _add_policy_version(cfg: dict, value: str) -> dict:
    """Add policy_version to config. Modifies in place and returns cfg."""
    if cfg.get("policy_version") is None:
        cfg["policy_version"] = value
    cs = cfg.get("control_snapshot")
    if isinstance(cs, dict) and cs.get("policy_version") is None:
        cs["policy_version"] = value
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill policy_version for runs without it")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating DB")
    parser.add_argument("--value", type=str, default=DEFAULT_POLICY_VERSION, help="policy_version value (default: static)")
    args = parser.parse_args()

    try:
        from evo_rhyme import db
    except ImportError as e:
        logger.error("Import failed: %s", e)
        return 1

    if not db.db_enabled():
        logger.error("DB not enabled. Set RAPBOT_USE_DB=1 and configure MySQL.")
        return 1

    runs = db.list_runs(limit=10000, offset=0)
    to_update = []
    for r in runs:
        cfg = r.get("config_json") or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except json.JSONDecodeError:
                continue
        if not _has_policy_version(cfg):
            to_update.append((r["run_id"], cfg))

    logger.info("Runs total: %d, missing policy_version: %d", len(runs), len(to_update))

    if not to_update:
        logger.info("Nothing to update.")
        return 0

    if args.dry_run:
        logger.info("Dry run: would update run_ids %s ... (%d runs)", [rid for rid, _ in to_update[:5]], len(to_update))
        return 0

    updated = 0
    for run_id, cfg in to_update:
        _add_policy_version(cfg, args.value)
        db.update_run_config(run_id, cfg)
        updated += 1
        if updated % 50 == 0:
            logger.info("Updated %d runs...", updated)

    logger.info("Done. Updated %d runs with policy_version=%s.", updated, args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
