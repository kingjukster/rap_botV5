#!/usr/bin/env python3
"""Dispatch a full evolution campaign to RunPod Serverless.

Uses plain HTTP requests (no SDK dependencies). Runs inside Docker or locally.

Usage:
    # Start ngrok tunnel first
    source scripts/start_ngrok_db.sh

    # Dry run (list configs)
    python scripts/run_runpod_campaign.py --dry-run

    # Launch campaign
    python scripts/run_runpod_campaign.py

    # Via docker compose (no local Python needed)
    docker compose --profile evolution run --rm evolution \
      python scripts/run_runpod_campaign.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("runpod_campaign")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent
RUNPOD_API = "https://api.runpod.ai/v2"


def load_configs(config_path: str) -> List[Dict[str, Any]]:
    """Load run configs from YAML (same logic as run_continuous.py)."""
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    runs = data.get("runs", [])
    default_lm = int(data.get("lm_budget", 0))
    default_line_lm = int(data.get("line_lm_budget", 0))
    default_seed_from_archive = int(data.get("seed_from_archive", 0))
    default_archive_mode = data.get("archive_mode", "compact_style")
    default_immigrants = int(data.get("immigrants", 20))
    seeds = data.get("seeds")
    if seeds is None:
        seeds = []
    elif isinstance(seeds, int):
        seeds = [seeds]

    out: List[Dict[str, Any]] = []
    for r in runs:
        c = {
            "arm": r.get("arm", "unknown"),
            "theme": r.get("theme", "general"),
            "population": int(r.get("population", 60)),
            "generations": int(r.get("generations", 20)),
            "scheme": r.get("scheme", "AABB"),
            "init": r.get("init", "mixed"),
            "lm_budget": int(r.get("lm_budget", default_lm)),
            "line_lm_budget": int(r.get("line_lm_budget", default_line_lm)),
            "seed_from_archive": int(r.get("seed_from_archive", default_seed_from_archive)),
            "archive_mode": r.get("archive_mode", default_archive_mode),
            "immigrants": int(r.get("immigrants", default_immigrants)),
        }
        if seeds:
            for seed in seeds:
                cc = dict(c)
                cc["seed"] = seed
                out.append(cc)
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# RunPod API helpers (plain HTTP, no SDK)
# ---------------------------------------------------------------------------

def _runpod_request(
    endpoint_id: str,
    path: str,
    api_key: str,
    payload: Optional[dict] = None,
    method: str = "GET",
) -> dict:
    url = f"{RUNPOD_API}/{endpoint_id}/{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def submit_job(endpoint_id: str, api_key: str, cfg: dict) -> str:
    """Submit async job. Returns job ID."""
    result = _runpod_request(endpoint_id, "run", api_key, payload={"input": cfg}, method="POST")
    return result["id"]


def check_job_status(endpoint_id: str, api_key: str, job_id: str) -> dict:
    """Check job status. Returns {status, output?, error?}."""
    return _runpod_request(endpoint_id, f"status/{job_id}", api_key)


def check_health(endpoint_id: str, api_key: str) -> dict:
    return _runpod_request(endpoint_id, "health", api_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch evolution campaign to RunPod")
    parser.add_argument("--config", type=str, default="config/continuous_runs.yaml")
    parser.add_argument("--endpoint-id", type=str, default=os.environ.get("RUNPOD_EVO_ENDPOINT_ID", ""),
                        help="RunPod Serverless endpoint ID for evolution workers")
    parser.add_argument("--db-host", type=str, default="",
                        help="Public DB host for workers (e.g. bore.pub)")
    parser.add_argument("--db-port", type=str, default="",
                        help="Public DB port for workers (e.g. 24683)")
    parser.add_argument("--dry-run", action="store_true", help="List configs without dispatching")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls")
    parser.add_argument("--test", action="store_true",
                        help="Submit a single tiny test job (pop=10, gens=2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    configs = load_configs(args.config)
    logger.info("Loaded %d configs from %s", len(configs), args.config)

    if args.test:
        configs = [configs[0].copy()]
        configs[0].update({"population": 10, "generations": 2, "lm_budget": 1, "line_lm_budget": 0})
        logger.info("TEST MODE: 1 job, pop=10, gens=2")

    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"  [{i+1:2d}] arm={cfg['arm']:25s} seed={str(cfg.get('seed','?')):>2} "
                  f"pop={cfg['population']} gens={cfg['generations']} "
                  f"lm={cfg['lm_budget']} scheme={cfg['scheme']}")
        print(f"\nTotal: {len(configs)} configs")
        return

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        logger.error("RUNPOD_API_KEY not set")
        sys.exit(1)

    endpoint_id = args.endpoint_id or os.environ.get("RUNPOD_EVO_ENDPOINT_ID", "").strip()
    if not endpoint_id:
        logger.error("No endpoint ID. Set --endpoint-id or RUNPOD_EVO_ENDPOINT_ID env var")
        sys.exit(1)

    db_host = args.db_host or os.environ.get("RAPBOT_RUNPOD_DB_HOST", "").strip()
    db_port = args.db_port or os.environ.get("RAPBOT_RUNPOD_DB_PORT", "").strip()
    db_overrides: Dict[str, str] = {}
    if db_host and db_port:
        db_overrides = {
            "rapbot_db_host": db_host,
            "rapbot_db_port": db_port,
            "rapbot_use_db": "1",
        }
        logger.info("DB tunnel: %s:%s", db_host, db_port)
    else:
        logger.warning("No --db-host/--db-port set. Workers will use endpoint env vars for DB.")

    health = check_health(endpoint_id, api_key)
    workers = health.get("workers", {})
    logger.info("Endpoint %s health: %d ready, %d idle, %d running",
                endpoint_id, workers.get("ready", 0), workers.get("idle", 0), workers.get("running", 0))

    logger.info("Dispatching %d jobs...", len(configs))
    jobs: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        cfg.update(db_overrides)
        try:
            job_id = submit_job(endpoint_id, api_key, cfg)
            jobs.append({"index": i, "cfg": cfg, "job_id": job_id, "status": "IN_QUEUE"})
            logger.info("Dispatched [%d/%d]: arm=%s seed=%s -> job=%s",
                        i + 1, len(configs), cfg["arm"], cfg.get("seed", "?"), job_id)
        except Exception as exc:
            logger.error("Failed to dispatch [%d/%d] arm=%s: %s", i + 1, len(configs), cfg["arm"], exc)
            jobs.append({"index": i, "cfg": cfg, "job_id": None, "status": "SUBMIT_FAILED"})

    logger.info("All jobs dispatched. Polling for completion (every %ds)...", args.poll_interval)
    start_time = time.monotonic()

    while True:
        pending = [j for j in jobs if j["status"] not in ("COMPLETED", "FAILED", "SUBMIT_FAILED", "CANCELLED")]
        if not pending:
            break

        time.sleep(args.poll_interval)

        for entry in pending:
            if not entry["job_id"]:
                continue
            try:
                result = check_job_status(endpoint_id, api_key, entry["job_id"])
                status = result.get("status", "UNKNOWN")
                entry["status"] = status
            except Exception as exc:
                logger.debug("Status check failed for %s: %s", entry["job_id"], exc)
                continue

        done = sum(1 for j in jobs if j["status"] == "COMPLETED")
        failed = sum(1 for j in jobs if j["status"] in ("FAILED", "SUBMIT_FAILED", "CANCELLED"))
        in_progress = sum(1 for j in jobs if j["status"] == "IN_PROGRESS")
        in_queue = sum(1 for j in jobs if j["status"] == "IN_QUEUE")
        elapsed = time.monotonic() - start_time

        for entry in jobs:
            cfg = entry["cfg"]
            if entry["status"] == "COMPLETED" and not entry.get("_logged"):
                entry["_logged"] = True
                logger.info("DONE: arm=%s seed=%s (%.1f min)", cfg["arm"], cfg.get("seed", "?"), elapsed / 60)
            elif entry["status"] == "FAILED" and not entry.get("_logged"):
                entry["_logged"] = True
                logger.error("FAILED: arm=%s seed=%s (%.1f min)", cfg["arm"], cfg.get("seed", "?"), elapsed / 60)

        logger.info("Progress: %d done, %d failed, %d running, %d queued (%.1f min elapsed)",
                     done, failed, in_progress, in_queue, elapsed / 60)

    total_time = time.monotonic() - start_time
    completed = sum(1 for j in jobs if j["status"] == "COMPLETED")
    failed = sum(1 for j in jobs if j["status"] in ("FAILED", "SUBMIT_FAILED", "CANCELLED"))
    logger.info("Campaign complete: %d succeeded, %d failed out of %d (%.1f min)",
                completed, failed, len(jobs), total_time / 60)


if __name__ == "__main__":
    main()
