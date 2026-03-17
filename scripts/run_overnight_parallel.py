#!/usr/bin/env python3
"""
Launch parallel evolution runs with different models, running until 10 AM.

Usage:
    python scripts/run_overnight_parallel.py
    python scripts/run_overnight_parallel.py --until 10:00
    python scripts/run_overnight_parallel.py --until 08:30

Spawns 3 verse QD runs in parallel:
  - gpt-5-mini, theme: pressure,mask,survival
  - gpt-4.1-nano, theme: crown,empire,power
  - gpt-4o-mini, theme: flow,show,dream

All runs use --db, --runs-dir, --init lm. At 10 AM (or --until time),
all child processes are terminated gracefully.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default: run until 10:00 AM (next day if already past)
DEFAULT_UNTIL = "10:00"

RUNS = [
    {
        "model": "gpt-5-mini",
        "theme": "pressure,mask,survival",
        "output": "overnight_gpt5mini.json",
        "tag": "gpt5mini",
    },
    {
        "model": "gpt-4.1-nano",
        "theme": "crown,empire,power",
        "output": "overnight_gpt41nano.json",
        "tag": "gpt41nano",
    },
    {
        "model": "gpt-4o-mini",
        "theme": "flow,show,dream",
        "output": "overnight_gpt4omini.json",
        "tag": "gpt4omini",
    },
]


def parse_time(s: str) -> tuple[int, int]:
    """Parse HH:MM or H:MM into (hour, minute)."""
    parts = s.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return (h, m)


def seconds_until(target_h: int, target_m: int) -> float:
    """Seconds from now until target time (next occurrence)."""
    now = datetime.now()
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parallel evolution until a target time (default 10:00 AM)"
    )
    parser.add_argument(
        "--until",
        type=str,
        default=DEFAULT_UNTIL,
        metavar="HH:MM",
        help="Stop all runs at this time (default: 10:00)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=150,
        help="Generations per run (default: 150, will stop early if --until reached)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--lm-budget",
        type=int,
        default=80,
        help="LM mutation budget per generation (default: 80)",
    )
    args = parser.parse_args()

    target_h, target_m = parse_time(args.until)
    secs = seconds_until(target_h, target_m)
    stop_at = datetime.now() + timedelta(seconds=secs)
    logger.info("Will stop all runs at %s (in %.1f hours)", stop_at.strftime("%Y-%m-%d %H:%M"), secs / 3600)

    os.environ["RAPBOT_USE_DB"] = "1"

    script = ROOT / "scripts" / "run_verse_qd.py"
    log_dir = ROOT / "data" / "evo_rhyme" / "overnight_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen]] = []
    log_handles: list[object] = []

    for i, run in enumerate(RUNS):
        cmd = [
            sys.executable,
            str(script),
            "--theme", run["theme"],
            "--proposer-model", run["model"],
            "--init", "lm",
            "--generations", str(args.generations),
            "--population", str(args.population),
            "--lm-budget", str(args.lm_budget),
            "--runs-dir",
            "--db",
            "--output", run["output"],
        ]
        log_path = log_dir / f"{run['tag']}.log"
        logger.info("Starting run %d: %s (model=%s) -> %s", i + 1, run["tag"], run["model"], log_path)
        logf = open(log_path, "w", encoding="utf-8")
        log_handles.append(logf)
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        procs.append((run["tag"], p))
        time.sleep(5)  # Stagger starts to avoid timestamp collision

    logger.info("All %d runs started. Waiting until %s...", len(procs), stop_at.strftime("%H:%M"))

    while True:
        now = datetime.now()
        if now >= stop_at:
            logger.info("Stop time reached. Terminating all runs.")
            for tag, p in procs:
                if p.poll() is None:
                    logger.info("Terminating %s (pid=%d)", tag, p.pid)
                    p.terminate()
            time.sleep(3)
            for tag, p in procs:
                if p.poll() is None:
                    logger.warning("Killing %s (pid=%d)", tag, p.pid)
                    p.kill()
            break

        all_done = True
        for tag, p in procs:
            if p.poll() is None:
                all_done = False
            else:
                logger.info("Run %s finished (exit=%d)", tag, p.returncode)

        if all_done:
            logger.info("All runs completed.")
            break

        time.sleep(60)  # Check every minute

    for f in log_handles:
        try:
            f.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
