#!/usr/bin/env python3
"""
Bootstrap policy and run continuous evolution with learning.

Calls update_learned_policy.py to create/refresh artifacts/learned_policy.json
from recent DB runs, then runs run_continuous.py with policy enabled. Policy
is updated periodically by run_continuous (--update-policy-every).

Usage:
    python scripts/run_learning_loop.py
    python scripts/run_learning_loop.py --config config/continuous_runs.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap policy and run continuous evolution with learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/continuous_runs.yaml",
        help="YAML config for run_continuous (default: config/continuous_runs.yaml)",
    )
    parser.add_argument(
        "--update-policy-every",
        type=int,
        default=30,
        metavar="N",
        help="Update learned policy every N completed runs (default: 30)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip initial policy bootstrap (use existing learned_policy.json)",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Force Docker for evolution",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Force direct Python (no Docker)",
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Pause between runs (default: 0)",
    )
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=0.8,
        metavar="F",
        help="Failure penalty for update_learned_policy (default: 0.8)",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["RAPBOT_USE_DB"] = "1"

    if not args.no_bootstrap:
        logger.info("Bootstrapping learned policy from DB...")
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "update_learned_policy.py"),
                "--failure-penalty", str(args.failure_penalty),
            ],
            cwd=str(ROOT),
            env=env,
        )
        if rc.returncode != 0:
            logger.warning("update_learned_policy exited %d", rc.returncode)
        else:
            logger.info("Policy bootstrap complete")
    else:
        logger.info("Skipping policy bootstrap (--no-bootstrap)")

    logger.info("Starting continuous run loop with policy learning...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_continuous.py"),
        "--config", args.config,
        "--update-policy-every", str(args.update_policy_every),
        "--policy-failure-penalty", str(args.failure_penalty),
    ]
    if args.use_docker:
        cmd.append("--use-docker")
    if args.no_docker:
        cmd.append("--no-docker")
    if args.pause > 0:
        cmd.extend(["--pause", str(args.pause)])

    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
