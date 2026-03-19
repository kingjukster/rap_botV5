#!/usr/bin/env python
"""
run_generations_sweep.py

Run QD verse evolution at multiple generation budgets for tuning (Plan 4).
Calls run_verse_qd.py with --generations N for each N in the sweep.

Usage:
    python scripts/run_generations_sweep.py --theme "crown,empire" --generations 60,80,100 --runs-dir
    python scripts/run_generations_sweep.py --theme "pressure,mask" --generations 80 --replicates 2

Options:
    --theme         Comma-separated theme keywords (required)
    --generations   Comma-separated generation counts (e.g. 60,80,100,120)
    --replicates    Replicates per level (default: 1)
    --runs-dir      Pass through to run_verse_qd for run logging
    --seed          Base seed (optional)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_VERSE_QD = ROOT / "scripts" / "run_verse_qd.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generations sweep for QD verse evolution")
    parser.add_argument("--theme", required=True, help="Comma-separated theme keywords")
    parser.add_argument(
        "--generations",
        type=str,
        default="60,80,100",
        help="Comma-separated generation counts (default: 60,80,100)",
    )
    parser.add_argument("--replicates", type=int, default=1, help="Replicates per level")
    parser.add_argument("--runs-dir", action="store_true", help="Enable run logging")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--extra", type=str, default="", help="Extra args for run_verse_qd")
    args = parser.parse_args()

    gens = [int(x.strip()) for x in args.generations.split(",") if x.strip()]
    if not gens:
        print("No generation levels specified", file=sys.stderr)
        sys.exit(1)

    base_cmd = [
        sys.executable,
        str(RUN_VERSE_QD),
        "--theme",
        args.theme,
    ]
    if args.runs_dir:
        base_cmd.append("--runs-dir")
    if args.seed is not None:
        base_cmd.extend(["--seed", str(args.seed)])
    if args.extra:
        base_cmd.extend(args.extra.split())

    for gen in gens:
        for rep in range(args.replicates):
            seed_arg = [] if args.seed is None else ["--seed", str(args.seed + rep)]
            cmd = base_cmd + ["--generations", str(gen)] + seed_arg
            print(f"\n>>> generations={gen} replicate={rep + 1}/{args.replicates}: {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                print(f"Exit {rc} for gen={gen} rep={rep}", file=sys.stderr)
                sys.exit(rc)
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
