#!/usr/bin/env python3
"""
Evaluate automated verse scorers against human labels and pairwise judgments.

Spearman + MSE per axis; pairwise full-matrix + directional_aux (tie_policy v1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.metric_benchmark.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verses", type=Path, required=True, help="Split JSONL with labels")
    p.add_argument("--pairs", type=Path, default=None, help="pairs_*.jsonl with better filled")
    p.add_argument("--out", type=Path, default=ROOT / "data/metric_benchmark/eval_report.json")
    p.add_argument("--fast", action="store_true", help="Skip rhyme graph metrics")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--protocol", type=Path, default=ROOT / "data/metric_benchmark/splits/split_manifest.json")
    args = p.parse_args()

    report = run_evaluation(
        args.verses,
        args.pairs,
        fast=args.fast,
        epsilon=args.epsilon,
        protocol_path=args.protocol if args.protocol.exists() else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
