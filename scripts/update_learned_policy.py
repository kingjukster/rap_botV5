#!/usr/bin/env python3
"""
Update learned policy: analyze recent experiments and write recommended defaults to artifacts/learned_policy.json.

Applies failure penalty so configs that often fail are downweighted. Optionally skips update if too few runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Penalty applied to score when run status is 'failed' (avoids favoring unstable configs)
DEFAULT_FAILURE_PENALTY = 0.8
MIN_RUNS_FOR_UPDATE = 15
DEFAULT_SELECTION_PRESSURE = 2.0


def _load_recent_run_rows(limit: int = 300) -> List[Dict[str, Any]]:
    os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        from evo_rhyme.experiment_metrics import aggregate_outcome
    except ImportError:
        return []
    if not db.db_enabled():
        return []
    run_list = db.list_runs(limit=limit, offset=0)
    rows = []
    for r in run_list:
        run_id = r["run_id"]
        run = db.get_run(run_id)
        if not run:
            continue
        status = (run.get("status") or "").strip().lower()
        candidates = db.list_candidates(run_id, gen=None, limit=500)
        cands = [{"fitness": c.get("fitness"), "scores": c.get("scores") or c.get("scores_json")} for c in candidates]
        agg = aggregate_outcome(cands, mode="best", top_k=5, score_key="fitness", scores_key="scores")
        config = run.get("config_json") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                config = {}
        rows.append({
            "run_id": run_id,
            "controls": config,
            "fitness": agg["fitness"],
            "fitness_vector": agg["fitness_vector"],
            "status": status,
        })
    return rows


def _score_for_policy(
    row: Dict[str, Any],
    failure_penalty: float,
    selection_pressure: float = 1.0,
) -> float:
    """Score used for ranking configs. Penalizes failed runs. selection_pressure>1 strengthens top configs."""
    fitness = float(row.get("fitness") or 0.0)
    status = (row.get("status") or "").strip().lower()
    base = fitness ** selection_pressure
    if status == "failed":
        return max(0.0, base - failure_penalty)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update learned policy from recent DB runs (with failure penalty)"
    )
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=DEFAULT_FAILURE_PENALTY,
        metavar="F",
        help=f"Score penalty for failed runs (default: {DEFAULT_FAILURE_PENALTY})",
    )
    parser.add_argument(
        "--selection-pressure",
        type=float,
        default=DEFAULT_SELECTION_PRESSURE,
        metavar="F",
        help=f"Fitness exponent for ranking; >1 strengthens top configs (default: {DEFAULT_SELECTION_PRESSURE})",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=MIN_RUNS_FOR_UPDATE,
        metavar="N",
        help=f"Skip update if fewer than N runs (default: {MIN_RUNS_FOR_UPDATE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        metavar="N",
        help="Max runs to consider (default: 300)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update even if below min-runs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows = _load_recent_run_rows(limit=args.limit)
    if not rows:
        policy = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "no_data",
            "recommended_controls": {},
            "top_configs": [],
            "policy_version": "v1",
            "policy_hash": None,
        }
    elif not args.force and len(rows) < args.min_runs:
        logger.info(
            "Skipping policy update: %d runs (below --min-runs=%d). Use --force to override.",
            len(rows), args.min_runs,
        )
        return 0
    else:
        # Rank by score (fitness^pressure minus failure penalty). Failed configs are downweighted.
        score_fn = lambda r: _score_for_policy(r, args.failure_penalty, args.selection_pressure)
        top_k = sorted(rows, key=score_fn, reverse=True)[:5]
        best = top_k[0]
        top_configs = [
            {
                "controls": r.get("controls") or {},
                "score": score_fn(r),
                "fitness": float(r.get("fitness") or 0.0),
                "status": r.get("status"),
                "source_run_id": r.get("run_id"),
            }
            for r in top_k
        ]
        policy = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "best_run",
            "source_run_id": best.get("run_id"),
            "recommended_controls": best.get("controls") or {},
            "expected_fitness": best.get("fitness"),
            "expected_fitness_vector": best.get("fitness_vector"),
            "top_configs": top_configs,
            "policy_version": datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S"),
            "policy_hash": None,
        }
        policy_str = json.dumps(policy, sort_keys=True)
        policy["policy_hash"] = hashlib.sha256(policy_str.encode("utf-8")).hexdigest()[:16]

    out_path = ROOT / "artifacts" / "learned_policy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
