#!/usr/bin/env python3
"""
Recommend control settings to achieve target objective constraints (e.g. rhyme>=0.7, flow>=0.6).

Uses recent run data and a simple model or grid search to suggest control profiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_recent_run_rows(limit: int = 200) -> List[Dict[str, Any]]:
    """Load runs and aggregate outcomes (same as analyze_control_impact)."""
    import os
    os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        from evo_rhyme.experiment_metrics import aggregate_outcome
    except ImportError:
        return []
    if not db.db_enabled():
        return []
    run_list = db.list_runs(limit=limit, offset=0)
    run_ids = [r["run_id"] for r in run_list]
    rows = []
    for run_id in run_ids:
        run = db.get_run(run_id)
        if not run:
            continue
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
        })
    return rows


def recommend_by_constraints(
    rows: List[Dict[str, Any]],
    constraints: Dict[str, float],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Filter rows that meet all constraints (e.g. rhyme >= 0.7, flow >= 0.6)
    and return top_n by fitness.
    """
    filtered = []
    for r in rows:
        vec = r.get("fitness_vector") or {}
        ok = True
        for key, min_val in constraints.items():
            if key == "fitness":
                if (r.get("fitness") or 0.0) < min_val:
                    ok = False
                    break
            else:
                if vec.get(key, 0.0) < min_val:
                    ok = False
                    break
        if ok:
            filtered.append(r)
    filtered.sort(key=lambda x: x.get("fitness") or 0.0, reverse=True)
    return filtered[:top_n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Recommend controls for target objectives")
    ap.add_argument("--constraints", type=str, default=None, help='JSON e.g. {"rhyme": 0.7, "flow": 0.6}')
    ap.add_argument("--limit", type=int, default=200, help="Max runs to load")
    ap.add_argument("--top", type=int, default=10, help="Top N recommendations")
    ap.add_argument("--output", type=str, default=None, help="Write JSON here")
    args = ap.parse_args()

    constraints = {}
    if args.constraints:
        try:
            constraints = json.loads(args.constraints)
        except json.JSONDecodeError as e:
            print(f"Invalid --constraints JSON: {e}", file=sys.stderr)
            return 1

    rows = load_recent_run_rows(limit=args.limit)
    if not rows:
        print("No run data available (DB disabled or empty).", file=sys.stderr)
        return 1

    if not constraints:
        # No constraints: return top runs by fitness
        recs = sorted(rows, key=lambda x: x.get("fitness") or 0.0, reverse=True)[: args.top]
    else:
        recs = recommend_by_constraints(rows, constraints, top_n=args.top)

    out = {
        "constraints": constraints,
        "recommendations": [
            {
                "run_id": r["run_id"],
                "controls": r["controls"],
                "fitness": r["fitness"],
                "fitness_vector": r.get("fitness_vector"),
            }
            for r in recs
        ],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
