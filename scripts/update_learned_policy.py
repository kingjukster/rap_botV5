#!/usr/bin/env python3
"""
Update learned policy: analyze recent experiments and write recommended defaults to artifacts/learned_policy.json.
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        candidates = db.list_candidates(run_id, gen=None, limit=500)
        cands = [{"fitness": c.get("fitness"), "scores": c.get("scores") or c.get("scores_json")} for c in candidates]
        agg = aggregate_outcome(cands, mode="best", top_k=5, score_key="fitness", scores_key="scores")
        config = run.get("config_json") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                config = {}
        rows.append({"run_id": run_id, "controls": config, "fitness": agg["fitness"], "fitness_vector": agg["fitness_vector"]})
    return rows


def main() -> int:
    rows = _load_recent_run_rows(limit=300)
    if not rows:
        policy = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "no_data",
            "recommended_controls": {},
            "top_configs": [],
            "policy_version": "v1",
            "policy_hash": None,
        }
    else:
        # Use top-K runs as policy distribution; best remains recommended baseline
        top_k = sorted(rows, key=lambda r: r.get("fitness") or 0.0, reverse=True)[:5]
        best = top_k[0]
        top_configs = [
            {
                "controls": r.get("controls") or {},
                "score": float(r.get("fitness") or 0.0),
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
