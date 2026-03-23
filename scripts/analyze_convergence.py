#!/usr/bin/env python3
"""
Analyze convergence and overfitting of the evolutionary rap system.

Uses real data from the database and artifacts/learned_policy.json.
Produces a structured report on convergence, diversity, and stagnation.

Usage:
    python scripts/analyze_convergence.py [--limit N] [--output report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of config (control-relevant keys) for grouping."""
    keys = sorted(config.keys())
    canonical = json.dumps({k: config[k] for k in keys if config.get(k) is not None}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_lightweight_modules() -> Tuple[Any, Any]:
    """Load db and experiment_metrics without pulling in evo_rhyme (numpy/torch)."""
    import importlib.util
    os.environ["RAPBOT_USE_DB"] = "1"
    # Load evo_rhyme.db without triggering evo_rhyme.__init__
    spec_db = importlib.util.spec_from_file_location("evo_rhyme.db", ROOT / "evo_rhyme" / "db.py")
    db = importlib.util.module_from_spec(spec_db)
    sys.modules["evo_rhyme"] = type(sys)("evo_rhyme")  # placeholder
    sys.modules["evo_rhyme.db"] = db
    spec_db.loader.exec_module(db)
    # Load experiment_metrics (no heavy deps)
    spec_em = importlib.util.spec_from_file_location(
        "evo_rhyme.experiment_metrics", ROOT / "evo_rhyme" / "experiment_metrics.py"
    )
    em = importlib.util.module_from_spec(spec_em)
    sys.modules["evo_rhyme.experiment_metrics"] = em
    spec_em.loader.exec_module(em)
    return db, em


def load_runs_with_outcomes(limit: int = 2000) -> List[Dict[str, Any]]:
    """Load runs with status, config, and fitness (best candidate)."""
    try:
        db, em = _load_lightweight_modules()
        aggregate_outcome = em.aggregate_outcome
    except Exception as e:
        logger.error("Import failed: %s", e)
        return []

    if not db.db_enabled():
        logger.warning("DB not enabled (RAPBOT_USE_DB=1 and MySQL credentials required)")
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
        cands = [
            {"fitness": c.get("fitness"), "scores": c.get("scores") or c.get("scores_json")}
            for c in candidates
        ]
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
            "config_hash": _config_hash(config),
            "fitness": agg["fitness"],
            "status": status,
            "created_at": r.get("created_at"),
        })
    return rows


def load_policy() -> Dict[str, Any]:
    """Load learned_policy.json."""
    path = ROOT / "artifacts" / "learned_policy.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (bits) of a count distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        if n > 0:
            p = n / total
            ent -= p * math.log2(p)
    return ent


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze convergence and overfitting")
    parser.add_argument("--limit", type=int, default=2000, help="Max runs to load")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report")
    parser.add_argument("--rolling-window", type=int, default=80, help="Rolling window for failure rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = load_runs_with_outcomes(limit=args.limit)
    policy = load_policy()

    if not rows:
        report = {
            "summary": {"converging": "UNKNOWN", "overfitting": "UNKNOWN"},
            "metrics": {"total_runs": 0, "message": "No runs in DB"},
            "key_findings": ["No runs found. Run more evolution experiments."],
            "recommendations": ["Run more experiments with RAPBOT_USE_DB=1"],
        }
    else:
        # Order by run_id (proxy for time; newer runs have higher IDs)
        rows_sorted = sorted(rows, key=lambda x: x["run_id"])

        # --- 1. Run-level metrics ---
        by_status = Counter(r["status"] for r in rows)
        total = len(rows)
        completed = by_status.get("completed", 0)
        failed = by_status.get("failed", 0)
        running = by_status.get("running", 0)

        # Failure rate over rolling window
        w = min(args.rolling_window, total)
        recent = rows_sorted[-w:] if w > 0 else []
        recent_failed = sum(1 for r in recent if r["status"] == "failed")
        recent_total_terminal = sum(1 for r in recent if r["status"] in ("completed", "failed"))
        recent_failure_rate = recent_failed / recent_total_terminal if recent_total_terminal > 0 else 0.0

        # Mean and max fitness over time (by run order)
        fitnesses = [r["fitness"] for r in rows_sorted if r["status"] == "completed"]
        mean_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        max_fitness_overall = max(fitnesses) if fitnesses else 0.0

        # Split into early vs recent halves for trend
        mid = len(rows_sorted) // 2
        early = [r for r in rows_sorted[:mid] if r["status"] == "completed"]
        late = [r for r in rows_sorted[mid:] if r["status"] == "completed"]
        mean_early = sum(r["fitness"] for r in early) / len(early) if early else 0.0
        mean_late = sum(r["fitness"] for r in late) / len(late) if late else 0.0
        max_early = max(r["fitness"] for r in early) if early else 0.0
        max_late = max(r["fitness"] for r in late) if late else 0.0

        mean_trend = "increasing" if mean_late > mean_early + 0.001 else ("decreasing" if mean_late < mean_early - 0.001 else "flat")
        max_trend = "improving" if max_late > max_early + 0.001 else ("declining" if max_late < max_early - 0.001 else "plateaued")

        # --- 2. Policy analysis ---
        top_configs = policy.get("top_configs") or []
        policy_source = policy.get("source", "unknown")
        policy_diversity = "N/A"
        if top_configs:
            config_hashes = [_config_hash(c.get("controls") or {}) for c in top_configs]
            policy_diversity = f"{len(set(config_hashes))} unique in top 5"

        # --- 3. Config convergence ---
        config_counts = Counter(r["config_hash"] for r in rows)
        config_fitness: Dict[str, List[float]] = defaultdict(list)
        config_failures: Dict[str, int] = defaultdict(int)
        for r in rows:
            h = r["config_hash"]
            if r["status"] == "completed":
                config_fitness[h].append(r["fitness"])
            elif r["status"] == "failed":
                config_failures[h] += 1

        avg_fitness_per_config = {h: sum(v) / len(v) for h, v in config_fitness.items() if v}
        top_configs_by_count = config_counts.most_common(10)
        top_configs_by_avg_fitness = sorted(
            avg_fitness_per_config.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # --- 4. Diversity metrics ---
        last_50 = rows_sorted[-50:] if len(rows_sorted) >= 50 else rows_sorted
        last_100 = rows_sorted[-100:] if len(rows_sorted) >= 100 else rows_sorted
        first_50 = rows_sorted[:50] if len(rows_sorted) >= 50 else rows_sorted
        first_100 = rows_sorted[:100] if len(rows_sorted) >= 100 else rows_sorted

        unique_last_50 = len(set(r["config_hash"] for r in last_50))
        unique_last_100 = len(set(r["config_hash"] for r in last_100))
        unique_first_50 = len(set(r["config_hash"] for r in first_50))
        unique_first_100 = len(set(r["config_hash"] for r in first_100))

        diversity_decreasing = unique_last_50 < unique_first_50 and unique_last_100 < unique_first_100
        entropy_all = compute_entropy(dict(config_counts))
        entropy_last_100 = compute_entropy(Counter(r["config_hash"] for r in last_100))

        # --- 5. Stagnation ---
        best_so_far = 0.0
        runs_since_improvement = 0
        for r in reversed(rows_sorted):
            if r["status"] != "completed":
                continue
            if r["fitness"] > best_so_far:
                best_so_far = r["fitness"]
                runs_since_improvement = 0
            else:
                runs_since_improvement += 1

        # --- 6. Failure analysis ---
        failed_config_hashes = set(r["config_hash"] for r in rows if r["status"] == "failed")
        failed_config_counts = Counter(r["config_hash"] for r in rows if r["status"] == "failed")
        top_failed_configs = failed_config_counts.most_common(5)

        # --- Summary and recommendations ---
        converging = "YES" if (mean_trend == "increasing" or max_trend == "improving") else "NO"
        if mean_trend == "flat" and max_trend == "plateaued" and diversity_decreasing:
            converging = "YES (plateaued)"

        overfitting = "NONE"
        if diversity_decreasing and unique_last_50 <= 5 and mean_trend == "flat":
            overfitting = "EARLY"
        if diversity_decreasing and unique_last_50 <= 3 and runs_since_improvement > 50:
            overfitting = "STRONG"

        recommendations: List[str] = []
        if overfitting in ("EARLY", "STRONG"):
            recommendations.append("Increase exploration: raise elite_replay_fraction or epsilon")
            recommendations.append("Add elite replay from diverse configs, not just top performers")
        if recent_failure_rate > 0.15:
            recommendations.append("Policy may not be avoiding bad configs; run update_learned_policy with --failure-penalty")
        if runs_since_improvement > 30:
            recommendations.append("Consider adjusting scoring weights or adding novelty bonus")
        if policy_source == "no_data":
            recommendations.append("Policy has no data; run update_learned_policy.py after more runs complete")
        if not recommendations:
            recommendations.append("Continue monitoring; system appears healthy")

        # Build explicit top 5 configs (control snapshot)
        top_5_configs_explicit: List[Dict[str, Any]] = []
        for h, _ in top_configs_by_avg_fitness[:5]:
            sample = next((r for r in rows if r["config_hash"] == h), None)
            if sample:
                top_5_configs_explicit.append({
                    "config_hash": h,
                    "controls": sample["controls"],
                    "avg_fitness": avg_fitness_per_config.get(h, 0),
                    "count": config_counts.get(h, 0),
                    "failures": config_failures.get(h, 0),
                })

        report = {
            "summary": {
                "converging": converging,
                "overfitting": overfitting,
            },
            "metrics": {
                "total_runs": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "recent_failure_rate": round(recent_failure_rate, 3),
                "recent_window": w,
                "mean_fitness": round(mean_fitness, 4),
                "max_fitness": round(max_fitness_overall, 4),
                "mean_fitness_trend": mean_trend,
                "max_fitness_trend": max_trend,
                "mean_early_half": round(mean_early, 4),
                "mean_late_half": round(mean_late, 4),
                "runs_since_last_improvement": runs_since_improvement,
            },
            "policy_behavior": {
                "policy_source": policy_source,
                "top_configs_count": len(top_configs),
                "policy_diversity": policy_diversity,
                "top_configs_sample": [
                    {"controls": c.get("controls"), "score": c.get("score"), "fitness": c.get("fitness")}
                    for c in top_configs[:3]
                ],
            },
            "config_convergence": {
                "unique_configs_total": len(config_counts),
                "unique_last_50": unique_last_50,
                "unique_last_100": unique_last_100,
                "top_5_configs_by_fitness": top_5_configs_explicit,
                "top_5_configs_by_count": [{"hash": h, "count": n} for h, n in top_configs_by_count[:5]],
            },
            "diversity": {
                "unique_last_50": unique_last_50,
                "unique_last_100": unique_last_100,
                "unique_first_50": unique_first_50,
                "unique_first_100": unique_first_100,
                "diversity_decreasing": diversity_decreasing,
                "entropy_all": round(entropy_all, 3),
                "entropy_last_100": round(entropy_last_100, 3),
            },
            "stagnation": {
                "runs_since_last_improvement": runs_since_improvement,
                "best_fitness": round(best_so_far, 4),
            },
            "failure_analysis": {
                "failed_configs_count": len(failed_config_hashes),
                "top_failed_config_hashes": [{"hash": h, "failures": n} for h, n in top_failed_configs],
            },
            "key_findings": [
                f"Total {total} runs: {completed} completed, {failed} failed, {running} running.",
                f"Recent failure rate (last {w}): {recent_failure_rate:.1%}.",
                f"Mean fitness trend: {mean_trend}; max fitness trend: {max_trend}.",
                f"Diversity: {unique_last_50} unique configs in last 50 vs {unique_first_50} in first 50.",
                f"Runs since last max fitness improvement: {runs_since_improvement}.",
                f"Policy source: {policy_source}; top_configs: {len(top_configs)}.",
            ],
            "recommendations": recommendations,
        }

    # Print human-readable report
    print("\n" + "=" * 60)
    print("CONVERGENCE & OVERFITTING ANALYSIS REPORT")
    print("=" * 60)
    print("\n### Summary")
    print(f"  Converging? {report['summary']['converging']}")
    print(f"  Overfitting? {report['summary']['overfitting']}")

    metrics = report.get("metrics", {})
    if metrics:
        print("\n### Metrics")
        print(f"  Total runs: {metrics.get('total_runs', 'N/A')}")
        print(f"  Recent failure rate: {metrics.get('recent_failure_rate', 'N/A')}")
        print(f"  Mean fitness trend: {metrics.get('mean_fitness_trend', 'N/A')}")
        print(f"  Max fitness trend: {metrics.get('max_fitness_trend', 'N/A')}")
        print(f"  Runs since last improvement: {metrics.get('runs_since_last_improvement', 'N/A')}")

    pb = report.get("policy_behavior", {})
    if pb:
        print("\n### Policy Behavior")
        print(f"  Source: {pb.get('policy_source', 'N/A')}")
        print(f"  Top configs: {pb.get('top_configs_count', 0)}")
        print(f"  Diversity: {pb.get('policy_diversity', 'N/A')}")

    print("\n### Key Findings")
    for f in report.get("key_findings", []):
        print(f"  • {f}")

    print("\n### Recommendations")
    for r in report.get("recommendations", []):
        print(f"  • {r}")
    print()

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
