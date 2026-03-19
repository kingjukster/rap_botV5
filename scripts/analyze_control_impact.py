#!/usr/bin/env python3
"""
Analyze control impact: load runs (and optionally experiment arms), compute outcomes,
report mean differences, confidence intervals, effect sizes, and correlations.

Usage:
  python scripts/analyze_control_impact.py --experiment-id 1 --output report.json
  python scripts/analyze_control_impact.py --limit 50 --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def load_run_outcomes(
    run_ids: List[int],
    aggregation_mode: str = "best",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Load runs and their candidates; compute outcome per run."""
    try:
        from evo_rhyme import db
        from evo_rhyme.experiment_metrics import aggregate_outcome, fitness_vector_from_scores
    except ImportError as e:
        logger.error("Import failed: %s", e)
        return []

    rows = []
    for run_id in run_ids:
        run = db.get_run(run_id)
        if not run:
            continue
        candidates = db.list_candidates(run_id, gen=None, limit=500)
        # Normalize to list of dicts with fitness, scores
        cands = []
        for c in candidates:
            cands.append({
                "fitness": c.get("fitness"),
                "scores": c.get("scores") if "scores" in c else c.get("scores_json"),
            })
        agg = aggregate_outcome(cands, mode=aggregation_mode, top_k=top_k, score_key="fitness", scores_key="scores")
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


def analyze_control_impact(
    rows: List[Dict[str, Any]],
    control_keys: Optional[List[str]] = None,
    bootstrap_n: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Compute per-control effects and optional correlations. Delegates to evo_rhyme.experiment_analysis."""
    from evo_rhyme.experiment_analysis import analyze_control_impact as _analyze
    return _analyze(rows, control_keys=control_keys, bootstrap_n=bootstrap_n, ci=ci)


def report_to_markdown(report: Dict[str, Any]) -> str:
    """Produce a Markdown control effect map."""
    lines = [
        "# Control effect map",
        "",
        f"Runs: {report.get('runs', 0)}",
        "",
    ]
    # Policy progression table
    policy_perf = report.get("policy_performance") or {}
    if policy_perf:
        versions = sorted(policy_perf.keys())
        best_version = max(
            versions,
            key=lambda v: float((policy_perf.get(v) or {}).get("avg_fitness") or 0.0),
        )
        best_avg = float((policy_perf.get(best_version) or {}).get("avg_fitness") or 0.0)
        best_runs = int((policy_perf.get(best_version) or {}).get("n_runs") or 0)
        lines.append("## Policy Progression")
        lines.append("")
        lines.append(f"Best policy so far: **{best_version}** (avg_fitness={best_avg:.3f}, runs={best_runs})")
        lines.append("")
        lines.append("| policy_version | runs | avg_fitness | rhyme | flow | semantic | novelty | punchline | delta_vs_prev | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        prev = None
        for v in versions:
            item = policy_perf.get(v) or {}
            n_runs = int(item.get("n_runs") or 0)
            avg = float(item.get("avg_fitness") or 0.0)
            vec = item.get("fitness_vector_mean") or {}
            rhyme = float(vec.get("rhyme", 0.0))
            flow = float(vec.get("flow", 0.0))
            semantic = float(vec.get("semantic", 0.0))
            novelty = float(vec.get("novelty", 0.0))
            punchline = float(vec.get("punchline", 0.0))
            if prev is None:
                delta_text = "—"
                status = "baseline"
            else:
                delta = avg - prev
                delta_text = f"{delta:+.3f}"
                if delta > 0.01:
                    status = "improved"
                elif delta < -0.01:
                    status = "regressed"
                else:
                    status = "neutral"
            lines.append(
                f"| {v} | {n_runs} | {avg:.3f} | {rhyme:.3f} | {flow:.3f} | {semantic:.3f} | {novelty:.3f} | {punchline:.3f} | {delta_text} | {status} |"
            )
            prev = avg
        lines.append("")
        lines.append("Note: policy comparisons with fewer than 5 runs are low-confidence.")
        lines.append("")

    by_control = report.get("by_control") or {}
    for ck, data in by_control.items():
        lines.append(f"## {ck}")
        lines.append("")
        n_per = data.get("n_per_value") or {}
        lines.append("| Value | N |")
        lines.append("|-------|---|")
        for v, n in n_per.items():
            lines.append(f"| {v} | {n} |")
        lines.append("")
        deltas = data.get("mean_differences") or {}
        cohens = data.get("cohens_d") or {}
        if deltas:
            lines.append("### Mean difference (fitness)")
            for key, d in deltas.items():
                d_val = cohens.get(key, 0.0)
                lines.append(f"- **{key}**: Δ = {d:.4f}, Cohen's d = {d_val:.3f}")
            lines.append("")
        ci_by = data.get("ci_by_value") or {}
        if ci_by:
            lines.append("### Bootstrap 95% CI (fitness)")
            for v, ci in ci_by.items():
                lines.append(f"- **{v}**: mean = {ci.get('mean', 0):.4f}, CI = [{ci.get('ci_low', 0):.4f}, {ci.get('ci_high', 0):.4f}]")
            lines.append("")
    corr = report.get("correlations") or {}
    if corr:
        lines.append("## Correlations (numeric controls)")
        lines.append("")
        for ck, vals in corr.items():
            lines.append(f"### {ck}")
            for k, r in vals.items():
                lines.append(f"- {k}: r = {r:.4f}")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze control impact from runs")
    ap.add_argument("--experiment-id", type=int, default=None, help="Limit to runs of this experiment")
    ap.add_argument("--arm-id", type=int, default=None, help="Limit to this arm (with --experiment-id)")
    ap.add_argument("--run-ids", type=str, default=None, help="Comma-separated run IDs")
    ap.add_argument("--limit", type=int, default=100, help="Max runs to load (when not using experiment-id/run-ids)")
    ap.add_argument("--aggregation", type=str, default="best", choices=["best", "top_k_mean", "mean"], help="Outcome aggregation per run")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k for top_k_mean aggregation")
    ap.add_argument("--bootstrap-n", type=int, default=500, help="Bootstrap samples for CI")
    ap.add_argument("--output", type=str, default=None, help="Write JSON report here")
    ap.add_argument("--md", type=str, default=None, help="Write Markdown control effect map here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        if not db.db_enabled():
            logger.error("DB not enabled (RAPBOT_USE_DB, credentials)")
            return 1
    except Exception as e:
        logger.error("DB unavailable: %s", e)
        return 1

    run_ids: List[int] = []
    if args.run_ids:
        run_ids = [int(x.strip()) for x in args.run_ids.split(",") if x.strip()]
    elif args.experiment_id is not None:
        run_list = db.list_runs_for_experiment(args.experiment_id, arm_id=args.arm_id, limit=args.limit)
        run_ids = [r["run_id"] for r in run_list]
    else:
        run_list = db.list_runs(limit=args.limit, offset=0)
        run_ids = [r["run_id"] for r in run_list]

    if not run_ids:
        logger.warning("No runs found")
        report = {"runs": 0, "controls": {}, "by_control": {}, "correlations": {}}
    else:
        logger.info("Loading outcomes for %d runs", len(run_ids))
        rows = load_run_outcomes(run_ids, aggregation_mode=args.aggregation, top_k=args.top_k)
        logger.info("Computed outcomes for %d runs", len(rows))
        report = analyze_control_impact(rows, bootstrap_n=args.bootstrap_n)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    if args.md:
        md_path = Path(args.md)
        if not md_path.is_absolute():
            md_path = ROOT / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report_to_markdown(report), encoding="utf-8")
        logger.info("Wrote %s", md_path)

    if not args.output and not args.md:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import os  # noqa: F401 - used for RAPBOT_USE_DB in main
    sys.exit(main())
