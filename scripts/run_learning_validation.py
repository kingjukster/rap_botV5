#!/usr/bin/env python3
"""
Orchestrate learning-behavior validation:
1) sanity static->learned wiring
2) A/B static vs learned
3) A/B learned vs explore_mix
4) optional longitudinal cycles

Writes per-phase analysis artifacts and a manifest with verdicts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def classify_delta(delta: float, threshold: float = 0.01) -> str:
    if delta > threshold:
        return "improved"
    if delta < -threshold:
        return "regressed"
    return "neutral"


def run_cmd(argv: List[str], *, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"argv": argv, "exit_code": 0, "duration_s": 0.0, "dry_run": True}
    t0 = time.time()
    cp = subprocess.run(argv, cwd=str(ROOT), env={**os.environ})
    return {
        "argv": argv,
        "exit_code": int(cp.returncode),
        "duration_s": round(time.time() - t0, 3),
        "dry_run": False,
    }


def latest_experiment_id_by_name(name: str) -> Optional[int]:
    os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        if not db.db_enabled():
            return None
        exps = db.list_experiments(limit=200, offset=0)
        for e in exps:
            if e.get("name") == name:
                return int(e["experiment_id"])
        return None
    except Exception:
        return None


def run_analysis_for_experiment(experiment_id: int, out_dir: Path, phase_name: str, dry_run: bool = False) -> Dict[str, Any]:
    out_json = out_dir / f"{phase_name}_report.json"
    out_md = out_dir / f"{phase_name}_report.md"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_control_impact.py"),
        "--experiment-id",
        str(experiment_id),
        "--output",
        str(out_json),
        "--md",
        str(out_md),
    ]
    cmd_res = run_cmd(argv, dry_run=dry_run)
    report: Dict[str, Any] = {}
    if not dry_run and out_json.exists():
        report = json.loads(out_json.read_text(encoding="utf-8"))
    return {"command": cmd_res, "report_json": str(out_json), "report_md": str(out_md), "report": report}


def run_analysis_for_run_ids(run_ids: List[int], out_dir: Path, phase_name: str, dry_run: bool = False) -> Dict[str, Any]:
    out_json = out_dir / f"{phase_name}_report.json"
    out_md = out_dir / f"{phase_name}_report.md"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_control_impact.py"),
        "--run-ids",
        ",".join(str(x) for x in run_ids),
        "--output",
        str(out_json),
        "--md",
        str(out_md),
    ]
    cmd_res = run_cmd(argv, dry_run=dry_run)
    report: Dict[str, Any] = {}
    if not dry_run and out_json.exists():
        report = json.loads(out_json.read_text(encoding="utf-8"))
    return {"command": cmd_res, "report_json": str(out_json), "report_md": str(out_md), "report": report}


def compute_policy_mode_verdict(report: Dict[str, Any], *, baseline: str, candidate: str) -> Dict[str, Any]:
    by_control = report.get("by_control") or {}
    pm = by_control.get("policy_mode") or {}
    means = pm.get("mean_outcome_by_value") or {}
    n_per = pm.get("n_per_value") or {}
    base_avg = float((means.get(baseline) or {}).get("fitness", 0.0))
    cand_avg = float((means.get(candidate) or {}).get("fitness", 0.0))
    delta = cand_avg - base_avg
    verdict = classify_delta(delta)
    low_confidence = (int(n_per.get(baseline, 0)) < 5) or (int(n_per.get(candidate, 0)) < 5)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_avg_fitness": base_avg,
        "candidate_avg_fitness": cand_avg,
        "policy_improvement": delta,
        "verdict": verdict,
        "low_confidence": low_confidence,
        "n_per_value": n_per,
    }


def phase_specs(profile: str) -> List[Tuple[str, str]]:
    specs = [
        ("sanity", "data/experiments/sanity_static_then_learned.yaml"),
        ("ab_static_vs_learned", "data/experiments/ab_static_vs_learned.yaml"),
        ("ab_learned_vs_exploremix", "data/experiments/ab_learned_vs_exploremix.yaml"),
    ]
    return specs


def run_ids_for_experiment(experiment_id: int) -> List[int]:
    os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        if not db.db_enabled():
            return []
        runs = db.list_runs_for_experiment(experiment_id, limit=5000)
        return [int(r["run_id"]) for r in runs]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Run learning-behavior validation harness")
    ap.add_argument("--profile", choices=["quick", "standard", "longitudinal"], default="standard")
    ap.add_argument("--cycles", type=int, default=3, help="Longitudinal cycles when profile=longitudinal")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", action="store_true", default=True)
    ap.add_argument("--limit-seeds", type=int, default=None, help="Override seeds per arm for speed")
    ap.add_argument(
        "--offline-safe",
        action="store_true",
        help="Disable LM rewriter calls so validation can run without API connectivity",
    )
    ap.add_argument(
        "--lm-fail-fast",
        action="store_true",
        help="Minimize LM retry overhead by aborting each LM call after first failure",
    )
    ap.add_argument(
        "--require-openai-key",
        action="store_true",
        help="Abort early if OPENAI_API_KEY is missing (ignored in --offline-safe mode)",
    )
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "experiments" / f"learning_validation_{stamp}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.offline_safe:
        os.environ["RAPBOT_DISABLE_LM_REWRITER"] = "1"
    if args.lm_fail_fast:
        os.environ["RAPBOT_LM_FAIL_FAST"] = "1"
    if args.require_openai_key and not args.offline_safe and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing; aborting due to --require-openai-key")
        return 2

    manifest: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "runtime_modes": {
            "offline_safe": bool(args.offline_safe),
            "lm_fail_fast": bool(args.lm_fail_fast),
            "require_openai_key": bool(args.require_openai_key),
            "disable_lm_env": os.environ.get("RAPBOT_DISABLE_LM_REWRITER", ""),
            "lm_fail_fast_env": os.environ.get("RAPBOT_LM_FAIL_FAST", ""),
        },
        "phases": [],
        "final": {},
    }

    os.environ["RAPBOT_USE_DB"] = "1"

    # Run core three levels
    analyzed_experiment_ids: List[int] = []
    for phase_name, spec_rel in phase_specs(args.profile):
        spec_path = ROOT / spec_rel
        exp_name = Path(spec_rel).stem
        argv = [sys.executable, str(ROOT / "scripts" / "run_control_experiment.py"), "--spec", str(spec_path), "--db"]
        if args.dry_run:
            argv.append("--dry-run")
        if args.profile == "quick":
            argv += ["--limit-seeds", "2"]
        elif args.limit_seeds is not None:
            argv += ["--limit-seeds", str(args.limit_seeds)]
        run_res = run_cmd(argv, dry_run=args.dry_run)

        # Update learned policy between phases (except after last of core three)
        upd_res = {"skipped": False}
        if phase_name != "ab_learned_vs_exploremix":
            upd_argv = [sys.executable, str(ROOT / "scripts" / "update_learned_policy.py")]
            upd_res = run_cmd(upd_argv, dry_run=args.dry_run)

        experiment_id = latest_experiment_id_by_name(exp_name) if not args.dry_run else None
        analysis_res = (
            run_analysis_for_experiment(experiment_id, out_dir, phase_name, dry_run=args.dry_run)
            if experiment_id
            else {"report": {}, "command": {"skipped": True}, "report_json": None, "report_md": None}
        )
        if experiment_id:
            analyzed_experiment_ids.append(experiment_id)

        verdict = None
        if phase_name == "ab_static_vs_learned" and analysis_res.get("report"):
            verdict = compute_policy_mode_verdict(analysis_res["report"], baseline="static", candidate="learned")
        elif phase_name == "ab_learned_vs_exploremix" and analysis_res.get("report"):
            verdict = compute_policy_mode_verdict(analysis_res["report"], baseline="learned", candidate="explore_mix")

        manifest["phases"].append(
            {
                "phase": phase_name,
                "spec": str(spec_path),
                "experiment_name": exp_name,
                "experiment_id": experiment_id,
                "run_command": run_res,
                "policy_update_command": upd_res,
                "analysis": analysis_res.get("command"),
                "analysis_report_json": analysis_res.get("report_json"),
                "analysis_report_md": analysis_res.get("report_md"),
                "verdict": verdict,
            }
        )

    # Longitudinal cycles
    if args.profile == "longitudinal":
        longitudinal = []
        for i in range(max(1, args.cycles)):
            phase_name = f"longitudinal_cycle_{i+1}"
            spec_rel = "data/experiments/ab_static_vs_learned.yaml"
            spec_path = ROOT / spec_rel
            exp_name = Path(spec_rel).stem
            run_argv = [sys.executable, str(ROOT / "scripts" / "run_control_experiment.py"), "--spec", str(spec_path), "--db"]
            if args.dry_run:
                run_argv.append("--dry-run")
            run_res = run_cmd(run_argv, dry_run=args.dry_run)
            upd_res = run_cmd([sys.executable, str(ROOT / "scripts" / "update_learned_policy.py")], dry_run=args.dry_run)
            experiment_id = latest_experiment_id_by_name(exp_name) if not args.dry_run else None
            analysis_res = (
                run_analysis_for_experiment(experiment_id, out_dir, phase_name, dry_run=args.dry_run)
                if experiment_id
                else {"report": {}}
            )
            if experiment_id:
                analyzed_experiment_ids.append(experiment_id)
            longitudinal.append(
                {
                    "phase": phase_name,
                    "experiment_id": experiment_id,
                    "run_command": run_res,
                    "policy_update_command": upd_res,
                    "analysis_report_json": analysis_res.get("report_json"),
                }
            )
        manifest["final"]["longitudinal"] = longitudinal

    # Final aggregate report across all executed experiments
    all_run_ids: List[int] = []
    if not args.dry_run:
        for exp_id in analyzed_experiment_ids:
            all_run_ids.extend(run_ids_for_experiment(exp_id))
        all_run_ids = sorted(set(all_run_ids))
    if all_run_ids:
        aggregate = run_analysis_for_run_ids(all_run_ids, out_dir, "aggregate", dry_run=args.dry_run)
        manifest["final"]["aggregate"] = {
            "n_experiments": len(set(analyzed_experiment_ids)),
            "n_runs": len(all_run_ids),
            "analysis": aggregate.get("command"),
            "analysis_report_json": aggregate.get("report_json"),
            "analysis_report_md": aggregate.get("report_md"),
        }

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        out_file = out_dir / "validation_manifest.json"
        out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {out_file}")
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
