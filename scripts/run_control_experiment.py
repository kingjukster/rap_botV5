#!/usr/bin/env python3
"""
Run a causal control experiment: multiple arms (control configurations), multiple seeds per arm.

Spec (YAML/JSON):
  runner: couplet | qd
  fixed_controls: { theme: "...", population: 100, generations: 30, ... }
  treatments:  # either list of arms or single-variable sweep
    - arm_name: "elites_3"
      controls: { elites: 3 }
    - arm_name: "elites_10"
      controls: { elites: 10 }
  # OR
  variable: "elites"
  values: [3, 5, 10]
  n_seeds_per_arm: 3
  experiment_name: "elites_sweep"

Usage:
  python scripts/run_control_experiment.py --spec experiments/elites_sweep.yaml --db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def load_spec(spec_path: str) -> Dict[str, Any]:
    path = Path(spec_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Spec not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            raise RuntimeError("PyYAML required for YAML spec")
    return json.loads(text)


def build_arms(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build list of arms from spec (treatments list or variable/values sweep)."""
    fixed = spec.get("fixed_controls") or {}
    arms = []

    if "treatments" in spec:
        for t in spec["treatments"]:
            name = t.get("arm_name") or t.get("name") or json.dumps(t.get("controls", {}), sort_keys=True)[:64]
            controls = dict(fixed)
            controls.update(t.get("controls") or {})
            arms.append({"arm_name": name, "controls": controls})
    elif "variable" in spec and "values" in spec:
        var = spec["variable"]
        for v in spec["values"]:
            name = f"{var}_{v}"
            controls = dict(fixed)
            controls[var] = v
            arms.append({"arm_name": name, "controls": controls})
    else:
        arms = [{"arm_name": "baseline", "controls": dict(fixed)}]
    return arms


def controls_to_argv(controls: Dict[str, Any], runner: str) -> List[str]:
    """Map control dict to CLI argv for the given runner (couplet or qd)."""
    argv = []
    # Map snake_case keys to --kebab-case
    for k, v in controls.items():
        if v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(v))
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description="Run causal control experiment")
    ap.add_argument("--spec", required=True, help="Path to experiment spec (YAML or JSON)")
    ap.add_argument("--db", action="store_true", help="Enable DB and link runs to experiment/arm")
    ap.add_argument("--dry-run", action="store_true", help="Print arms and commands only")
    ap.add_argument("--limit-arms", type=int, default=None, help="Max arms to run (for testing)")
    ap.add_argument("--limit-seeds", type=int, default=None, help="Max seeds per arm (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = load_spec(args.spec)
    runner = (spec.get("runner") or "couplet").lower()
    if runner not in ("couplet", "qd"):
        logger.error("runner must be 'couplet' or 'qd'")
        return 1

    n_seeds = spec.get("n_seeds_per_arm", 3)
    if args.limit_seeds is not None:
        n_seeds = min(n_seeds, args.limit_seeds)
    experiment_name = spec.get("experiment_name") or spec.get("name") or Path(args.spec).stem

    arms = build_arms(spec)
    if args.limit_arms is not None:
        arms = arms[: args.limit_arms]

    if args.db:
        os.environ["RAPBOT_USE_DB"] = "1"
    try:
        from evo_rhyme import db
        db_ok = args.db and db.db_enabled()
    except Exception:
        db_ok = False

    experiment_id = None
    if db_ok:
        experiment_id = db.insert_experiment(
            experiment_name,
            description=spec.get("description"),
            mode=spec.get("mode", "single_control"),
        )
        if experiment_id <= 0:
            logger.warning("Failed to insert experiment")
            experiment_id = None
        else:
            for arm in arms:
                aid = db.insert_experiment_arm(experiment_id, arm["arm_name"], arm["controls"])
                arm["arm_id"] = aid
            logger.info("Experiment id=%s, %d arms", experiment_id, len(arms))

    manifest_dir = ROOT / "data" / "experiments" / experiment_name
    if not args.dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.jsonl"

    if runner == "couplet":
        script = ROOT / "scripts" / "run_couplet_evolution.py"
    else:
        script = ROOT / "scripts" / "run_verse_qd.py"

    if not script.exists():
        logger.error("Runner script not found: %s", script)
        return 1

    manifest_lines = []
    for arm in arms:
        arm_id = arm.get("arm_id")
        for seed_idx in range(n_seeds):
            seed = (hash(experiment_name + arm["arm_name"]) + seed_idx * 1009) % (2 ** 31)
            base_argv = [sys.executable, str(script), "--db"] if args.db else [sys.executable, str(script)]
            base_argv += ["--seed", str(seed)]
            if experiment_id is not None and arm_id is not None:
                base_argv += ["--experiment-id", str(experiment_id), "--arm-id", str(arm_id)]
            base_argv += controls_to_argv(arm["controls"], runner)

            if args.dry_run:
                logger.info("Would run: %s", " ".join(base_argv))
                continue

            logger.info("Run arm=%s seed=%s: %s", arm["arm_name"], seed, " ".join(base_argv[:8]) + "...")
            try:
                result = subprocess.run(
                    base_argv,
                    cwd=str(ROOT),
                    env={**os.environ},
                    timeout=spec.get("run_timeout_seconds") or 7200,
                )
                manifest_lines.append(json.dumps({
                    "arm_name": arm["arm_name"],
                    "arm_id": arm_id,
                    "seed": seed,
                    "exit_code": result.returncode,
                }) + "\n")
            except subprocess.TimeoutExpired:
                logger.warning("Run timed out: arm=%s seed=%s", arm["arm_name"], seed)
                manifest_lines.append(json.dumps({
                    "arm_name": arm["arm_name"],
                    "arm_id": arm_id,
                    "seed": seed,
                    "exit_code": -1,
                    "timeout": True,
                }) + "\n")

    if manifest_lines and manifest_path:
        manifest_path.write_text("".join(manifest_lines), encoding="utf-8")
        logger.info("Wrote %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
