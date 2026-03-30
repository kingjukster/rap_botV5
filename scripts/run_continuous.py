#!/usr/bin/env python3
"""
Run evolution in a continuous loop with varying configs. Each run uses a different
config (theme, init, population, etc.) so data is comparable across controlled sweeps.

Usage:
    python scripts/run_continuous.py
    python scripts/run_continuous.py --config config/continuous_runs.yaml
    python scripts/run_continuous.py --config config/continuous_runs.yaml --start-at 3  # skip first 3 configs
    python scripts/run_continuous.py --use-docker
    python scripts/run_continuous.py --pause 30
    python scripts/run_continuous.py --parallel 2  # 2 runs in flight for throughput

Docker-only (no host Python/DB access):
    docker compose --profile evolution run --rm evolution python scripts/run_continuous.py --no-docker --config config/continuous_runs.yaml
    docker compose --profile evolution run --rm evolution python scripts/run_continuous.py --no-docker --bootstrap-policy  # use --no-docker so evolution runs inline

Bootstrap learned policy (one-time, populates top_configs from DB):
    docker compose --profile evolution run --rm evolution python scripts/update_learned_policy.py --force --limit 300

Config file (YAML) format:
    runs:
      - arm: theme_pressure    # short id for comparison
        theme: pressure,mask,survival
        population: 60
        generations: 20
        scheme: AABB
        init: mixed
      - arm: theme_crown
        ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
_DB_MODULE: Optional[Any] = None

# Built-in defaults if no config file (lm_budget=0 = no OpenAI calls)
# seed_from_archive=10: seed each run with top-10 verses from prior completed runs
DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {"arm": "pressure", "theme": "pressure,mask,survival", "population": 60, "generations": 20, "scheme": "AABB", "init": "mixed", "lm_budget": 0, "line_lm_budget": 0, "seed_from_archive": 10},
    {"arm": "crown", "theme": "crown,empire,power", "population": 60, "generations": 20, "scheme": "AABB", "init": "mixed", "lm_budget": 0, "line_lm_budget": 0, "seed_from_archive": 10},
    {"arm": "flow", "theme": "flow,show,dream", "population": 60, "generations": 20, "scheme": "AABB", "init": "mixed", "lm_budget": 0, "line_lm_budget": 0, "seed_from_archive": 10},
    {"arm": "random", "theme": "pressure,mask,survival", "population": 60, "generations": 20, "scheme": "AABB", "init": "random", "lm_budget": 0, "line_lm_budget": 0, "seed_from_archive": 10},
    {"arm": "template", "theme": "pressure,mask,survival", "population": 60, "generations": 20, "scheme": "AABB", "init": "template", "lm_budget": 0, "line_lm_budget": 0, "seed_from_archive": 10},
]


def load_configs(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load run configs from YAML. Returns (configs, metadata) with metadata containing seeds, policy_mode."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML required for --config. Install: pip install pyyaml")
        return [], {}
    if not path.exists():
        logger.error("Config file not found: %s", path)
        return [], {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])
    continuous_experiment_id = data.get("continuous_experiment_id")
    continuous_arm_id = data.get("continuous_arm_id")
    elite_replay_schemes_raw = data.get("elite_replay_schemes")
    if isinstance(elite_replay_schemes_raw, str):
        elite_replay_schemes = [s.strip().upper() for s in elite_replay_schemes_raw.split(",") if s.strip()]
    elif isinstance(elite_replay_schemes_raw, list):
        elite_replay_schemes = [str(s).strip().upper() for s in elite_replay_schemes_raw if str(s).strip()]
    else:
        elite_replay_schemes = []
    default_lm = int(data.get("lm_budget", 0))
    default_line_lm = int(data.get("line_lm_budget", 0))
    default_seed_from_archive = int(data.get("seed_from_archive", 0))
    default_archive_mode = data.get("archive_mode", "compact_style")
    default_immigrants = int(data.get("immigrants", 20))
    default_novelty_weight = data.get("novelty_weight")
    default_early_stop_stagnant_gens = data.get("early_stop_stagnant_gens")
    default_early_stop_min_delta = data.get("early_stop_min_delta")
    default_min_coherence = data.get("min_coherence")
    default_min_rhyme_scheme_score = data.get("min_rhyme_scheme_score")
    default_operator_db_weights = bool(data.get("operator_db_weights", False))
    default_seed_max_per_source_run = data.get("seed_max_per_source_run")
    default_no_seed_diversify = bool(data.get("no_seed_diversify", False))
    default_curriculum_tighten_on_stagnation = bool(
        data.get("curriculum_tighten_on_stagnation", True)
    )
    default_curriculum_tighten_after_stagnant_gens = data.get(
        "curriculum_tighten_after_stagnant_gens"
    )
    default_curriculum_tighten_fluency_step = data.get("curriculum_tighten_fluency_step")
    default_curriculum_tighten_coherence_step = data.get("curriculum_tighten_coherence_step")
    default_curriculum_tighten_fluency_cap = data.get("curriculum_tighten_fluency_cap")
    default_curriculum_tighten_coherence_cap = data.get("curriculum_tighten_coherence_cap")
    seeds = data.get("seeds")
    if seeds is None:
        seeds = []
    elif isinstance(seeds, int):
        seeds = [seeds]
    policy_mode = data.get("policy_mode", "learned")
    out = []
    for r in runs:
        c = {
            "arm": r.get("arm", "unknown"),
            "theme": r.get("theme", "general"),
            "population": int(r.get("population", 60)),
            "generations": int(r.get("generations", 20)),
            "scheme": r.get("scheme", "AABB"),
            "init": r.get("init", "mixed"),
            "lm_budget": int(r.get("lm_budget", default_lm)),
            "line_lm_budget": int(r.get("line_lm_budget", default_line_lm)),
            "seed_from_archive": int(r.get("seed_from_archive", default_seed_from_archive)),
            "archive_mode": r.get("archive_mode", default_archive_mode),
            "immigrants": int(r.get("immigrants", default_immigrants)),
        }
        if r.get("experiment_id") is not None:
            c["experiment_id"] = int(r["experiment_id"])
        elif continuous_experiment_id is not None:
            c["experiment_id"] = int(continuous_experiment_id)
        if r.get("arm_id") is not None:
            c["arm_id"] = int(r["arm_id"])
        elif continuous_arm_id is not None:
            c["arm_id"] = int(continuous_arm_id)
        for key, default_val in (
            ("early_stop_stagnant_gens", default_early_stop_stagnant_gens),
            ("early_stop_min_delta", default_early_stop_min_delta),
            ("min_coherence", default_min_coherence),
            ("min_rhyme_scheme_score", default_min_rhyme_scheme_score),
            ("operator_db_weights", default_operator_db_weights),
            ("seed_max_per_source_run", default_seed_max_per_source_run),
            ("novelty_weight", default_novelty_weight),
        ):
            val = r.get(key, default_val)
            if val is not None:
                c[key] = val
        c["no_seed_diversify"] = bool(
            r.get("no_seed_diversify", default_no_seed_diversify)
        )
        c["curriculum_tighten_on_stagnation"] = bool(
            r.get(
                "curriculum_tighten_on_stagnation",
                default_curriculum_tighten_on_stagnation,
            )
        )
        for ck, default_val in (
            (
                "curriculum_tighten_after_stagnant_gens",
                default_curriculum_tighten_after_stagnant_gens,
            ),
            ("curriculum_tighten_fluency_step", default_curriculum_tighten_fluency_step),
            ("curriculum_tighten_coherence_step", default_curriculum_tighten_coherence_step),
            ("curriculum_tighten_fluency_cap", default_curriculum_tighten_fluency_cap),
            ("curriculum_tighten_coherence_cap", default_curriculum_tighten_coherence_cap),
        ):
            val = r.get(ck, default_val)
            if val is not None:
                c[ck] = val
        if seeds:
            for seed in seeds:
                cc = dict(c)
                cc["seed"] = seed
                out.append(cc)
        else:
            out.append(c)
    elite_replay_fraction = float(data.get("elite_replay_fraction", 0.3))
    epsilon_decay_factor = float(data.get("epsilon_decay_factor", 0.98))
    epsilon_decay_cap = int(data.get("epsilon_decay_cap", 50))
    default_generations = max(int(r.get("generations", 20)) for r in runs) if runs else 40
    metadata = {
        "seeds": seeds,
        "policy_mode": policy_mode,
        "elite_replay_fraction": elite_replay_fraction,
        "epsilon_decay_factor": epsilon_decay_factor,
        "epsilon_decay_cap": epsilon_decay_cap,
        "default_lm_budget": default_lm,
        "default_line_lm_budget": default_line_lm,
        "default_seed_from_archive": default_seed_from_archive,
        "default_generations": default_generations,
        "default_archive_mode": default_archive_mode,
        "default_immigrants": default_immigrants,
        "default_novelty_weight": default_novelty_weight,
        "elite_replay_schemes": elite_replay_schemes,
        "continuous_experiment_id": continuous_experiment_id,
        "continuous_arm_id": continuous_arm_id,
    }
    return out, metadata


def _clamp_lm_budgets_for_continuous(
    proposed_lm: int,
    proposed_line_lm: int,
    *,
    default_lm_budget: int,
    default_line_lm_budget: int,
) -> tuple[int, int]:
    """Honor continuous YAML LM defaults: if both are 0, never enable LM from DB/model."""
    if default_lm_budget == 0 and default_line_lm_budget == 0:
        return 0, 0
    return (
        max(int(proposed_lm), int(default_lm_budget)),
        max(int(proposed_line_lm), int(default_line_lm_budget)),
    )


def _sample_elite_config(
    learned_policy_path: Path,
    default_lm_budget: int = 0,
    default_line_lm_budget: int = 0,
    default_generations: int = 40,
    default_archive_mode: str = "compact_style",
    default_immigrants: int = 20,
    *,
    default_seed_from_archive: int = 0,
    elite_replay_schemes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sample a config from learned policy top_configs (score-weighted).
    Returns config dict compatible with run_evolution(), or None if no viable configs.
    YAML-level defaults are enforced as floors over stale policy values.
    """
    if not learned_policy_path.exists():
        return None
    try:
        data = json.loads(learned_policy_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    top_configs = data.get("top_configs")
    if not isinstance(top_configs, list) or not top_configs:
        return None
    allowed_schemes = None
    if elite_replay_schemes:
        allowed_schemes = {str(s).strip().upper() for s in elite_replay_schemes if str(s).strip()}
    viable = []
    for i, item in enumerate(top_configs):
        if not isinstance(item, dict):
            continue
        if allowed_schemes:
            ctrls_preview = item.get("controls") or {}
            sch = str(
                ctrls_preview.get("scheme")
                or ctrls_preview.get("rhyme_scheme")
                or "AABB",
            ).strip().upper()
            if sch not in allowed_schemes:
                continue
        viable.append((i, item))
    weights = []
    indices = []
    for i, item in viable:
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score <= 0:
            continue
        weights.append(score)
        indices.append((i, item))
    if not indices:
        return None
    idx_in_viable = random.choices(range(len(indices)), weights=weights, k=1)[0]
    _, chosen = indices[idx_in_viable]
    ctrls = chosen.get("controls") or {}
    if not isinstance(ctrls, dict):
        return None
    theme = ctrls.get("theme") or "pressure,mask,survival"
    if isinstance(theme, list):
        theme = ",".join(str(t) for t in theme)
    lm_raw = int(ctrls.get("lm_budget", 0))
    line_lm_raw = int(ctrls.get("line_lm_budget", 0))
    lm_budget, line_lm_budget = _clamp_lm_budgets_for_continuous(
        lm_raw,
        line_lm_raw,
        default_lm_budget=default_lm_budget,
        default_line_lm_budget=default_line_lm_budget,
    )
    cfg = {
        "arm": ctrls.get("arm") or f"elite_replay_{chosen.get('source_run_id', 'unknown')[:12]}",
        "theme": str(theme),
        "population": int(ctrls.get("population", 80)),
        "generations": max(int(ctrls.get("generations", 20)), default_generations),
        "scheme": str(ctrls.get("scheme", "AABB")),
        "init": str(ctrls.get("init", "mixed")),
        "lm_budget": lm_budget,
        "line_lm_budget": line_lm_budget,
        "archive_mode": default_archive_mode,
        "immigrants": default_immigrants,
    }
    if ctrls.get("seed") is not None:
        cfg["seed"] = int(ctrls["seed"])
    cfg["seed_from_archive"] = int(default_seed_from_archive)
    cfg["config_source"] = "elite_replay"
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evolution continuously with varying configs for comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML config file with runs (default: config/continuous_runs.yaml or built-in)",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        metavar="N",
        help="Start at config index N (default: 0)",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Force Docker evolution container",
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
        help="Pause between runs in seconds (default: 0)",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["static", "learned", "explore_mix"],
        default=None,
        help="Policy mode: static, learned, or explore_mix (default: from config or learned)",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Disable policy learning (use static defaults)",
    )
    parser.add_argument(
        "--update-policy-every",
        type=int,
        default=30,
        metavar="N",
        help="Call update_learned_policy.py every N completed runs (0=never)",
    )
    parser.add_argument(
        "--elite-replay-fraction",
        type=float,
        default=None,
        metavar="F",
        help="Probability of sampling config from top_configs (0=disabled, default: 0.3 or from config)",
    )
    parser.add_argument(
        "--policy-failure-penalty",
        type=float,
        default=0.8,
        metavar="F",
        help="Failure penalty for update_learned_policy (default: 0.8)",
    )
    parser.add_argument(
        "--epsilon-decay-factor",
        type=float,
        default=None,
        metavar="F",
        help="Epsilon decay per run for explore_mix (default: 0.98 or from config)",
    )
    parser.add_argument(
        "--epsilon-decay-cap",
        type=int,
        default=50,
        metavar="N",
        help="Runs after which epsilon decay stops (default: 50)",
    )
    parser.add_argument(
        "--bootstrap-policy",
        action="store_true",
        help="Run update_learned_policy.py once before the main loop (populates top_configs from DB)",
    )
    parser.add_argument(
        "--bootstrap-if-empty",
        action="store_true",
        help="Auto-run policy update before first run when policy has no top_configs (requires update-policy-every > 0)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Max evolution runs in flight (default: 1). Increase for throughput.",
    )
    parser.add_argument(
        "--restart-on-plateau",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive restart runs when recent completed-run fitness plateaus (default: on).",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=5,
        metavar="N",
        help="Completed runs window used to detect plateaus (default: 5).",
    )
    parser.add_argument(
        "--plateau-min-improvement",
        type=float,
        default=0.005,
        metavar="F",
        help="Minimum latest-run best-fitness improvement over prior window max (default: 0.005).",
    )
    parser.add_argument(
        "--plateau-restart-runs",
        type=int,
        default=2,
        metavar="N",
        help="How many restart-tuned runs to inject after a plateau signal (default: 2).",
    )
    return parser.parse_args()


def _build_evolution_cmd(
    cfg: Dict[str, Any],
    use_docker: bool,
    *,
    policy_mode: str = "static",
    learned_policy_path: Optional[Path] = None,
    epsilon: Optional[float] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Build (cmd, env) for run_verse_qd.py. Shared by run_evolution and run_evolution_async."""
    cmd = (
        [
            "docker",
            "compose",
            "--profile",
            "evolution",
            "run",
            "--rm",
            "evolution",
            "python",
            "scripts/run_verse_qd.py",
        ]
        if use_docker
        else [sys.executable, str(ROOT / "scripts" / "run_verse_qd.py")]
    )
    cmd.extend([
        "--theme", cfg["theme"],
        "--population", str(cfg["population"]),
        "--generations", str(cfg["generations"]),
        "--scheme", cfg["scheme"],
        "--init", cfg["init"],
        "--lm-budget", str(cfg.get("lm_budget", 0)),
        "--line-lm-budget", str(cfg.get("line_lm_budget", 0)),
        "--prompt-llm-fraction", "0",
        "--db",
        "--runs-dir",
        "--archive-mode", cfg.get("archive_mode", "compact_style"),
        "--immigrants", str(cfg.get("immigrants", 20)),
    ])
    if cfg.get("novelty_weight") is not None:
        cmd.extend(["--novelty-weight", str(float(cfg["novelty_weight"]))])
    seed_from_archive = cfg.get("seed_from_archive", 0)
    if seed_from_archive and int(seed_from_archive) > 0:
        cmd.extend(["--seed-from-archive", str(int(seed_from_archive))])
    if cfg.get("arm"):
        cmd.extend(["--arm", str(cfg["arm"])])
    if cfg.get("seed") is not None:
        cmd.extend(["--seed", str(cfg["seed"])])
    if cfg.get("experiment_id") is not None:
        cmd.extend(["--experiment-id", str(int(cfg["experiment_id"]))])
    if cfg.get("arm_id") is not None:
        cmd.extend(["--arm-id", str(int(cfg["arm_id"]))])
    if cfg.get("early_stop_stagnant_gens") is not None:
        cmd.extend(["--early-stop-stagnant-gens", str(int(cfg["early_stop_stagnant_gens"]))])
    if cfg.get("early_stop_min_delta") is not None:
        cmd.extend(["--early-stop-min-delta", str(float(cfg["early_stop_min_delta"]))])
    if cfg.get("min_coherence") is not None:
        cmd.extend(["--min-coherence", str(float(cfg["min_coherence"]))])
    if cfg.get("min_rhyme_scheme_score") is not None:
        cmd.extend(["--min-rhyme-scheme-score", str(float(cfg["min_rhyme_scheme_score"]))])
    if cfg.get("operator_db_weights"):
        cmd.append("--operator-db-weights")
    if cfg.get("seed_max_per_source_run") is not None:
        cmd.extend(["--seed-max-per-source-run", str(int(cfg["seed_max_per_source_run"]))])
    if cfg.get("no_seed_diversify"):
        cmd.append("--no-seed-diversify")
    if "curriculum_tighten_on_stagnation" in cfg and not cfg.get("curriculum_tighten_on_stagnation"):
        cmd.append("--no-curriculum-tighten-on-stagnation")
    for ck, flag in (
        ("curriculum_tighten_after_stagnant_gens", "--curriculum-tighten-after-stagnant-gens"),
        ("curriculum_tighten_fluency_step", "--curriculum-tighten-fluency-step"),
        ("curriculum_tighten_coherence_step", "--curriculum-tighten-coherence-step"),
        ("curriculum_tighten_fluency_cap", "--curriculum-tighten-fluency-cap"),
        ("curriculum_tighten_coherence_cap", "--curriculum-tighten-coherence-cap"),
    ):
        if cfg.get(ck) is not None:
            cmd.extend([flag, str(cfg[ck])])
    if policy_mode in ("learned", "explore_mix") and learned_policy_path and learned_policy_path.exists():
        cmd.extend([
            "--policy-mode", policy_mode,
            "--learned-policy-path", str(learned_policy_path),
        ])
    if epsilon is not None:
        cmd.extend(["--epsilon", f"{epsilon:.4f}"])
    env = os.environ.copy()
    env["RAPBOT_USE_DB"] = "1"
    if cfg.get("lm_budget", 0) == 0 and cfg.get("line_lm_budget", 0) == 0:
        env["RAPBOT_DISABLE_LM_REWRITER"] = "1"
    return cmd, env


def run_evolution(
    cfg: Dict[str, Any],
    use_docker: bool,
    *,
    policy_mode: str = "static",
    learned_policy_path: Optional[Path] = None,
    epsilon: Optional[float] = None,
) -> int:
    """Run evolution with given config. Returns exit code."""
    cmd, env = _build_evolution_cmd(
        cfg, use_docker,
        policy_mode=policy_mode,
        learned_policy_path=learned_policy_path,
        epsilon=epsilon,
    )
    logger.info(
        "Config arm=%s theme=%s pop=%d gens=%d lm_budget=%d scheme=%s init=%s",
        cfg.get("arm", "?"),
        cfg["theme"],
        cfg["population"],
        cfg["generations"],
        cfg.get("lm_budget", 0),
        cfg["scheme"],
        cfg["init"],
    )
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return proc.returncode


def run_evolution_async(
    cfg: Dict[str, Any],
    use_docker: bool,
    *,
    policy_mode: str = "static",
    learned_policy_path: Optional[Path] = None,
    epsilon: Optional[float] = None,
) -> subprocess.Popen:
    """Start evolution asynchronously. Returns Popen (caller must reap)."""
    cmd, env = _build_evolution_cmd(
        cfg, use_docker,
        policy_mode=policy_mode,
        learned_policy_path=learned_policy_path,
        epsilon=epsilon,
    )
    logger.info(
        "Config arm=%s theme=%s pop=%d gens=%d lm_budget=%d scheme=%s init=%s",
        cfg.get("arm", "?"),
        cfg["theme"],
        cfg["population"],
        cfg["generations"],
        cfg.get("lm_budget", 0),
        cfg["scheme"],
        cfg["init"],
    )
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_update_learned_policy(
    use_docker: bool,
    failure_penalty: float,
    *,
    limit: int = 300,
    force: bool = False,
) -> int:
    """
    Run update_learned_policy.py. Uses Docker when use_docker so DB is reachable.
    Returns exit code.
    """
    if use_docker:
        cmd = [
            "docker", "compose", "--profile", "evolution", "run", "--rm", "evolution",
            "python", "scripts/update_learned_policy.py",
            "--failure-penalty", str(failure_penalty),
            "--limit", str(limit),
        ]
        if force:
            cmd.append("--force")
        proc = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "RAPBOT_USE_DB": "1"})
        return proc.returncode
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "update_learned_policy.py"),
            "--failure-penalty", str(failure_penalty),
            "--limit", str(limit),
        ]
        if force:
            cmd.append("--force")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, "RAPBOT_USE_DB": "1"},
        )
        return proc.returncode


def _propose_model_config(
    default_lm_budget: int = 0,
    default_line_lm_budget: int = 0,
    default_generations: int = 40,
    default_archive_mode: str = "compact_style",
    default_immigrants: int = 20,
    *,
    default_seed_from_archive: int = 0,
) -> Optional[Dict[str, Any]]:
    """Use the control model to propose a config by training on recent DB runs.

    Returns a run config dict or None if the model can't propose (too few runs, low R^2).
    YAML-level defaults are enforced as floors over model-proposed values.
    """
    try:
        import os
        os.environ.setdefault("RAPBOT_USE_DB", "1")
        from evo_rhyme import db
        from evo_rhyme.control_model import propose_config_from_model

        if not db.db_enabled():
            return None

        runs = db.list_runs(limit=300, status_filter="completed")
        if len(runs) < 20:
            logger.debug("Too few completed runs (%d) for control model", len(runs))
            return None

        rows = []
        for r in runs:
            cfg = r.get("config_json") or {}
            if not isinstance(cfg, dict):
                continue
            fitness = None
            derived = r.get("derived")
            if isinstance(derived, dict):
                fitness = derived.get("final_best_fitness")
            if fitness is None:
                continue
            rows.append({"controls": cfg, "fitness": float(fitness)})

        if len(rows) < 20:
            return None

        proposals = propose_config_from_model(rows, n_proposals=30, top_k=1)
        if not proposals:
            return None

        chosen = proposals[0]
        ctrls = chosen.get("controls", {})
        theme = ctrls.get("theme") or ctrls.get("theme_keywords") or "pressure,mask,survival"
        if isinstance(theme, list):
            theme = ",".join(str(t) for t in theme)

        lm_raw = int(
            ctrls.get("lm_budget", ctrls.get("lm_mutation_budget_per_gen", 0))
        )
        line_lm_raw = int(
            ctrls.get("line_lm_budget", ctrls.get("line_lm_mutation_budget", 0))
        )
        lm_budget, line_lm_budget = _clamp_lm_budgets_for_continuous(
            lm_raw,
            line_lm_raw,
            default_lm_budget=default_lm_budget,
            default_line_lm_budget=default_line_lm_budget,
        )
        cfg = {
            "arm": f"model_proposed_{chosen.get('predicted_fitness', 0):.2f}",
            "theme": str(theme),
            "population": int(ctrls.get("population", ctrls.get("population_size", 80))),
            "generations": max(int(ctrls.get("generations", ctrls.get("num_generations", 20))), default_generations),
            "scheme": str(ctrls.get("scheme", ctrls.get("rhyme_scheme", "AABB"))),
            "init": str(ctrls.get("init", "mixed")),
            "lm_budget": lm_budget,
            "line_lm_budget": line_lm_budget,
            "config_source": "model_proposed",
            "seed_from_archive": int(default_seed_from_archive),
            "archive_mode": default_archive_mode,
            "immigrants": default_immigrants,
        }
        logger.info(
            "Control model proposed config: predicted_fitness=%.4f R^2=%.3f",
            chosen.get("predicted_fitness", 0),
            chosen.get("cv_r2", 0),
        )
        return cfg
    except Exception as e:
        logger.debug("Control model proposal failed: %s", e)
        return None


def _policy_has_top_configs(learned_policy_path: Path) -> bool:
    """Return True if learned_policy.json has non-empty top_configs."""
    if not learned_policy_path.exists():
        return False
    try:
        data = json.loads(learned_policy_path.read_text(encoding="utf-8"))
        top = data.get("top_configs")
        return isinstance(top, list) and len(top) > 0
    except Exception:
        return False


def _load_db_module() -> Optional[Any]:
    """Load evo_rhyme.db directly so we can read run metrics without heavy imports."""
    global _DB_MODULE
    if _DB_MODULE is not None:
        return _DB_MODULE
    db_path = ROOT / "evo_rhyme" / "db.py"
    if not db_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("evo_rhyme.db", db_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DB_MODULE = mod
        return mod
    except Exception as e:
        logger.debug("Failed to load evo_rhyme.db directly: %s", e)
        return None


def _latest_completed_best_fitness() -> Optional[float]:
    """
    Return best fitness from the most recently completed run.
    Uses run_derived when present; falls back to last generation best.
    """
    db = _load_db_module()
    if db is None:
        return None
    try:
        if not getattr(db, "db_enabled", lambda: False)():
            return None
        runs = getattr(db, "list_runs")(limit=5, offset=0, status_filter="completed")
        if not runs:
            return None
        for run in runs:
            derived = run.get("derived") or {}
            val = derived.get("final_best_fitness")
            if val is not None:
                return float(val)
            rid = run.get("run_id")
            if rid is not None:
                gens = getattr(db, "list_generations")(int(rid))
                if gens:
                    last = gens[-1]
                    if last.get("best_fitness") is not None:
                        return float(last["best_fitness"])
    except Exception as e:
        logger.debug("Could not fetch latest completed best_fitness: %s", e)
    return None


def _make_plateau_restart_cfg(base_cfg: Dict[str, Any], run_num: int) -> Dict[str, Any]:
    """
    Deterministic restart-tuned variant of a config.
    This biases toward exploration and quicker feedback while preserving reproducibility.
    """
    cfg = dict(base_cfg)
    cfg["config_source"] = "plateau_restart"
    cfg["init"] = "mixed"
    cfg["scheme"] = "AABB"
    # Shorter probe-like restart to avoid spending full budget during stagnation.
    gens = int(cfg.get("generations", 20))
    cfg["generations"] = max(10, int(round(gens * 0.75)))
    # Increase fresh material and cross-run seeds during restarts.
    cfg["immigrants"] = min(50, int(cfg.get("immigrants", 20)) + 10)
    cfg["seed_from_archive"] = max(3, int(cfg.get("seed_from_archive", 0)))
    # Ensure deterministic seed assignment for restart runs.
    if cfg.get("seed") is not None:
        cfg["seed"] = int(cfg["seed"]) + 100000 + run_num
    else:
        cfg["seed"] = (run_num * 9973) % 2147483647
    arm = str(cfg.get("arm", "unknown"))
    cfg["arm"] = f"{arm}_restart"
    return cfg


def main() -> int:
    args = parse_args()

    configs: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    if args.config:
        p = Path(args.config) if os.path.isabs(args.config) else ROOT / args.config
        configs, metadata = load_configs(p)
    else:
        default_path = ROOT / "config" / "continuous_runs.yaml"
        if default_path.exists():
            configs, metadata = load_configs(default_path)
    if not configs:
        configs = DEFAULT_CONFIGS
        logger.info("Using built-in configs (%d)", len(configs))
    else:
        seeds = metadata.get("seeds", [])
        if seeds:
            logger.info("Using seeds %s (%d configs total)", seeds, len(configs))

    policy_mode = "static"
    if not getattr(args, "no_policy", False):
        policy_mode = args.policy_mode or metadata.get("policy_mode", "learned")
    learned_policy_path = ROOT / "artifacts" / "learned_policy.json"
    update_policy_every = getattr(args, "update_policy_every", 0)
    elite_replay_fraction = (
        args.elite_replay_fraction
        if args.elite_replay_fraction is not None
        else metadata.get("elite_replay_fraction", 0.3)
    )
    epsilon_decay_factor = (
        args.epsilon_decay_factor
        if args.epsilon_decay_factor is not None
        else metadata.get("epsilon_decay_factor", 0.98)
    )
    epsilon_decay_cap = getattr(args, "epsilon_decay_cap", None) or metadata.get("epsilon_decay_cap", 50)
    elite_replay_schemes = metadata.get("elite_replay_schemes") or None
    default_lm = metadata.get("default_lm_budget", 0)
    default_line_lm = metadata.get("default_line_lm_budget", 0)
    default_seed_from_archive = int(metadata.get("default_seed_from_archive", 0))
    default_gens = metadata.get("default_generations", 40)
    default_archive_mode = metadata.get("default_archive_mode", "compact_style")
    default_immigrants = metadata.get("default_immigrants", 20)
    default_novelty_weight = metadata.get("default_novelty_weight")
    policy_failure_penalty = getattr(args, "policy_failure_penalty", 0.8)

    logger.info(
        "Policy mode=%s, update every %d runs, elite_replay=%.2f, epsilon_decay=%.2f",
        policy_mode, update_policy_every or 0, elite_replay_fraction, epsilon_decay_factor,
    )

    start = max(0, min(args.start_at, len(configs) - 1))
    if start > 0:
        logger.info("Starting at config index %d", start)

    use_docker = args.use_docker
    if not args.use_docker and not args.no_docker:
        use_docker = shutil.which("docker") is not None and os.path.exists("/var/run/docker.sock")
    logger.info("Using %s", "Docker" if use_docker else "direct Python")

    # Bootstrap policy: run update_learned_policy once before main loop
    if args.bootstrap_policy:
        logger.info("Bootstrapping learned policy from DB (--bootstrap-policy)")
        rc = _run_update_learned_policy(
            use_docker, policy_failure_penalty, limit=300, force=True
        )
        if rc != 0:
            logger.warning("Policy bootstrap exited with code %d", rc)

    # Auto-bootstrap when policy is empty and we expect to use it
    if (
        args.bootstrap_if_empty
        and update_policy_every > 0
        and policy_mode in ("learned", "explore_mix")
        and not _policy_has_top_configs(learned_policy_path)
    ):
        logger.info(
            "Policy has no top_configs; auto-bootstrapping (--bootstrap-if-empty)"
        )
        rc = _run_update_learned_policy(
            use_docker, policy_failure_penalty, limit=300, force=True
        )
        if rc != 0:
            logger.warning("Policy auto-bootstrap exited with code %d", rc)

    parallel = max(1, getattr(args, "parallel", 1))
    restart_on_plateau = bool(getattr(args, "restart_on_plateau", True))
    plateau_window = max(2, int(getattr(args, "plateau_window", 5)))
    plateau_min_improvement = float(getattr(args, "plateau_min_improvement", 0.005))
    plateau_restart_runs = max(0, int(getattr(args, "plateau_restart_runs", 2)))
    run_count = 0
    completed_count = 0
    idx = start
    epsilon_base = 0.10  # default for explore_mix
    recent_completed_fitness: List[float] = []
    plateau_restart_remaining = 0

    if parallel > 1:
        logger.info("Parallel mode: up to %d runs in flight", parallel)
    if restart_on_plateau:
        logger.info(
            "Adaptive restart: window=%d min_improvement=%.4f restart_runs=%d",
            plateau_window,
            plateau_min_improvement,
            plateau_restart_runs,
        )

    # active: proc -> (run_num, cfg) for in-flight runs
    active: Dict[subprocess.Popen, Tuple[int, Dict[str, Any]]] = {}

    def _reap_finished() -> None:
        nonlocal completed_count, plateau_restart_remaining
        done = [p for p in active if p.poll() is not None]
        for proc in done:
            run_num, cfg = active.pop(proc)
            if proc.returncode == 0:
                completed_count += 1
                logger.info("Run #%d completed (arm=%s)", run_num, cfg.get("arm", "?"))
                latest_best = _latest_completed_best_fitness()
                if latest_best is not None:
                    recent_completed_fitness.append(latest_best)
                    if len(recent_completed_fitness) > (plateau_window + 1):
                        recent_completed_fitness.pop(0)
                    if restart_on_plateau and len(recent_completed_fitness) >= (plateau_window + 1):
                        prev_window = recent_completed_fitness[-(plateau_window + 1):-1]
                        latest = recent_completed_fitness[-1]
                        prev_max = max(prev_window)
                        improvement = latest - prev_max
                        logger.info(
                            "Plateau check: latest=%.4f prev_max=%.4f improvement=%.4f",
                            latest,
                            prev_max,
                            improvement,
                        )
                        if improvement < plateau_min_improvement and plateau_restart_runs > 0:
                            plateau_restart_remaining = max(
                                plateau_restart_remaining,
                                plateau_restart_runs,
                            )
                            logger.info(
                                "Plateau detected (improvement %.4f < %.4f); scheduling %d restart-tuned run(s)",
                                improvement,
                                plateau_min_improvement,
                                plateau_restart_remaining,
                            )
                if update_policy_every > 0 and completed_count % update_policy_every == 0:
                    logger.info("Updating learned policy after %d completed runs", completed_count)
                    _run_update_learned_policy(
                        use_docker,
                        policy_failure_penalty,
                        limit=300,
                        force=False,
                    )
            else:
                logger.warning(
                    "Run #%d exited with code %d (arm=%s)",
                    run_num, proc.returncode, cfg.get("arm", "?"),
                )

    def _terminate_all() -> None:
        for proc in list(active):
            if proc.poll() is None:
                logger.info("Terminating run (pid=%d)", proc.pid)
                proc.terminate()
        time.sleep(3)
        for proc in list(active):
            if proc.poll() is None:
                proc.kill()

    try:
        while True:
            _reap_finished()

            # Start new runs until we have parallel in flight
            while len(active) < parallel:
                cfg = configs[idx]
                if completed_count > 0 and completed_count % 5 == 0 and run_count > 0:
                    model_cfg = _propose_model_config(
                        default_lm_budget=default_lm,
                        default_line_lm_budget=default_line_lm,
                        default_generations=default_gens,
                        default_archive_mode=default_archive_mode,
                        default_immigrants=default_immigrants,
                        default_seed_from_archive=default_seed_from_archive,
                    )
                    if model_cfg is not None:
                        cfg = model_cfg
                elif elite_replay_fraction > 0 and random.random() < elite_replay_fraction:
                    elite_cfg = _sample_elite_config(
                        learned_policy_path,
                        default_lm_budget=default_lm,
                        default_line_lm_budget=default_line_lm,
                        default_generations=default_gens,
                        default_archive_mode=default_archive_mode,
                        default_immigrants=default_immigrants,
                        default_seed_from_archive=default_seed_from_archive,
                        elite_replay_schemes=elite_replay_schemes,
                    )
                    if elite_cfg is not None:
                        base_slot = configs[idx]
                        for k in ("experiment_id", "arm_id"):
                            if base_slot.get(k) is not None:
                                elite_cfg[k] = base_slot[k]
                        cfg = elite_cfg
                if default_novelty_weight is not None:
                    try:
                        cfg.setdefault("novelty_weight", float(default_novelty_weight))
                    except (TypeError, ValueError):
                        pass
                if restart_on_plateau and plateau_restart_remaining > 0:
                    cfg = _make_plateau_restart_cfg(cfg, run_count + 1)
                    plateau_restart_remaining -= 1

                run_count += 1
                seed_info = f" seed={cfg.get('seed')}" if cfg.get("seed") is not None else ""
                config_src = cfg.get("config_source", "yaml")
                slot_info = f" slot={len(active)+1}/{parallel}" if parallel > 1 else ""
                logger.info(
                    "Run #%d (config %d/%d, arm=%s%s, source=%s%s)",
                    run_count, idx + 1, len(configs), cfg.get("arm", "?"), seed_info, config_src, slot_info,
                )

                epsilon = None
                if policy_mode == "explore_mix" and epsilon_decay_factor < 1.0:
                    decayed = epsilon_base * (epsilon_decay_factor ** min(completed_count, epsilon_decay_cap))
                    epsilon = max(0.05, decayed)

                proc = run_evolution_async(
                    cfg, use_docker,
                    policy_mode=policy_mode,
                    learned_policy_path=learned_policy_path,
                    epsilon=epsilon,
                )
                active[proc] = (run_count, cfg)
                idx = (idx + 1) % len(configs)

                if args.pause > 0:
                    logger.info("Pausing %d seconds", args.pause)
                    time.sleep(args.pause)

            # Wait for at least one to finish (responsive sleep)
            if active:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopped after %d run(s), %d completed. Terminating in-flight runs...", run_count, completed_count)
        _terminate_all()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
