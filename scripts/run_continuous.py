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
    default_lm = int(data.get("lm_budget", 0))
    default_line_lm = int(data.get("line_lm_budget", 0))
    default_seed_from_archive = int(data.get("seed_from_archive", 0))
    default_archive_mode = data.get("archive_mode", "compact_style")
    default_immigrants = int(data.get("immigrants", 20))
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
        "default_generations": default_generations,
        "default_archive_mode": default_archive_mode,
        "default_immigrants": default_immigrants,
    }
    return out, metadata


def _sample_elite_config(
    learned_policy_path: Path,
    default_lm_budget: int = 0,
    default_line_lm_budget: int = 0,
    default_generations: int = 40,
    default_archive_mode: str = "compact_style",
    default_immigrants: int = 20,
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
    viable = [
        (i, item)
        for i, item in enumerate(top_configs)
        if isinstance(item, dict)
    ]
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
    cfg = {
        "arm": ctrls.get("arm") or f"elite_replay_{chosen.get('source_run_id', 'unknown')[:12]}",
        "theme": str(theme),
        "population": int(ctrls.get("population", 80)),
        "generations": max(int(ctrls.get("generations", 20)), default_generations),
        "scheme": str(ctrls.get("scheme", "AABB")),
        "init": str(ctrls.get("init", "mixed")),
        "lm_budget": max(int(ctrls.get("lm_budget", 0)), default_lm_budget),
        "line_lm_budget": max(int(ctrls.get("line_lm_budget", 0)), default_line_lm_budget),
        "archive_mode": default_archive_mode,
        "immigrants": default_immigrants,
    }
    if ctrls.get("seed") is not None:
        cfg["seed"] = int(ctrls["seed"])
    cfg["seed_from_archive"] = 3
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
    seed_from_archive = cfg.get("seed_from_archive", 0)
    if seed_from_archive and int(seed_from_archive) > 0:
        cmd.extend(["--seed-from-archive", str(int(seed_from_archive))])
    if cfg.get("arm"):
        cmd.extend(["--arm", str(cfg["arm"])])
    if cfg.get("seed") is not None:
        cmd.extend(["--seed", str(cfg["seed"])])
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

        cfg = {
            "arm": f"model_proposed_{chosen.get('predicted_fitness', 0):.2f}",
            "theme": str(theme),
            "population": int(ctrls.get("population", ctrls.get("population_size", 80))),
            "generations": max(int(ctrls.get("generations", ctrls.get("num_generations", 20))), default_generations),
            "scheme": str(ctrls.get("scheme", ctrls.get("rhyme_scheme", "AABB"))),
            "init": str(ctrls.get("init", "mixed")),
            "lm_budget": max(int(ctrls.get("lm_budget", ctrls.get("lm_mutation_budget_per_gen", 0))), default_lm_budget),
            "line_lm_budget": max(int(ctrls.get("line_lm_budget", ctrls.get("line_lm_mutation_budget", 0))), default_line_lm_budget),
            "config_source": "model_proposed",
            "seed_from_archive": 3,
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
    default_lm = metadata.get("default_lm_budget", 0)
    default_line_lm = metadata.get("default_line_lm_budget", 0)
    default_gens = metadata.get("default_generations", 40)
    default_archive_mode = metadata.get("default_archive_mode", "compact_style")
    default_immigrants = metadata.get("default_immigrants", 20)
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
    run_count = 0
    completed_count = 0
    idx = start
    epsilon_base = 0.10  # default for explore_mix

    if parallel > 1:
        logger.info("Parallel mode: up to %d runs in flight", parallel)

    # active: proc -> (run_num, cfg) for in-flight runs
    active: Dict[subprocess.Popen, Tuple[int, Dict[str, Any]]] = {}

    def _reap_finished() -> None:
        nonlocal completed_count
        done = [p for p in active if p.poll() is not None]
        for proc in done:
            run_num, cfg = active.pop(proc)
            if proc.returncode == 0:
                completed_count += 1
                logger.info("Run #%d completed (arm=%s)", run_num, cfg.get("arm", "?"))
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
                    )
                    if elite_cfg is not None:
                        cfg = elite_cfg

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
