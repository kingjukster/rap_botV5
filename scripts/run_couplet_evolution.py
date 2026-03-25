#!/usr/bin/env python
"""
run_couplet_evolution.py

CLI to run couplet evolution and save results.

Usage:
    python scripts/run_couplet_evolution.py --theme "pressure,mask,survival" --population 100 --generations 30 --output results.json

Loads paths from config/settings or RAPBOT_CONFIG / RAPBOT_* env vars.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_safe_path(user_path: str, base: Path, desc: str = "path") -> Path:
    """Resolve path and ensure it stays under base (prevents path traversal)."""
    path = Path(user_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError:
        raise ValueError(f"{desc} path must be within project directory")
    return path


from evo_rhyme.constraints import passes_constraints
from evo_rhyme.evolution import EvolutionConfig, _effective_constraint_config, evolve, evolve_multiobjective
from evo_rhyme.individual import analyze_individual
from evo_rhyme.style_profile import build_style_profile, load_lines_from_file
from evo_rhyme.population import (
    RandomGenerator,
    TemplateGenerator,
    create_initial_population,
    create_mixed_population,
)
from evo_rhyme.seed_generator import SeedGenerator, load_corpus_lines


def _parse_args_with_defaults() -> argparse.Namespace:
    # Two-phase parse to allow --config to influence defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file path (YAML/JSON). Overrides defaults via config/settings.py.",
    )
    known, _ = pre.parse_known_args()

    from config.settings import get_evolution_defaults

    defaults = get_evolution_defaults(config_path=known.config)

    parser = argparse.ArgumentParser(
        description="Run couplet evolution",
        parents=[pre],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Seeds Python/NumPy/Torch when available.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Comma-separated theme keywords (e.g. pressure,mask,survival)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=int(defaults.get("population", 100)),
        help="Population size (max 5000)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=int(defaults.get("generations", 10)),
        help="Number of generations (max 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output JSON path for top candidates and config",
    )
    parser.add_argument(
        "--elites",
        type=int,
        default=int(defaults.get("elites", 5)),
        help="Number of elites per generation",
    )
    parser.add_argument(
        "--immigrants",
        type=int,
        default=int(defaults.get("immigrants", 10)),
        help="Random immigrants per generation (10-15%% of pop recommended to prevent stagnation)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Override corpus path for seed generation",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["mixed", "random", "template"],
        default=str(defaults.get("init", "mixed")),
        help="Population init: mixed (40%% template, 40%% corpus, 20%% random; or 60/40 if no corpus), random, or template",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--runs-dir",
        action="store_true",
        help="Enable run logging: write to data/evo_rhyme/runs/{timestamp}/ (config, score_history, top_candidates)",
    )
    parser.add_argument(
        "--use-embeddings",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_embeddings", False)),
        help="Use SiameseRhymeScorer for embedding-based semantic scoring (blended with keyword score)",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=float(defaults.get("embedding_weight", 0.5)),
        help="Weight of embedding score in semantic blend (default: 0.5)",
    )
    parser.add_argument(
        "--multiobjective",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("multiobjective", False)),
        help="Use Pareto multi-objective evolution (rhyme, fluency, semantic)",
    )
    parser.add_argument(
        "--use-niching",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_niching", False)),
        help="Use niching in elite selection (preserve top per rhyme family)",
    )
    parser.add_argument(
        "--min-fluency",
        type=float,
        default=defaults.get("min_fluency", None),
        metavar="FLOAT",
        help="Reject mutations with fluency below this (0=disabled). Default: 0.6 when --theme set, else 0.0.",
    )
    parser.add_argument(
        "--min-semantic",
        type=float,
        default=defaults.get("min_semantic", None),
        metavar="FLOAT",
        help="Reject mutations with semantic below this (0=disabled). Default: 0.25 when --theme set, else 0.0.",
    )
    parser.add_argument(
        "--min-lexical",
        type=float,
        default=float(defaults.get("min_lexical", 0.0)),
        metavar="FLOAT",
        help="Reject mutations with lexical_validity below this (0=disabled). Requires corpus. Try 0.5-0.6 to block nonsense.",
    )
    parser.add_argument(
        "--min-ngram",
        type=float,
        default=defaults.get("min_ngram", None),
        metavar="FLOAT",
        help="Reject mutations with ngram_fluency below this (0=disabled). Default: 0.2 when corpus available. Blocks nonsense phrase structure.",
    )
    parser.add_argument(
        "--style-corpus",
        type=str,
        default=None,
        help="Path to reference lines/couplets file for style matching (e.g. path/to/lines.txt). If not provided, style_weight is 0.",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=float(defaults.get("style_weight", 0.1)),
        help="Weight of style similarity in fitness when --style-corpus is provided (default: 0.1)",
    )
    parser.add_argument(
        "--require-theme",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("require_theme", False)),
        help="Enforce at least one theme keyword in each couplet (requires --theme)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to evolved weights JSON (from run_weight_tuner.py). Uses 'weights' key.",
    )
    parser.add_argument(
        "--lm-fluency",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("lm_fluency", False)),
        help="Blend ngram fluency with LM perplexity (DistilGPT-2) for stronger nonsense detection.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Enable MySQL persistence (RAPBOT_USE_DB=1, run/generation/candidate logging)",
    )
    parser.add_argument(
        "--lm-fluency-weight",
        type=float,
        default=float(defaults.get("lm_fluency_weight", 0.5)),
        metavar="FLOAT",
        help="Weight of LM score in ngram_fluency blend when --lm-fluency (default: 0.5).",
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        default=None,
        help="Link run to this experiment (for control experiment runner).",
    )
    parser.add_argument(
        "--arm-id",
        type=int,
        default=None,
        help="Link run to this experiment arm (for control experiment runner).",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["static", "learned", "explore_mix"],
        default=None,
        help="Control source mode: static defaults, learned policy, or epsilon-greedy explore_mix.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Exploration probability for explore_mix mode.",
    )
    parser.add_argument(
        "--learned-policy-path",
        type=str,
        default=None,
        help="Path to learned policy JSON (default from config experiments.learned_policy_path).",
    )
    return parser.parse_args(), parser


def main():
    args, parser = _parse_args_with_defaults()
    from config.settings import get_experiment_defaults
    from evo_rhyme.policy_runtime import (
        apply_controls_to_args,
        resolve_policy_controls,
        maybe_epsilon_perturb,
    )

    exp_defaults = get_experiment_defaults(config_path=getattr(args, "config", None))
    args.policy_mode = args.policy_mode or str(exp_defaults.get("policy_mode", "static"))
    args.epsilon = float(args.epsilon if args.epsilon is not None else exp_defaults.get("epsilon", 0.10))
    args.learned_policy_path = args.learned_policy_path or str(
        exp_defaults.get("learned_policy_path", "artifacts/learned_policy.json")
    )
    args._policy_source = "defaults"
    args._exploration_applied = False
    args._policy_overrides = []
    args._policy_version = None
    args._policy_hash = None
    args._sampled_policy_rank = None

    protected = {"theme", "db", "experiment_id", "arm_id", "output", "runs_dir", "config", "seed", "verbose"}
    if args.policy_mode in ("learned", "explore_mix"):
        learned_controls, pmeta = resolve_policy_controls(args.learned_policy_path, ROOT)
        src = pmeta.get("policy_source")
        args._policy_version = pmeta.get("policy_version")
        args._policy_hash = pmeta.get("policy_hash")
        args._sampled_policy_rank = pmeta.get("sampled_policy_rank")
        if learned_controls:
            overrides = []
            for key, value in learned_controls.items():
                if key in protected or not hasattr(args, key):
                    continue
                try:
                    if getattr(args, key) != value:
                        overrides.append(key)
                except Exception:
                    pass
            apply_controls_to_args(args, learned_controls, protected_keys=protected)
            args._policy_source = src
            args._policy_overrides = sorted(set(overrides))
            # Couplet evolution doesn't support init=lm; map to mixed to avoid early crash
            if getattr(args, "init", None) == "lm":
                args.init = "mixed"
                overrides = list(getattr(args, "_policy_overrides", []) or [])
                if "init" not in overrides:
                    overrides.append("init")
                args._policy_overrides = sorted(set(overrides))
            if args.policy_mode == "explore_mix":
                args._exploration_applied = maybe_epsilon_perturb(
                    args,
                    epsilon=args.epsilon,
                    protected_keys=protected,
                )

    seed_info: Dict[str, Any] | None = None
    if args.seed is not None:
        try:
            from evo_rhyme.repro import seed_everything

            seed_info = seed_everything(int(args.seed))
        except Exception:
            seed_info = {"seed": int(args.seed), "error": "seed_everything_failed"}

    if args.population > 5000:
        parser.error(f"--population {args.population} exceeds max 5000")
    if args.generations > 1000:
        parser.error(f"--generations {args.generations} exceeds max 1000")
    # Couplet evolution: population > 60 correlates with ~89% failure rate (run analysis)
    if args.population > 60:
        parser.error(
            f"--population {args.population} exceeds recommended max 60 for couplet evolution "
            "(higher values cause timeouts/OOM). Use 40-60 for reliable runs."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Policy mode=%s source=%s version=%s hash=%s sampled_rank=%s epsilon=%.3f exploration_applied=%s",
        args.policy_mode,
        getattr(args, "_policy_source", "defaults"),
        getattr(args, "_policy_version", None),
        getattr(args, "_policy_hash", None),
        getattr(args, "_sampled_policy_rank", None),
        float(args.epsilon),
        bool(getattr(args, "_exploration_applied", False)),
    )
    overrides = getattr(args, "_policy_overrides", []) or []
    if overrides:
        preview = ", ".join(overrides[:8]) + (" ..." if len(overrides) > 8 else "")
        logger.info("Policy overrides (%d): %s", len(overrides), preview)

    theme_keywords = [w.strip() for w in args.theme.split(",") if w.strip()] or None
    from config import get_elite_corpus_path
    corpus_path = Path(args.corpus) if args.corpus else get_elite_corpus_path()

    # Theme-aware defaults for min_fluency/min_semantic: when theme set, default 0.6/0.25; else 0.0 (disabled)
    min_fluency_accept = (
        args.min_fluency if args.min_fluency is not None
        else (0.6 if theme_keywords else 0.0)
    )
    min_semantic_accept = (
        args.min_semantic if args.min_semantic is not None
        else (0.25 if theme_keywords else 0.0)
    )
    min_lexical_accept = args.min_lexical

    corpus_lines = load_corpus_lines(corpus_path)
    if corpus_lines:
        corpus_lines = corpus_lines[:2000]
    # When corpus empty, still apply 0.2 floor if --lm-fluency (LM provides fluency signal)
    # to block nonsense exploitation (valid words + rhyme + theme = high score, but nonsense)
    min_ngram_fluency_accept = (
        args.min_ngram if args.min_ngram is not None
        else (0.2 if corpus_lines else (0.2 if args.lm_fluency else 0.0))
    )

    style_profile = None
    style_weight = 0.0
    if args.style_corpus:
        try:
            style_path = _resolve_safe_path(args.style_corpus, ROOT, "style-corpus")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        lines = load_lines_from_file(style_path)
        if lines:
            style_profile = build_style_profile(lines)
            style_weight = args.style_weight
            logger.info(f"Style corpus: {style_path} ({len(lines)} lines) -> style_weight={style_weight}")
        else:
            logger.warning(f"Style corpus empty or not found: {style_path}")

    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Theme: {theme_keywords or 'none'}")
    logger.info(f"Population: {args.population}, Generations: {args.generations}, Init: {args.init}")

    output_dir = None
    if args.runs_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "data" / "evo_rhyme" / "runs" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run logging to {output_dir}")

    require_theme = bool(args.require_theme and theme_keywords)
    if args.require_theme and not theme_keywords:
        logger.warning("--require-theme ignored: no --theme keywords provided")

    fitness_weights = None
    if args.weights:
        try:
            wpath = _resolve_safe_path(args.weights, ROOT, "weights")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        if wpath.exists():
            with wpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fitness_weights = data.get("weights", data)
            logger.info(f"Loaded weights from {wpath}")
        else:
            logger.warning(f"Weights file not found: {wpath}")

    run_id = None
    if args.db:
        import os
        os.environ["RAPBOT_USE_DB"] = "1"
        try:
            from evo_rhyme import db
            from evo_rhyme.experiment_controls import build_control_snapshot_from_couplet_args
            from config.settings import get_evolution_defaults
            if db.db_enabled():
                defaults = get_evolution_defaults(config_path=getattr(args, "config", None))
                control_snapshot = build_control_snapshot_from_couplet_args(
                    args,
                    defaults,
                    fitness_weights=fitness_weights,
                    mutation_weights=None,
                    seed_info=seed_info,
                )
                run_id = db.insert_run(
                    "run_couplet_evolution",
                    ",".join(theme_keywords) if theme_keywords else "",
                    control_snapshot,
                    experiment_id=args.experiment_id if getattr(args, "experiment_id", None) else None,
                    arm_id=args.arm_id if getattr(args, "arm_id", None) else None,
                )
                if run_id > 0:
                    logger.info("DB run_id=%d", run_id)
        except Exception as e:
            logger.warning("DB insert_run failed: %s", e)

    config = EvolutionConfig(
        population_size=args.population,
        num_elites=args.elites,
        random_immigrants_per_gen=args.immigrants,
        fitness_weights=fitness_weights,
        output_dir=output_dir,
        run_id=run_id if run_id and run_id > 0 else None,
        use_embeddings=args.use_embeddings,
        embedding_weight=args.embedding_weight,
        population_init=args.init,
        multiobjective=args.multiobjective,
        use_niching=args.use_niching,
        min_fluency_accept=min_fluency_accept,
        min_semantic_accept=min_semantic_accept,
        min_lexical_accept=min_lexical_accept,
        min_ngram_fluency_accept=min_ngram_fluency_accept,
        use_lm_fluency=args.lm_fluency,
        lm_fluency_weight=args.lm_fluency_weight,
        style_profile=style_profile,
        style_weight=style_weight,
        corpus_lines=corpus_lines if corpus_lines else None,
        require_theme_presence=require_theme,
    )

    generator = SeedGenerator(corpus_path=corpus_path)
    if config.population_init == "mixed":
        raw_population = create_mixed_population(
            corpus_path=corpus_path,
            theme_keywords=theme_keywords,
            size=args.population,
        )
    elif config.population_init == "template":
        raw_population = create_initial_population(
            TemplateGenerator(),
            theme_keywords=theme_keywords,
            size=args.population,
        )
    else:
        raw_population = create_initial_population(
            RandomGenerator(),
            theme_keywords=theme_keywords,
            size=args.population,
        )
    logger.info(f"Initial population: {len(raw_population)} couplets (before constraint filter)")

    prompt_kw_set = set(theme_keywords) if theme_keywords else None
    effective_constraint = _effective_constraint_config(config, prompt_kw_set)

    population = []
    for ind in raw_population:
        analyze_individual(ind)
        if passes_constraints(ind, effective_constraint):
            population.append(ind)
    max_oversample = 5
    for _ in range(max_oversample):
        if len(population) >= args.population:
            break
        if config.population_init == "mixed":
            extra = create_mixed_population(
                corpus_path=corpus_path,
                theme_keywords=theme_keywords,
                size=args.population,
            )
        elif config.population_init == "template":
            extra = TemplateGenerator().generate_seed_couplets(
                theme_keywords=theme_keywords,
                size=args.population,
            )
        else:
            extra = RandomGenerator().generate_seed_couplets(
                theme_keywords=theme_keywords,
                size=args.population,
            )
        for ind in extra:
            if len(population) >= args.population:
                break
            analyze_individual(ind)
            if passes_constraints(ind, effective_constraint):
                population.append(ind)
    population = population[:args.population]
    logger.info(f"Filtered to {len(population)} couplets passing constraints")

    if len(population) == 0:
        msg = (
            "No couplets passed constraints after oversampling. "
            "Try looser constraints, different init (mixed/random), or different theme."
        )
        if run_id and run_id > 0:
            try:
                from evo_rhyme import db
                db.update_run_status(run_id, "failed", failure_reason=msg)
            except Exception:
                pass
        raise RuntimeError(msg)

    def immigrant_gen(size: int):
        return generator.generate_seed_couplets(theme_keywords=theme_keywords, size=size)

    tracer_tok = None
    if output_dir is not None:
        from evo_rhyme.operator_telemetry import OperatorTracer, set_operator_tracer

        tracer_tok = set_operator_tracer(
            OperatorTracer(output_dir, run_id=run_id or 0)
        )

    try:
        evolve_fn = evolve_multiobjective if args.multiobjective else evolve
        population = evolve_fn(
            population,
            generations=args.generations,
            config=config,
            prompt_keywords=set(theme_keywords) if theme_keywords else None,
            immigrant_generator=immigrant_gen,
        )
    except Exception as e:
        if run_id and run_id > 0:
            try:
                from evo_rhyme import db
                reason = f"{type(e).__name__}: {str(e)}"[:4096]
                db.update_run_status(run_id, "failed", failure_reason=reason)
            except Exception:
                pass
        raise
    finally:
        if tracer_tok is not None:
            from evo_rhyme.operator_telemetry import reset_operator_tracer

            reset_operator_tracer(tracer_tok)

    if run_id and run_id > 0:
        try:
            from evo_rhyme import db
            db.update_run_status(run_id, "completed")
        except Exception as e:
            logger.warning("DB update_run_status failed: %s", e)

    # Save results
    top = population[: min(50, len(population))]
    output_data = {
        "config": {
            **({"seed_info": seed_info} if seed_info else {}),
            "theme": args.theme,
            "population": args.population,
            "generations": args.generations,
            "elites": args.elites,
            "immigrants": args.immigrants,
            "init": args.init,
        },
        "candidates": [
            {
                "line1": ind.line1,
                "line2": ind.line2,
                "fitness": ind.fitness,
                "scores": ind.scores,
            }
            for ind in top
        ],
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved top {len(top)} candidates to {out_path}")

    # Print top 5
    print("\n--- Top 5 ---")
    for i, ind in enumerate(population[:5], 1):
        print(f"{i}. [{ind.fitness:.4f}] {ind.line1} | {ind.line2}")


if __name__ == "__main__":
    main()
