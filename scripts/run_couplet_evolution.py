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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _load_corpus_path():
    """Load elite corpus path from config/settings or env. Falls back to phaseA_kaggle_verse.txt if primary doesn't exist."""
    path = None
    try:
        from config.settings import load_settings
        settings = load_settings()
        path = Path(settings.elite_corpus_path)
        if not path.is_absolute():
            path = ROOT / path
    except Exception:
        pass
    if path is None:
        env_path = __import__("os").environ.get("RAPBOT_ELITE_CORPUS")
        if env_path:
            path = Path(env_path)
    if path is None:
        path = ROOT / "data" / "elite_kaggle_corpus_clean.txt"
    if not path.exists():
        path = ROOT / "data" / "elite_kaggle_corpus_clean_plus.jsonl"
    if not path.exists():
        path = ROOT / "data" / "phaseA_kaggle_verse.txt"
    return path


def main():
    parser = argparse.ArgumentParser(description="Run couplet evolution")
    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Comma-separated theme keywords (e.g. pressure,mask,survival)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations",
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
        default=5,
        help="Number of elites per generation",
    )
    parser.add_argument(
        "--immigrants",
        type=int,
        default=10,
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
        default="mixed",
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
        action="store_true",
        help="Use SiameseRhymeScorer for embedding-based semantic scoring (blended with keyword score)",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.5,
        help="Weight of embedding score in semantic blend (default: 0.5)",
    )
    parser.add_argument(
        "--multiobjective",
        action="store_true",
        help="Use Pareto multi-objective evolution (rhyme, fluency, semantic)",
    )
    parser.add_argument(
        "--use-niching",
        action="store_true",
        help="Use niching in elite selection (preserve top per rhyme family)",
    )
    parser.add_argument(
        "--min-fluency",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Reject mutations with fluency below this (0=disabled). Default: 0.6 when --theme set, else 0.0.",
    )
    parser.add_argument(
        "--min-semantic",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Reject mutations with semantic below this (0=disabled). Default: 0.25 when --theme set, else 0.0.",
    )
    parser.add_argument(
        "--min-lexical",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Reject mutations with lexical_validity below this (0=disabled). Requires corpus. Try 0.5-0.6 to block nonsense.",
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
        default=0.1,
        help="Weight of style similarity in fitness when --style-corpus is provided (default: 0.1)",
    )
    parser.add_argument(
        "--require-theme",
        action="store_true",
        help="Enforce at least one theme keyword in each couplet (requires --theme)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to evolved weights JSON (from run_weight_tuner.py). Uses 'weights' key.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    theme_keywords = [w.strip() for w in args.theme.split(",") if w.strip()] or None
    corpus_path = Path(args.corpus) if args.corpus else _load_corpus_path()

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

    style_profile = None
    style_weight = 0.0
    if args.style_corpus:
        style_path = Path(args.style_corpus)
        if not style_path.is_absolute():
            style_path = ROOT / style_path
        lines = load_lines_from_file(style_path)
        if lines:
            style_profile = build_style_profile(lines)
            style_weight = args.style_weight
            logger.info(f"Style corpus: {style_path} ({len(lines)} lines) -> style_weight={style_weight}")
        else:
            logger.warning(f"Style corpus empty or not found: {style_path}")

    corpus_lines = load_corpus_lines(corpus_path)
    if corpus_lines:
        corpus_lines = corpus_lines[:2000]
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
        wpath = Path(args.weights)
        if not wpath.is_absolute():
            wpath = ROOT / wpath
        if wpath.exists():
            with wpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fitness_weights = data.get("weights", data)
            logger.info(f"Loaded weights from {wpath}")
        else:
            logger.warning(f"Weights file not found: {wpath}")

    config = EvolutionConfig(
        population_size=args.population,
        num_elites=args.elites,
        random_immigrants_per_gen=args.immigrants,
        fitness_weights=fitness_weights,
        output_dir=output_dir,
        use_embeddings=args.use_embeddings,
        embedding_weight=args.embedding_weight,
        population_init=args.init,
        multiobjective=args.multiobjective,
        use_niching=args.use_niching,
        min_fluency_accept=min_fluency_accept,
        min_semantic_accept=min_semantic_accept,
        min_lexical_accept=min_lexical_accept,
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

    def immigrant_gen(size: int):
        return generator.generate_seed_couplets(theme_keywords=theme_keywords, size=size)

    evolve_fn = evolve_multiobjective if args.multiobjective else evolve
    population = evolve_fn(
        population,
        generations=args.generations,
        config=config,
        prompt_keywords=set(theme_keywords) if theme_keywords else None,
        immigrant_generator=immigrant_gen,
    )

    # Save results
    top = population[: min(50, len(population))]
    output_data = {
        "config": {
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
