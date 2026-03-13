#!/usr/bin/env python
"""
run_verse_evolution.py

CLI to run 4-line verse evolution and save results.

Usage:
    python scripts/run_verse_evolution.py --theme "pressure,mask,survival" --population 80 --generations 30 --scheme AABB --runs-dir

Options:
    --theme       Comma-separated theme keywords
    --population  Population size (default: 80)
    --generations Number of generations (default: 30)
    --scheme      Rhyme scheme: AABB or ABAB (default: AABB)
    --runs-dir    Write to data/evo_rhyme/runs/{timestamp}/
    --output      Output JSON path for top candidates
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

from evo_rhyme.constraints import passes_verse_constraints
from evo_rhyme.individual import analyze_verse_individual
from evo_rhyme.population import (
    VerseSeedGenerator,
    create_initial_verse_population,
)
from evo_rhyme.verse_evolution import (
    VerseEvolutionConfig,
    evolve_verse_population,
)


def _load_corpus_path() -> Path:
    """Load elite corpus path from config/settings or env."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-line verse evolution")
    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Comma-separated theme keywords (e.g. pressure,mask,survival)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=80,
        help="Population size (max 5000)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations (max 1000)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["AABB", "ABAB"],
        default="AABB",
        help="Rhyme scheme: AABB (lines 1-2, 3-4) or ABAB (1-3, 2-4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_verse.json",
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
        default=5,
        help="Random immigrants per generation",
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
        help="Population init: mixed, random, or template",
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
        help="Enable run logging: write to data/evo_rhyme/runs/verse_{timestamp}/",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use SiameseRhymeScorer for embedding-based semantic scoring",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.5,
        help="Weight of embedding score in semantic blend (default: 0.5)",
    )
    parser.add_argument(
        "--use-niching",
        action="store_true",
        help="Use niching in elite selection (preserve top per rhyme family)",
    )
    args = parser.parse_args()

    if args.population > 5000:
        parser.error(f"--population {args.population} exceeds max 5000")
    if args.generations > 1000:
        parser.error(f"--generations {args.generations} exceeds max 1000")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    theme_keywords = [w.strip() for w in args.theme.split(",") if w.strip()] or None
    corpus_path = Path(args.corpus) if args.corpus else _load_corpus_path()

    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Theme: {theme_keywords or 'none'}")
    logger.info(f"Population: {args.population}, Generations: {args.generations}, Scheme: {args.scheme}, Init: {args.init}")

    output_dir = None
    if args.runs_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "data" / "evo_rhyme" / "runs" / f"verse_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run logging to {output_dir}")

    corpus_lines = None
    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as cf:
                corpus_lines = [l.strip() for l in cf if l.strip() and not l.startswith("<")]
            if corpus_lines:
                corpus_lines = corpus_lines[:5000]
                logger.info(f"Loaded {len(corpus_lines)} corpus lines for scoring")
        except Exception as e:
            logger.warning(f"Could not load corpus lines for scoring: {e}")

    config = VerseEvolutionConfig(
        population_size=args.population,
        num_elites=args.elites,
        random_immigrants_per_gen=args.immigrants,
        rhyme_scheme=args.scheme,
        output_dir=output_dir,
        use_embeddings=args.use_embeddings,
        embedding_weight=args.embedding_weight,
        use_niching=args.use_niching,
        corpus_lines=corpus_lines,
    )

    raw_population = create_initial_verse_population(
        theme_keywords=theme_keywords,
        size=args.population,
        corpus_path=corpus_path,
        init_mode=args.init,
    )
    logger.info(f"Initial population: {len(raw_population)} verses (before constraint filter)")

    population = []
    for ind in raw_population:
        analyze_verse_individual(ind)
        if passes_verse_constraints(ind, config.constraint_config):
            population.append(ind)

    max_oversample = 5
    for _ in range(max_oversample):
        if len(population) >= args.population:
            break
        extra = create_initial_verse_population(
            theme_keywords=theme_keywords,
            size=args.population,
            corpus_path=corpus_path,
            init_mode=args.init,
        )
        for ind in extra:
            if len(population) >= args.population:
                break
            analyze_verse_individual(ind)
            if passes_verse_constraints(ind, config.constraint_config):
                population.append(ind)

    population = population[:args.population]
    logger.info(f"Filtered to {len(population)} verses passing constraints")

    verse_gen = VerseSeedGenerator(corpus_path=corpus_path, init_mode=args.init)

    def immigrant_gen(size: int):
        return verse_gen.generate_seed_verses(
            theme_keywords=theme_keywords,
            size=size,
        )

    population = evolve_verse_population(
        population,
        generations=args.generations,
        config=config,
        prompt_keywords=set(theme_keywords) if theme_keywords else None,
        immigrant_generator=immigrant_gen,
    )

    top = population[: min(50, len(population))]
    output_data = {
        "config": {
            "theme": args.theme,
            "population": args.population,
            "generations": args.generations,
            "scheme": args.scheme,
            "elites": args.elites,
            "immigrants": args.immigrants,
            "init": args.init,
        },
        "candidates": [
            {
                "lines": ind.lines,
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

    print("\n--- Top 5 ---")
    for i, ind in enumerate(population[:5], 1):
        print(f"{i}. [{ind.fitness:.4f}]")
        for j, line in enumerate(ind.lines, 1):
            print(f"   {j}. {line}")


if __name__ == "__main__":
    main()
