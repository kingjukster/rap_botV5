#!/usr/bin/env python
"""
run_verse_qd.py

Quality-Diversity verse evolution using MAP-Elites archive.

Usage:
    python scripts/run_verse_qd.py --theme "pressure,mask,survival" --population 120 --generations 30 --scheme AABB --runs-dir
    python scripts/run_verse_qd.py --theme "crown,empire" --init lm --proposer-model gpt-4o-mini --lm-budget 200

Options:
    --theme         Comma-separated theme keywords (required)
    --population    Population size (default: 120, max: 5000)
    --generations   Number of generations (default: 30, max: 1000)
    --scheme        Rhyme scheme: AABB, ABAB, ABBA, AAAA (default: AABB)
    --num-lines     Lines per verse: 4, 8, 16 (default: 4)
    --init          Population init: mixed, random, template, lm (default: lm)
    --lm-budget     LM mutation budget per generation (default: 50)
    --runs-dir      Write to data/evo_rhyme/runs/qd_{timestamp}/
    --output        Output JSON path for top candidates
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-Diversity verse evolution (MAP-Elites)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help="Comma-separated theme keywords (e.g. pressure,mask,survival)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=120,
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
        choices=["AABB", "ABAB", "ABBA", "AAAA"],
        default="AABB",
        help="Rhyme scheme (default: AABB)",
    )
    parser.add_argument(
        "--num-lines",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Lines per verse (default: 4)",
    )
    parser.add_argument(
        "--elites",
        type=int,
        default=10,
        help="Number of elites per generation",
    )
    parser.add_argument(
        "--immigrants",
        type=int,
        default=5,
        help="Immigrants per generation",
    )
    parser.add_argument(
        "--lm-budget",
        type=int,
        default=200,
        help="LM mutation budget per generation (default: 200)",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["mixed", "random", "template", "lm"],
        default="lm",
        help="Population init mode (default: lm)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Override corpus path for seed generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_qd.json",
        help="Output JSON path (default: results_qd.json)",
    )
    parser.add_argument(
        "--runs-dir",
        action="store_true",
        help="Enable run logging to data/evo_rhyme/runs/qd_{timestamp}/",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Enable embedding-based semantic scoring",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.10,
        help="Weight for embedding scorer (default: 0.10)",
    )
    parser.add_argument(
        "--proposer-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for bar proposer (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--proposer-backend",
        type=str,
        choices=["openai", "local_hf"],
        default="openai",
        help="Backend for proposer (default: openai)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for local LM endpoint",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated roles per line (optional)",
    )
    parser.add_argument(
        "--min-fluency",
        type=float,
        default=0.3,
        help="Minimum fluency floor (default: 0.3)",
    )
    parser.add_argument(
        "--min-semantic",
        type=float,
        default=0.0,
        help="Minimum semantic floor (default: 0.0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.population > 5000:
        parser.error(f"--population {args.population} exceeds max 5000")
    if args.generations > 1000:
        parser.error(f"--generations {args.generations} exceeds max 1000")

    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        from evo_rhyme.verse_evolution import evolve_verse_qd, QDEvolutionConfig
    except ImportError as exc:
        logger.error(
            "Could not import QD evolution components from evo_rhyme.verse_evolution. "
            "Make sure evolve_verse_qd and QDEvolutionConfig are implemented.\n%s", exc,
        )
        sys.exit(1)

    from evo_rhyme.archive import create_verse_archive
    from evo_rhyme.constraints import passes_verse_constraints
    from evo_rhyme.individual import analyze_verse_individual

    theme_keywords = [t.strip() for t in args.theme.split(",") if t.strip()]
    if not theme_keywords:
        logger.error("--theme must contain at least one keyword")
        sys.exit(1)

    roles = [r.strip() for r in args.roles.split(",") if r.strip()] if args.roles else None

    corpus_path = Path(args.corpus) if args.corpus else _load_corpus_path()
    logger.info("Corpus: %s", corpus_path)
    logger.info("Theme: %s", theme_keywords)
    logger.info(
        "Population: %d, Generations: %d, Scheme: %s, Init: %s, LM budget: %d",
        args.population, args.generations, args.scheme, args.init, args.lm_budget,
    )

    # ---- Run directory ------------------------------------------------
    output_dir = None
    if args.runs_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "data" / "evo_rhyme" / "runs" / f"qd_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run logging to %s", output_dir)

    # ---- Corpus lines -------------------------------------------------
    corpus_lines: list[str] = []
    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as cf:
                corpus_lines = [l.strip() for l in cf if l.strip() and not l.startswith("<")]
            if corpus_lines:
                corpus_lines = corpus_lines[:5000]
                logger.info("Loaded %d corpus lines for scoring", len(corpus_lines))
        except Exception as e:
            logger.warning("Could not load corpus lines: %s", e)

    corpus_vocab = (
        set(w.lower() for line in corpus_lines for w in line.split())
        if corpus_lines
        else None
    )

    # ---- QD config ----------------------------------------------------
    qd_config = QDEvolutionConfig(
        population_size=args.population,
        num_generations=args.generations,
        num_elites=args.elites,
        random_immigrants_per_gen=args.immigrants,
        rhyme_scheme=args.scheme,
        theme_keywords=theme_keywords,
        num_lines=args.num_lines,
        lm_mutation_budget_per_gen=args.lm_budget,
        min_fluency=args.min_fluency,
        min_semantic=args.min_semantic,
        corpus_vocab=corpus_vocab,
        use_embeddings=args.use_embeddings,
        embedding_weight=args.embedding_weight,
        output_dir=output_dir if args.runs_dir else None,
    )

    # ---- Initial population -------------------------------------------
    if args.init == "lm":
        try:
            from evo_rhyme.population import LMVerseSeedGenerator
        except ImportError as exc:
            logger.error(
                "LMVerseSeedGenerator not available. Install dependencies or use --init mixed.\n%s",
                exc,
            )
            sys.exit(1)

        proposer_config = {
            "model": args.proposer_model,
            "backend": args.proposer_backend,
        }
        if args.api_base:
            proposer_config["api_base"] = args.api_base

        gen = LMVerseSeedGenerator(
            theme_keywords=theme_keywords,
            scheme=args.scheme,
            num_lines=args.num_lines,
            proposer_config=proposer_config,
            roles=roles,
        )
        population = gen.generate(args.population)
        logger.info("LM proposer generated %d seed verses", len(population))
    else:
        from evo_rhyme.population import (
            VerseSeedGenerator,
            create_initial_verse_population,
        )

        raw_population = create_initial_verse_population(
            theme_keywords=theme_keywords,
            size=args.population,
            corpus_path=corpus_path,
            init_mode=args.init,
        )
        logger.info("Initial population: %d verses (before constraint filter)", len(raw_population))

        population = []
        for ind in raw_population:
            analyze_verse_individual(ind)
            if passes_verse_constraints(ind, qd_config.constraint_config if hasattr(qd_config, "constraint_config") else None):
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
                if passes_verse_constraints(ind, qd_config.constraint_config if hasattr(qd_config, "constraint_config") else None):
                    population.append(ind)

    # Pad with fallback seeds if LM didn't produce enough
    if len(population) < args.population:
        shortfall = args.population - len(population)
        logger.info("Population short by %d; padding with template seeds", shortfall)
        try:
            from evo_rhyme.population import create_initial_verse_population
            extra = create_initial_verse_population(
                theme_keywords=theme_keywords,
                size=shortfall * 2,
                corpus_path=corpus_path,
                init_mode="mixed",
            )
            for ind in extra:
                if len(population) >= args.population:
                    break
                analyze_verse_individual(ind)
                if passes_verse_constraints(ind, qd_config.constraint_config if hasattr(qd_config, "constraint_config") else None):
                    population.append(ind)
        except Exception as e:
            logger.warning("Fallback padding failed: %s", e)

    population = population[: args.population]
    logger.info("Final seed population: %d verses", len(population))

    # ---- Immigrant generator ------------------------------------------
    def immigrant_generator(size: int):
        if args.init == "lm":
            try:
                from evo_rhyme.population import LMVerseSeedGenerator
                proposer_cfg = {
                    "model": args.proposer_model,
                    "backend": args.proposer_backend,
                }
                if args.api_base:
                    proposer_cfg["api_base"] = args.api_base
                lm_gen = LMVerseSeedGenerator(
                    theme_keywords=theme_keywords,
                    scheme=args.scheme,
                    num_lines=args.num_lines,
                    proposer_config=proposer_cfg,
                    roles=roles,
                )
                return lm_gen.generate(size)
            except Exception:
                pass
        # Fallback to template-based immigrants
        from evo_rhyme.population import VerseSeedGenerator
        fallback = VerseSeedGenerator(corpus_path=corpus_path, init_mode="mixed")
        return fallback.generate_seed_verses(theme_keywords=theme_keywords, size=size)

    # ---- Run QD evolution ---------------------------------------------
    logger.info("Starting QD evolution (%d generations)...", args.generations)

    archive, final_pop = evolve_verse_qd(
        population=population,
        config=qd_config,
        immigrant_generator=immigrant_generator,
    )

    # ---- Output results -----------------------------------------------
    print(f"\nArchive coverage: {archive.coverage() * 100:.1f}%")
    print(f"Occupied niches: {archive.occupied_niches()}/{archive.total_niches()}")

    top = archive.top_k(50)
    results = {
        "config": {
            "theme": args.theme,
            "population": args.population,
            "generations": args.generations,
            "scheme": args.scheme,
            "num_lines": args.num_lines,
            "init": args.init,
            "lm_budget": args.lm_budget,
            "elites": args.elites,
            "immigrants": args.immigrants,
            "use_embeddings": args.use_embeddings,
            "embedding_weight": args.embedding_weight,
            "min_fluency": args.min_fluency,
            "min_semantic": args.min_semantic,
        },
        "archive_summary": archive.summary(),
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
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved top %d candidates to %s", len(top), out_path)

    if output_dir:
        archive_path = output_dir / "archive.json"
        with archive_path.open("w", encoding="utf-8") as f:
            json.dump(archive.to_json(), f, indent=2, default=str)
        logger.info("Full archive written to %s", archive_path)

    # ---- Print top 5 --------------------------------------------------
    print("\n=== Top 5 Verses ===")
    for i, ind in enumerate(top[:5], 1):
        print(f"\n--- #{i} (fitness={ind.fitness:.4f}) ---")
        for line in ind.lines:
            print(f"  {line}")
        if ind.scores:
            key_scores = {
                k: f"{v:.3f}"
                for k, v in ind.scores.items()
                if isinstance(v, float) and abs(v) > 0.001
            }
            print(f"  scores: {key_scores}")


if __name__ == "__main__":
    main()
