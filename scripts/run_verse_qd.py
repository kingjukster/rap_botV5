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
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    # Two-phase parse to allow --config to influence defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file path (YAML/JSON). Overrides defaults via config/settings.py.",
    )
    known, _ = pre.parse_known_args()

    from config.settings import get_qd_defaults

    defaults = get_qd_defaults(config_path=known.config)

    parser = argparse.ArgumentParser(
        description="Quality-Diversity verse evolution (MAP-Elites)",
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
        required=True,
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
        default=int(defaults.get("generations", 100)),
        help="Number of generations (max 1000)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["AABB", "ABAB", "ABBA", "AAAA", "ABCB", "AABA"],
        default=str(defaults.get("scheme", "AABB")),
        help="Rhyme scheme (default: AABB)",
    )
    parser.add_argument(
        "--num-lines",
        type=int,
        choices=[4, 8, 16],
        default=int(defaults.get("num_lines", 4)),
        help="Lines per verse (default: 4)",
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
        default=int(defaults.get("immigrants", 20)),
        help="Immigrants per generation",
    )
    parser.add_argument(
        "--lm-budget",
        type=int,
        default=int(defaults.get("lm_budget", 20)),
        help="LM mutation budget per verse generation (default: 20)",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["mixed", "random", "template", "lm"],
        default=str(defaults.get("init", "lm")),
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
        "--db",
        action="store_true",
        help="Enable MySQL persistence (RAPBOT_USE_DB=1)",
    )
    parser.add_argument(
        "--resume",
        type=int,
        metavar="RUN_ID",
        default=None,
        help="Resume from a previous run: load archive from DB (requires --db)",
    )
    parser.add_argument(
        "--use-embeddings",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_embeddings", False)),
        help="Enable embedding-based semantic scoring",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=float(defaults.get("embedding_weight", 0.40)),
        help="Weight for embedding scorer (default: 0.40)",
    )
    parser.add_argument(
        "--proposer-model",
        type=str,
        default=str(defaults.get("proposer_model", "gpt-4.1-nano")),
        help="Model name for bar proposer (default: gpt-4.1-nano)",
    )
    parser.add_argument(
        "--proposer-backend",
        type=str,
        choices=["openai", "local_hf"],
        default=str(defaults.get("proposer_backend", "openai")),
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
        default=float(defaults.get("min_fluency", 0.3)),
        help="Minimum fluency floor (default: 0.3)",
    )
    parser.add_argument(
        "--min-semantic",
        type=float,
        default=float(defaults.get("min_semantic", 0.0)),
        help="Minimum semantic floor (default: 0.0)",
    )
    parser.add_argument(
        "--line-pop",
        type=int,
        default=int(defaults.get("line_pop", 1500)),
        help="Line population size for two-tier evolution",
    )
    parser.add_argument(
        "--line-gens",
        type=int,
        default=int(defaults.get("line_gens", 3)),
        help="Line evolution generations per verse generation",
    )
    parser.add_argument(
        "--line-seeds",
        type=int,
        default=int(defaults.get("line_seeds", 80)),
        help="LM seed lines for line evolution (gen 0 only)",
    )
    parser.add_argument(
        "--line-lm-budget",
        type=int,
        default=int(defaults.get("line_lm_budget", 15)),
        help="LM mutation budget per line evolution generation (default: 15)",
    )
    parser.add_argument(
        "--compose-ratio",
        type=float,
        default=float(defaults.get("compose_ratio", 0.5)),
        help="Fraction of offspring from line archive assembly",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--emitter-strategy",
        type=str,
        choices=["multi", "classic"],
        default=str(defaults.get("emitter_strategy", "multi")),
        help="Evolution strategy: multi (emitter-based MAP-Elites) or classic (default: multi)",
    )
    parser.add_argument(
        "--novelty-weight",
        type=float,
        default=float(defaults.get("novelty_weight", 0.3)),
        help="Novelty weight in effective fitness (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--schemes",
        type=str,
        default=str(defaults.get("schemes", "AABB,ABAB,ABBA,ABCB")),
        help="Comma-separated allowed rhyme schemes (default: AABB,ABAB,ABBA,ABCB)",
    )
    parser.add_argument(
        "--archive-mode",
        type=str,
        choices=["default", "style_chain", "compact_style", "ultra_compact", "curriculum_compact"],
        default=str(defaults.get("archive_mode", "compact_style")),
        help="Archive dimensions mode (default: compact_style)",
    )
    parser.add_argument(
        "--curriculum-switch-gen",
        type=int,
        default=int(defaults.get("curriculum_switch_gen", 20)),
        help="Generation index for curriculum switch to compact archive (default: 20)",
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("fast_mode", True)),
        help="Enable staged fast scoring (default: enabled)",
    )
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=int(defaults.get("graph_top_k", 24)),
        help="Compute rhyme-graph metrics for top-k candidates per batch (default: 24)",
    )
    parser.add_argument(
        "--expensive-top-k",
        type=int,
        default=int(defaults.get("expensive_top_k", 40)),
        help="Compute expensive LM/coherence scores for top-k candidates in fast mode",
    )
    parser.add_argument(
        "--graph-edge-mode",
        type=str,
        choices=["phonetic", "embedding"],
        default=str(defaults.get("graph_edge_mode", "phonetic")),
        help="Rhyme graph edge mode (default: phonetic)",
    )
    parser.add_argument(
        "--style-genome",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("style_genome", True)),
        help="Enable style genome metadata and evolution operators (default: enabled)",
    )
    parser.add_argument(
        "--prompt-genome",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("prompt_genome", True)),
        help="Enable prompt genome metadata and operators (default: enabled)",
    )
    parser.add_argument(
        "--prompt-llm-fraction",
        type=float,
        default=float(defaults.get("prompt_llm_fraction", 0.2)),
        help="Fraction of random emitter candidates generated through prompt-conditioned LM path",
    )
    parser.add_argument(
        "--enable-controllability-probes",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("enable_controllability_probes", False)),
        help="Log lightweight style controllability probes every few generations",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=defaults.get("coverage_target", None),
        metavar="FRAC",
        help="Target archive coverage (0.0-1.0). When >= 0.5, overrides population=100 and generations=150 for compact_style. For ultra_compact, 80/80 is sufficient.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to JSON file with evolved scoring weights (expects 'weights' key). Used for weight-tuned runs.",
    )

    args = parser.parse_args()

    if args.population > 5000:
        parser.error(f"--population {args.population} exceeds max 5000")
    if args.generations > 1000:
        parser.error(f"--generations {args.generations} exceeds max 1000")

    # Auto-tune for 50% coverage target
    if args.coverage_target is not None and args.coverage_target >= 0.5:
        if args.archive_mode == "ultra_compact":
            if args.population == 100 and args.generations == 100:
                args.population = 80
                args.generations = 80
        else:
            if args.population == 100 and args.generations == 100:
                args.population = 100
                args.generations = 150
            elif args.population <= 100 and args.generations <= 100:
                args.population = max(args.population, 100)
                args.generations = max(args.generations, 150)

    return args


def main() -> None:
    args = parse_args()

    seed_info: Optional[Dict[str, Any]] = None
    if args.seed is not None:
        try:
            from evo_rhyme.repro import seed_everything

            seed_info = seed_everything(int(args.seed))
        except Exception:
            seed_info = {"seed": int(args.seed), "error": "seed_everything_failed"}

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "asyncio", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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

    from config import get_elite_corpus_path
    corpus_path = Path(args.corpus) if args.corpus else get_elite_corpus_path()
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

    # ---- DB run -------------------------------------------------------
    run_id = None
    if args.db:
        import os
        os.environ["RAPBOT_USE_DB"] = "1"
        try:
            from evo_rhyme import db
            if db.db_enabled():
                run_id = db.insert_run(
                    "run_verse_qd",
                    ",".join(theme_keywords) if theme_keywords else "",
                    {
                        "theme": args.theme,
                        "population": args.population,
                        "generations": args.generations,
                        "scheme": args.scheme,
                        "num_lines": args.num_lines,
                        "init": args.init,
                        "emitter_strategy": args.emitter_strategy,
                        **({"seed_info": seed_info} if seed_info else {}),
                    },
                )
                if run_id > 0:
                    logger.info("DB run_id=%d", run_id)
        except Exception as e:
            logger.warning("DB insert_run failed: %s", e)

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
        min_coherence=0.25,
        min_semantic=args.min_semantic,
        corpus_vocab=corpus_vocab,
        use_embeddings=args.use_embeddings,
        embedding_weight=args.embedding_weight,
        output_dir=output_dir if args.runs_dir else None,
        line_population_size=args.line_pop,
        line_generations_per_verse_gen=args.line_gens,
        line_lm_seed_count=args.line_seeds,
        line_lm_mutation_budget=args.line_lm_budget,
        composed_offspring_ratio=args.compose_ratio,
        archive_mode=args.archive_mode,
        fast_mode=args.fast_mode,
        graph_top_k=args.graph_top_k,
        max_expensive_scoring_candidates=args.expensive_top_k,
        graph_edge_mode=args.graph_edge_mode,
        enable_style_genome=args.style_genome,
        enable_prompt_genome=args.prompt_genome,
        prompt_llm_fraction=args.prompt_llm_fraction,
        curriculum_switch_gen=args.curriculum_switch_gen,
        enable_controllability_probes=args.enable_controllability_probes,
        coverage_target=args.coverage_target,
        run_id=run_id if run_id and run_id > 0 else None,
        initial_archive=None,  # Set below if --resume
    )
    if args.resume is not None and args.resume > 0:
        if not args.db:
            logger.warning("--resume requires --db; enabling DB")
            import os
            os.environ["RAPBOT_USE_DB"] = "1"
        try:
            from evo_rhyme import db
            from evo_rhyme.archive import (
                MAPElitesArchive,
                compact_style_dimensions,
                style_chain_dimensions,
                ultra_compact_dimensions,
                default_verse_dimensions,
            )
            cells = db.load_archive_cells(args.resume)
            if cells:
                mode = getattr(args, "archive_mode", "compact_style")
                if mode == "style_chain":
                    dims = style_chain_dimensions()
                elif mode == "ultra_compact":
                    dims = ultra_compact_dimensions()
                elif mode == "curriculum_compact":
                    dims = default_verse_dimensions()
                else:
                    dims = compact_style_dimensions()
                data = [
                    {
                        "niche": [int(x) for x in c["cell_key"].split("_")],
                        "lines": c["lines"],
                        "fitness": c["fitness"],
                        "scores": c["scores"] or {},
                    }
                    for c in cells
                ]
                qd_config.initial_archive = MAPElitesArchive.from_json(data, dims)
                logger.info("Resumed archive from run_id=%d (%d niches)", args.resume, len(cells))
            else:
                logger.warning("No archive cells found for run_id=%d", args.resume)
        except Exception as e:
            logger.warning("Resume failed: %s", e)

    if args.weights:
        wpath = Path(args.weights)
        if not wpath.is_absolute():
            wpath = ROOT / wpath
        if wpath.exists():
            with wpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            qd_config.fitness_weights = data.get("weights", data)
            logger.info("Loaded fitness weights from %s", wpath)
        else:
            logger.warning("Weights file not found: %s", wpath)
    # Set emitter-specific config
    qd_config.use_emitters = (args.emitter_strategy == "multi")
    qd_config.novelty_weight = args.novelty_weight
    qd_config.allowed_schemes = [s.strip() for s in args.schemes.split(",")]
    qd_config.proposer_config = {
        "model": args.proposer_model,
        "backend": args.proposer_backend,
        **({"api_base": args.api_base} if args.api_base else {}),
    }
    if seed_info:
        # Used by VerseQDRunLogger.write_config for run-dir config.json.
        setattr(qd_config, "seed_info", seed_info)

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
    # Returns a single VerseIndividual per call (matches evolve_verse_qd signature).
    # Uses template-based generation to avoid burning API budget on immigrants.
    def immigrant_generator():
        from evo_rhyme.population import VerseSeedGenerator
        fallback = VerseSeedGenerator(corpus_path=corpus_path, init_mode="mixed")
        results = fallback.generate_seed_verses(theme_keywords=theme_keywords, size=1)
        if results:
            return results[0]
        return None

    # ---- Run QD evolution ---------------------------------------------
    logger.info("Starting QD evolution (%d generations)...", args.generations)
    verse_16_results = []

    try:
        if getattr(qd_config, 'use_emitters', False):
            from evo_rhyme.verse_evolution import evolve_verse_qd_emitters
            logger.info("Using emitter-based MAP-Elites strategy")
            archive, final_pop = evolve_verse_qd_emitters(
                population=population,
                config=qd_config,
                immigrant_generator=immigrant_generator,
            )
        else:
            archive, final_pop = evolve_verse_qd(
                population=population,
                config=qd_config,
                immigrant_generator=immigrant_generator,
            )
    except Exception as e:
        if run_id and run_id > 0:
            try:
                from evo_rhyme import db
                db.update_run_status(run_id, "failed")
            except Exception:
                pass
        raise

    if run_id and run_id > 0:
        try:
            from evo_rhyme import db
            db.update_run_status(run_id, "completed")
        except Exception as e:
            logger.warning("DB update_run_status failed: %s", e)

    # ---- 16-bar composition (when --num-lines 16) --------------------
    if args.num_lines == 16:
        logger.info("Starting 16-bar verse composition from 4-bar block archive...")
        try:
            from evo_rhyme.block_archive import BlockArchive
            from evo_rhyme.verse_composer import (
                compose_verse_batch,
                verse_16_crossover,
                verse_16_mutate,
            )
            from evo_rhyme.fitness import score_verse_16, compute_verse_16_fitness

            top_blocks = archive.top_k(200)
            block_archive = BlockArchive.from_verse_individuals(top_blocks)
            logger.info(
                "Block archive: %d blocks, roles: %s",
                block_archive.size(), block_archive.role_sizes(),
            )

            if block_archive.size() >= 4:
                verses_16 = compose_verse_batch(
                    block_archive,
                    count=min(50, args.population),
                    scheme=args.scheme,
                )
                logger.info("Composed %d initial 16-bar verses", len(verses_16))

                for evo_gen in range(10):
                    for v in verses_16:
                        block_fits = v.metadata.get("block_fitnesses", [])
                        v.scores = score_verse_16(v, block_fitnesses=block_fits)
                        v.fitness = compute_verse_16_fitness(v.scores)

                    verses_16.sort(key=lambda x: x.fitness or 0.0, reverse=True)

                    elites = verses_16[:5]
                    offspring = []
                    for _ in range(len(verses_16) - 5):
                        p1 = random.choice(verses_16[:20])
                        p2 = random.choice(verses_16[:20])
                        child = verse_16_crossover(p1, p2)
                        child = verse_16_mutate(child, config={
                            "theme_keywords": theme_keywords,
                            "min_syllables": 6,
                            "max_syllables": 18,
                        })
                        offspring.append(child)

                    for v in offspring:
                        v.scores = score_verse_16(v)
                        v.fitness = compute_verse_16_fitness(v.scores)

                    verses_16 = elites + offspring
                    best_16 = max(v.fitness or 0.0 for v in verses_16)
                    if evo_gen % 3 == 0:
                        logger.info(
                            "16-bar gen %d: best=%.3f, pop=%d",
                            evo_gen, best_16, len(verses_16),
                        )

                verses_16.sort(key=lambda x: x.fitness or 0.0, reverse=True)
                verse_16_results = verses_16[:20]
                logger.info(
                    "16-bar evolution complete. Top fitness: %.3f",
                    verse_16_results[0].fitness if verse_16_results else 0.0,
                )
            else:
                logger.warning(
                    "Not enough blocks for 16-bar composition (need 4, have %d)",
                    block_archive.size(),
                )
        except Exception as e:
            logger.error("16-bar composition failed: %s", e, exc_info=True)

    # ---- Output results -----------------------------------------------
    print(f"\nArchive coverage: {archive.coverage() * 100:.1f}%")
    print(f"Occupied niches: {archive.occupied_niches()}/{archive.total_niches()}")

    top = archive.top_k(50)
    results = {
        "config": {
            **({"seed_info": seed_info} if seed_info else {}),
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
            "emitter_strategy": args.emitter_strategy,
            "novelty_weight": args.novelty_weight,
            "schemes": args.schemes,
            "archive_mode": args.archive_mode,
            "fast_mode": args.fast_mode,
            "graph_top_k": args.graph_top_k,
            "expensive_top_k": args.expensive_top_k,
            "graph_edge_mode": args.graph_edge_mode,
            "style_genome": args.style_genome,
            "prompt_genome": args.prompt_genome,
            "prompt_llm_fraction": args.prompt_llm_fraction,
            "curriculum_switch_gen": args.curriculum_switch_gen,
            "enable_controllability_probes": args.enable_controllability_probes,
        },
        "archive_summary": archive.summary(),
        "archive_dimensions": {
            "total_niches": archive.total_niches(),
            "dimensions": [d.name for d in archive.dimensions],
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

    if verse_16_results:
        results["verses_16"] = [
            {
                "lines": ind.lines,
                "fitness": ind.fitness,
                "scores": ind.scores,
                "metadata": {
                    k: v for k, v in (ind.metadata or {}).items()
                    if k in ("block_roles", "block_fitnesses", "origin")
                },
            }
            for ind in verse_16_results
        ]

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

    if verse_16_results:
        print("\n=== Top 3 16-Bar Verses ===")
        for i, ind in enumerate(verse_16_results[:3], 1):
            print(f"\n--- 16-Bar #{i} (fitness={ind.fitness:.4f}) ---")
            for block_idx in range(len(ind.lines) // 4):
                start = block_idx * 4
                role = ind.metadata.get("block_roles", ["?"] * 4)[block_idx] if ind.metadata else "?"
                print(f"  [Block {block_idx + 1}: {role}]")
                for line in ind.lines[start:start + 4]:
                    print(f"    {line}")
            if ind.scores:
                key_scores = {
                    k: f"{v:.3f}"
                    for k, v in ind.scores.items()
                    if isinstance(v, float) and abs(v) > 0.001
                }
                print(f"  scores: {key_scores}")


if __name__ == "__main__":
    main()
