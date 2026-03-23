#!/usr/bin/env python
"""
run_hierarchical.py

Hierarchical verse evolution: couplet evolution -> 4-bar evolution.

Stage 1: Evolve couplets (or load from --couplet-archive JSON).
Stage 2: Build 4-bar verses from couplet archive, run QD evolution with
         structural mutations (swap couplets, rewrite transition).

Usage:
    python scripts/run_hierarchical.py --theme "pressure,mask,survival" --couplet-gen 20 --4bar-gen 30
    python scripts/run_hierarchical.py --theme "crown,empire" --couplet-archive results.json --4bar-gen 50 --runs-dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.block_builder import build_4bar_batch
from evo_rhyme.constraints import passes_verse_constraints
from evo_rhyme.couplet_archive import CoupletArchive, ScoredCouplet
from evo_rhyme.individual import analyze_verse_individual
from evo_rhyme.verse_evolution import QDEvolutionConfig, evolve_verse_qd, evolve_verse_qd_emitters


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Optional config file path.")
    known, _ = pre.parse_known_args()

    from config.settings import get_evolution_defaults, get_qd_defaults
    evo_defaults = get_evolution_defaults(config_path=known.config)
    qd_defaults = get_qd_defaults(config_path=known.config)

    parser = argparse.ArgumentParser(
        description="Hierarchical verse evolution: couplet -> 4-bar",
        parents=[pre],
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help="Comma-separated theme keywords (e.g. pressure,mask,survival)",
    )
    parser.add_argument(
        "--couplet-gen",
        type=int,
        default=20,
        help="Couplet evolution generations (default: 20)",
    )
    parser.add_argument(
        "--4bar-gen",
        type=int,
        default=30,
        dest="gen_4bar",
        help="4-bar QD evolution generations (default: 30)",
    )
    parser.add_argument(
        "--couplet-archive",
        type=str,
        default=None,
        help="Path to couplet evolution results JSON (skip Stage 1 if provided)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["AABB", "ABAB", "ABBA", "AAAA", "ABCB", "AABA"],
        default=str(qd_defaults.get("scheme", "AABB")),
        help="Rhyme scheme (default: AABB)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=int(qd_defaults.get("population", 80)),
        help="4-bar population size (default: 80)",
    )
    parser.add_argument(
        "--couplet-population",
        type=int,
        default=int(evo_defaults.get("population", 40)),
        help="Couplet evolution population (default: 40)",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["mixed", "random", "template"],
        default=str(evo_defaults.get("init", "mixed")),
        help="Couplet population init mode (default: mixed)",
    )
    parser.add_argument(
        "--emitter-strategy",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="QD emitter strategy: single (no emitters) or multi (default: single for hierarchical)",
    )
    parser.add_argument(
        "--runs-dir",
        action="store_true",
        help="Enable run logging to data/evo_rhyme/runs/hierarchical_{timestamp}/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_hierarchical.json",
        help="Output JSON path (default: results_hierarchical.json)",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Enable MySQL persistence",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Override corpus path for couplet seed generation",
    )
    return parser.parse_args()


def _run_couplet_evolution(
    theme_keywords: list[str],
    generations: int,
    population: int,
    init: str,
    corpus_path: Path,
    seed: Optional[int],
) -> list[tuple[str, str, float, dict]]:
    """Run couplet evolution and return list of (line1, line2, fitness, scores)."""
    from evo_rhyme.constraints import passes_constraints
    from evo_rhyme.evolution import EvolutionConfig, _effective_constraint_config, evolve
    from evo_rhyme.individual import analyze_individual
    from evo_rhyme.population import (
        create_mixed_population,
        create_initial_population,
        RandomGenerator,
        TemplateGenerator,
    )
    from evo_rhyme.seed_generator import SeedGenerator, load_corpus_lines

    if seed is not None:
        try:
            from evo_rhyme.repro import seed_everything
            seed_everything(seed)
        except Exception:
            pass

    corpus_lines = load_corpus_lines(corpus_path)
    if corpus_lines:
        corpus_lines = corpus_lines[:2000]
    min_fluency = 0.6 if theme_keywords else 0.0
    min_semantic = 0.25 if theme_keywords else 0.0
    min_ngram = 0.2 if corpus_lines else 0.0

    config = EvolutionConfig(
        population_size=min(population, 60),
        num_elites=5,
        random_immigrants_per_gen=10,
        population_init=init,
        min_fluency_accept=min_fluency,
        min_semantic_accept=min_semantic,
        min_ngram_fluency_accept=min_ngram,
        corpus_lines=corpus_lines if corpus_lines else None,
    )

    prompt_kw = set(theme_keywords) if theme_keywords else None
    effective_constraint = _effective_constraint_config(config, prompt_kw)

    if init == "mixed":
        raw = create_mixed_population(
            corpus_path=corpus_path,
            theme_keywords=theme_keywords,
            size=config.population_size,
        )
    elif init == "template":
        raw = create_initial_population(
            TemplateGenerator(),
            theme_keywords=theme_keywords,
            size=config.population_size,
        )
    else:
        raw = create_initial_population(
            RandomGenerator(),
            theme_keywords=theme_keywords,
            size=config.population_size,
        )

    population_list = []
    for ind in raw:
        analyze_individual(ind)
        if passes_constraints(ind, effective_constraint):
            population_list.append(ind)
    population_list = population_list[: config.population_size]

    if len(population_list) == 0:
        raise RuntimeError(
            "No couplets passed constraints. Try looser theme or different init."
        )

    gen = SeedGenerator(corpus_path=corpus_path)

    def immigrant_gen(size: int):
        return gen.generate_seed_couplets(
            theme_keywords=theme_keywords,
            size=size,
        )

    population_list = evolve(
        population_list,
        generations=generations,
        config=config,
        prompt_keywords=prompt_kw,
        immigrant_generator=immigrant_gen,
    )

    return [
        (ind.line1, ind.line2, ind.fitness or 0.0, ind.scores or {})
        for ind in population_list
    ]


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    theme_keywords = [w.strip() for w in args.theme.split(",") if w.strip()]
    from config import get_elite_corpus_path
    corpus_path = Path(args.corpus) if args.corpus else get_elite_corpus_path()

    seed_info: Optional[Dict[str, Any]] = None
    if args.seed is not None:
        try:
            from evo_rhyme.repro import seed_everything
            seed_info = seed_everything(int(args.seed))
        except Exception:
            seed_info = {"seed": int(args.seed)}

    output_dir: Optional[Path] = None
    if args.runs_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "data" / "evo_rhyme" / "runs" / f"hierarchical_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run logging to %s", output_dir)

    run_id = None
    if args.db:
        import os
        os.environ["RAPBOT_USE_DB"] = "1"
        try:
            from evo_rhyme import db
            if db.db_enabled():
                config_json = {
                    "script": "run_hierarchical",
                    "theme": args.theme,
                    "couplet_gen": args.couplet_gen,
                    "4bar_gen": getattr(args, "gen_4bar", 30),
                    "scheme": args.scheme,
                    "population": args.population,
                    "seed_info": seed_info,
                }
                run_id = db.insert_run(
                    "run_hierarchical",
                    args.theme,
                    config_json,
                )
                if run_id and run_id > 0:
                    logger.info("DB run_id=%d", run_id)
        except Exception as e:
            logger.warning("DB insert_run failed: %s", e)

    # ---- Stage 1: Couplet archive ----
    couplet_archive: CoupletArchive
    if args.couplet_archive:
        path = Path(args.couplet_archive)
        if not path.is_absolute():
            path = ROOT / path
        couplet_archive = CoupletArchive.from_results_json(path)
        logger.info("Loaded couplet archive from %s: %d couplets", path, couplet_archive.size())
        if couplet_archive.size() < 4:
            logger.error("Need at least 4 couplets from archive. Got %d.", couplet_archive.size())
            return 1
    else:
        logger.info("Stage 1: Running couplet evolution (%d gens)...", args.couplet_gen)
        results = _run_couplet_evolution(
            theme_keywords=theme_keywords,
            generations=args.couplet_gen,
            population=args.couplet_population,
            init=args.init,
            corpus_path=corpus_path,
            seed=args.seed,
        )
        couplet_archive = CoupletArchive()
        for line1, line2, fitness, scores in results:
            sc = ScoredCouplet.from_couplet(line1, line2, fitness=fitness, scores=scores)
            couplet_archive.add(sc)
        logger.info("Couplet evolution complete: %d couplets, %d groups",
                    couplet_archive.size(), couplet_archive.group_count())
        if output_dir:
            archive_data = couplet_archive.to_json()
            (output_dir / "couplet_archive.json").write_text(
                json.dumps(archive_data, indent=2),
                encoding="utf-8",
            )
        if run_id and run_id > 0:
            sid = couplet_archive.save_to_seed_bank(run_id=run_id)
            if sid > 0:
                logger.info("Saved couplet archive to seed_bank (id=%d)", sid)

    # ---- Stage 2: 4-bar evolution ----
    logger.info("Stage 2: Seeding 4-bar population from couplet archive...")
    population = build_4bar_batch(
        couplet_archive,
        count=args.population,
        scheme=args.scheme,
        theme_keywords=theme_keywords,
    )
    if not population:
        logger.error("Could not build any 4-bar verses from couplet archive.")
        return 1

    constraint_config = {
        "min_syllables": 6,
        "max_syllables": 18,
        "prompt_keywords": theme_keywords,
    }
    filtered = []
    for v in population:
        analyze_verse_individual(v)
        if passes_verse_constraints(v, constraint_config):
            filtered.append(v)
    population = filtered[: args.population]

    if len(population) < args.population // 2:
        logger.warning(
            "Only %d verses passed constraints; padding with more from archive.",
            len(population),
        )
        extra = build_4bar_batch(
            couplet_archive,
            count=args.population * 2,
            scheme=args.scheme,
            theme_keywords=theme_keywords,
        )
        for v in extra:
            if len(population) >= args.population:
                break
            analyze_verse_individual(v)
            if passes_verse_constraints(v, constraint_config):
                population.append(v)

    def immigrant_generator():
        from evo_rhyme.block_builder import build_4bar_from_couplets
        v = build_4bar_from_couplets(
            couplet_archive,
            scheme=args.scheme,
            theme_keywords=theme_keywords,
        )
        if v and passes_verse_constraints(v, constraint_config):
            analyze_verse_individual(v)
            return v
        return None

    qd_config = QDEvolutionConfig(
        population_size=args.population,
        num_generations=getattr(args, "gen_4bar", 30),
        num_elites=5,
        random_immigrants_per_gen=10,
        rhyme_scheme=args.scheme,
        theme_keywords=theme_keywords,
        num_lines=4,
        lm_mutation_budget_per_gen=20,
        min_fluency=0.4,
        min_coherence=0.25,
        output_dir=str(output_dir) if output_dir else None,
        run_id=run_id if run_id and run_id > 0 else None,
        use_structural_mutations=True,
        use_emitters=(args.emitter_strategy == "multi"),
    )
    if seed_info:
        setattr(qd_config, "seed_info", seed_info)

    logger.info("Starting 4-bar QD evolution (%d gens, structural mutations enabled)...",
                getattr(args, "gen_4bar", 30))
    try:
        if args.emitter_strategy == "multi":
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
                db.update_run_status(run_id, "failed", failure_reason=str(e)[:4096])
            except Exception:
                pass
        raise

    if run_id and run_id > 0:
        try:
            from evo_rhyme import db
            db.update_run_status(run_id, "completed")
        except Exception as e:
            logger.warning("DB update_run_status failed: %s", e)

    top = archive.top_k(50) if hasattr(archive, "top_k") else final_pop[:50]
    results = {
        "config": {
            **({"seed_info": seed_info} if seed_info else {}),
            "theme": args.theme,
            "couplet_gen": args.couplet_gen,
            "4bar_gen": getattr(args, "gen_4bar", 30),
            "scheme": args.scheme,
            "population": args.population,
            "couplet_archive_size": couplet_archive.size(),
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
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved top %d candidates to %s", len(top), out_path)

    print("\n--- Top 5 ---")
    for i, ind in enumerate(top[:5], 1):
        print(f"{i}. [{ind.fitness or 0:.4f}]")
        for line in ind.lines:
            print(f"   {line}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
