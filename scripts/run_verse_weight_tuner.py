#!/usr/bin/env python
"""
run_verse_weight_tuner.py

CLI to run outer-loop evolution of verse (4-line) scoring weights.
Evolves a compact weight genome (Structure / Meaning / Flavor / Penalty) so that
verse QD runs produce outputs that score higher on a proxy meta-fitness:
coherence, fluency, punchline, novelty, minus garbled/cliché/repetition.

Usage:
    python scripts/run_verse_weight_tuner.py --outer-generations 6 --output verse_weights_evolved.json
    python scripts/run_verse_weight_tuner.py --weights-file verse_weights_evolved.json --output verse_weights_refined.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.verse_weight_tuner import (
    DEFAULT_BENCHMARK_THEMES,
    evolve_verse_weights,
    genome_to_weights,
)


def _load_corpus_path() -> Path:
    """Load elite corpus path."""
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
    parser = argparse.ArgumentParser(
        description="Evolve verse scoring weights via outer-loop meta-optimization"
    )
    parser.add_argument(
        "--outer-population",
        type=int,
        default=10,
        help="Number of weight genomes per outer generation (default: 10)",
    )
    parser.add_argument(
        "--outer-generations",
        type=int,
        default=6,
        help="Outer generations (default: 6)",
    )
    parser.add_argument(
        "--inner-population",
        type=int,
        default=50,
        help="Inner verse population size per theme (default: 50)",
    )
    parser.add_argument(
        "--inner-generations",
        type=int,
        default=8,
        help="Inner verse QD generations per theme (default: 8)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top N verses per theme for meta-scoring (default: 5)",
    )
    parser.add_argument(
        "--themes",
        type=str,
        default=None,
        help="Comma-separated theme groups, e.g. 'flow,show:pressure,mask'. Default: flow/show, pressure/mask, pain/growth",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Override corpus path for inner evolution",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="verse_weights_evolved.json",
        help="Output JSON path for best weights (default: verse_weights_evolved.json)",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="Path to existing weights JSON to seed population (refinement run)",
    )
    parser.add_argument(
        "--use-emitters",
        action="store_true",
        help="Use emitter-based verse QD in inner loop (slower, more coverage)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    corpus_path = Path(args.corpus) if args.corpus else _load_corpus_path()
    corpus_lines = None
    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_lines = [l.strip() for l in f if l.strip() and not l.startswith("<")]
            if corpus_lines:
                corpus_lines = corpus_lines[:5000]
        except Exception:
            pass

    benchmark_themes = DEFAULT_BENCHMARK_THEMES
    if args.themes:
        benchmark_themes = []
        for part in args.themes.split(":"):
            benchmark_themes.append([w.strip() for w in part.split(",") if w.strip()])
        if not benchmark_themes:
            benchmark_themes = DEFAULT_BENCHMARK_THEMES

    seed_weights_path = Path(args.weights_file) if args.weights_file else None
    if seed_weights_path and not seed_weights_path.is_absolute():
        seed_weights_path = ROOT / seed_weights_path

    logger.info("Corpus: %s", corpus_path)
    logger.info("Benchmark themes: %s", benchmark_themes)
    logger.info(
        "Outer: %d genomes x %d gen | Inner: %d pop x %d gen, top_n=%d",
        args.outer_population, args.outer_generations,
        args.inner_population, args.inner_generations, args.top_n,
    )

    results = evolve_verse_weights(
        population_size=args.outer_population,
        generations=args.outer_generations,
        benchmark_themes=benchmark_themes,
        inner_population_size=args.inner_population,
        inner_generations=args.inner_generations,
        top_n_per_theme=args.top_n,
        corpus_path=corpus_path,
        corpus_lines=corpus_lines,
        seed=args.seed,
        use_emitters=args.use_emitters,
        seed_weights_path=seed_weights_path,
    )

    best_genome, best_meta = results[0]
    best_weights = genome_to_weights(best_genome)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "meta_fitness": best_meta,
        "genome": best_genome,
        "weights": best_weights,
        "config": {
            "outer_population": args.outer_population,
            "outer_generations": args.outer_generations,
            "inner_population": args.inner_population,
            "inner_generations": args.inner_generations,
            "top_n_per_theme": args.top_n,
            "benchmark_themes": benchmark_themes,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved best verse weights (meta_fitness=%.4f) to %s", best_meta, out_path)
    print("\n--- Best evolved verse weights ---")
    for k, v in sorted(best_weights.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
