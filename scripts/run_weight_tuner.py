#!/usr/bin/env python
"""
run_weight_tuner.py

CLI to run outer-loop weight evolution. Evolves fitness weights so that
the inner lyric evolution produces better outputs across a benchmark.

Usage:
    python scripts/run_weight_tuner.py --outer-generations 6 --output weights_evolved.json

The tuner runs lyric evolution on 5 benchmark prompts (flow/show, pressure/mask, etc.)
and ranks weight sets by meta-fitness: fluency, lexical validity, diversity,
minus nonsense and repetition rates.
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

from evo_rhyme.weight_tuner import (
    DEFAULT_BENCHMARK_PROMPTS,
    evolve_weights,
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
        description="Evolve fitness weights via outer-loop meta-optimization"
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
        default=40,
        help="Inner lyric population size per benchmark prompt (default: 40)",
    )
    parser.add_argument(
        "--inner-generations",
        type=int,
        default=8,
        help="Inner lyric generations per benchmark prompt (default: 8)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="Top N couplets per prompt for meta-scoring (default: 4)",
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
        default="weights_evolved.json",
        help="Output JSON path for best weights (default: weights_evolved.json)",
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
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Benchmark prompts: {DEFAULT_BENCHMARK_PROMPTS}")
    logger.info(
        f"Outer: {args.outer_population} genomes x {args.outer_generations} gen | "
        f"Inner: {args.inner_population} pop x {args.inner_generations} gen"
    )

    results = evolve_weights(
        population_size=args.outer_population,
        generations=args.outer_generations,
        inner_population_size=args.inner_population,
        inner_generations=args.inner_generations,
        top_n_per_prompt=args.top_n,
        corpus_path=corpus_path,
        seed=args.seed,
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
            "benchmark_prompts": [list(p) for p in DEFAULT_BENCHMARK_PROMPTS],
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved best weights (meta_fitness={best_meta:.4f}) to {out_path}")
    print("\n--- Best evolved weights ---")
    for k, v in sorted(best_weights.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
