#!/usr/bin/env python3
"""
Improve an existing verse: load 4 lines, run short evolution from seed, output top variants.

Usage:
  python scripts/improve_verse.py --input verse.txt --generations 5 --top 3
  echo "Line one\nLine two\nLine three\nLine four" | python scripts/improve_verse.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evolve improved versions of a 4-line verse")
    ap.add_argument("--input", type=str, default=None, help="Path to file with 4 lines (or stdin)")
    ap.add_argument("--generations", type=int, default=5, help="Evolution generations")
    ap.add_argument("--population", type=int, default=25, help="Population size")
    ap.add_argument("--top", type=int, default=5, help="Number of top variants to print")
    ap.add_argument("--scheme", type=str, default="AABB", help="Rhyme scheme")
    ap.add_argument("--theme", type=str, default="", help="Comma-separated theme keywords")
    args = ap.parse_args()

    if args.input:
        path = Path(args.input)
        if not path.is_absolute():
            path = ROOT / path
        text = path.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 4:
        print("Need at least 4 non-empty lines.", file=sys.stderr)
        return 1
    seed_lines = lines[:4]

    from evo_rhyme.verse_evolution import evolve_from_seed_verse

    prompt_keywords = set(w.strip() for w in args.theme.split(",") if w.strip()) or None
    population = evolve_from_seed_verse(
        seed_lines=seed_lines,
        generations=args.generations,
        population_size=args.population,
        objective_weights=None,
        scheme=args.scheme,
        prompt_keywords=prompt_keywords,
    )

    for i, ind in enumerate(population[: args.top]):
        fitness = ind.fitness or 0.0
        print(f"--- #{i + 1} fitness={fitness:.4f} ---")
        for line in ind.lines:
            print(line)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
