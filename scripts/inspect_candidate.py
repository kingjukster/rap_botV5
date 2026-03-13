#!/usr/bin/env python
"""
inspect_candidate.py

CLI to analyze and score a couplet, printing full breakdown.

Usage:
    python scripts/inspect_candidate.py --line1 "..." --line2 "..."
    python scripts/inspect_candidate.py --file path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_safe_path(user_path: str, base: Path, desc: str = "file") -> Path:
    """Resolve path and ensure it stays under base (prevents path traversal)."""
    path = Path(user_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError:
        print(f"Error: {desc} path must be within project directory", file=sys.stderr)
        sys.exit(1)
    return path


from evo_rhyme.constraints import passes_constraints
from evo_rhyme.fitness import DEFAULT_WEIGHTS, compute_fitness, score_couplet
from evo_rhyme.individual import CoupletIndividual, analyze_individual

MAX_LINE_LENGTH = 500


def main():
    parser = argparse.ArgumentParser(description="Inspect couplet: analyze and score")
    parser.add_argument("--line1", type=str, help="First line of couplet")
    parser.add_argument("--line2", type=str, help="Second line of couplet")
    parser.add_argument(
        "--file",
        type=str,
        help="JSON file with line1, line2 (or candidates array)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Comma-separated theme keywords for semantic score",
    )
    args = parser.parse_args()

    individuals: List[CoupletIndividual] = []

    if args.file:
        path = _resolve_safe_path(args.file, ROOT, "file")
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "candidates" in data:
            for c in data["candidates"]:
                l1, l2 = c.get("line1", ""), c.get("line2", "")
                if len(l1) > MAX_LINE_LENGTH or len(l2) > MAX_LINE_LENGTH:
                    print(f"Error: line length exceeds max {MAX_LINE_LENGTH} chars", file=sys.stderr)
                    sys.exit(1)
                individuals.append(CoupletIndividual(line1=l1, line2=l2))
        elif "line1" in data and "line2" in data:
            l1, l2 = data["line1"], data["line2"]
            if len(l1) > MAX_LINE_LENGTH or len(l2) > MAX_LINE_LENGTH:
                print(f"Error: line length exceeds max {MAX_LINE_LENGTH} chars", file=sys.stderr)
                sys.exit(1)
            individuals.append(CoupletIndividual(line1=l1, line2=l2))
        else:
            print("JSON must have 'line1'/'line2' or 'candidates' array", file=sys.stderr)
            sys.exit(1)
    elif args.line1 is not None and args.line2 is not None:
        if len(args.line1) > MAX_LINE_LENGTH or len(args.line2) > MAX_LINE_LENGTH:
            print(f"Error: line length exceeds max {MAX_LINE_LENGTH} chars", file=sys.stderr)
            sys.exit(1)
        individuals.append(
            CoupletIndividual(line1=args.line1, line2=args.line2)
        )
    else:
        parser.error("Provide --line1 and --line2, or --file")

    theme_keywords = [w.strip() for w in args.theme.split(",") if w.strip()] or None
    kw = set(w.lower() for w in theme_keywords) if theme_keywords else None

    for idx, ind in enumerate(individuals):
        if len(individuals) > 1:
            print(f"\n=== Candidate {idx + 1} ===")
        analyze_individual(ind)
        scores = score_couplet(ind, prompt_keywords=kw)
        fitness = compute_fitness(scores, DEFAULT_WEIGHTS)
        passes = passes_constraints(ind)

        print(f"Line 1: {ind.line1}")
        print(f"Line 2: {ind.line2}")
        print(f"\nFitness: {fitness:.4f}")
        print(f"Passes constraints: {passes}")
        print("\nScore breakdown:")
        for k, v in scores.items():
            w = DEFAULT_WEIGHTS.get(k, 0)
            contrib = w * v if k in DEFAULT_WEIGHTS else 0
            print(f"  {k}: {v:.4f} (weight={w:.2f}, contrib={contrib:.4f})")
        print(f"\nFeatures:")
        if ind.features1:
            print(f"  Line1 syllables: {ind.features1.syllable_count}, tokens: {ind.features1.tokens}")
        if ind.features2:
            print(f"  Line2 syllables: {ind.features2.syllable_count}, tokens: {ind.features2.tokens}")


if __name__ == "__main__":
    main()
