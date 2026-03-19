#!/usr/bin/env python
"""
audit_template_contamination.py

Plan 2: Sample verses from archive/top_candidates and report % coherent vs mixed-topic.
Helps track template contamination (e.g. empire/crown verses with "shot in the leg" lines).

Usage:
    python scripts/audit_template_contamination.py data/evo_rhyme/runs/qd_20260318_201102/top_candidates.json
    python scripts/audit_template_contamination.py data/evo_rhyme/runs/qd_20260318_201102/archive.json --sample 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Known orphan phrases (must match evo_rhyme.constraints.DEFAULT_ORPHAN_PHRASES)
ORPHAN_PHRASES = frozenset({
    "shot in the leg",
    "played around",
    "caught a shot",
})


def _load_candidates(path: Path, sample: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load verses from top_candidates.json or archive.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        # archive.json: {"cells": {...}} or similar
        if "cells" in data:
            cells = data["cells"]
            candidates = []
            for coord, entry in cells.items():
                if isinstance(entry, dict) and "lines" in entry:
                    candidates.append(entry)
                elif isinstance(entry, list):
                    candidates.append({"lines": entry})
        elif "entries" in data:
            candidates = data["entries"]
        else:
            candidates = list(data.values()) if data else []
    else:
        candidates = []

    if sample and len(candidates) > sample:
        import random
        random.seed(42)
        candidates = random.sample(candidates, sample)
    return candidates


def _classify_verse(
    entry: Dict[str, Any],
    theme_keywords: Optional[Set[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Classify verse as: coherent, mixed_topic, or garbled.
    Returns (label, reason).
    """
    lines = entry.get("lines", [])
    if not lines or len(lines) < 4:
        return "unknown", "fewer than 4 lines"

    line_texts = [l if isinstance(l, str) else str(l) for l in lines]
    scores = entry.get("scores") or {}
    coherence = scores.get("coherence", 0.0)

    word_re = re.compile(r"[A-Za-z']+")

    # Garbled: known broken patterns
    garbled_patterns = [
        r"\b(\w+)\s+\1\b",  # duplicate word
        r"\bi\s+m\b",
        r"shave never dose",
        r"flow lai ",
        r"crown the empire with a po sharp",
        r"po sharp as a peg",
    ]
    for line in line_texts:
        line_lower = line.lower()
        for pat in garbled_patterns:
            if re.search(pat, line_lower, re.IGNORECASE):
                return "garbled", f"garbled pattern in: {line[:50]}..."

    # Orphan: line with orphan phrase and no theme (when theme given)
    if theme_keywords:
        for i, line in enumerate(line_texts):
            line_lower = line.lower()
            line_words = set(word_re.findall(line_lower))
            has_theme = bool(line_words & theme_keywords)
            if has_theme:
                continue
            for phrase in ORPHAN_PHRASES:
                if phrase in line_lower:
                    return "mixed_topic", f"orphan phrase '{phrase}' in line {i+1}"

    # Theme consistency: when theme given, count lines with theme
    if theme_keywords:
        lines_with_theme = sum(
            1 for line in line_texts
            if set(word_re.findall(line.lower())) & theme_keywords
        )
        if lines_with_theme >= 2 and lines_with_theme < len(line_texts):
            return "mixed_topic", f"only {lines_with_theme}/4 lines have theme"

    # Coherence threshold
    if coherence < 0.45:
        return "mixed_topic", f"low coherence {coherence:.2f}"
    if coherence >= 0.55:
        return "coherent", None
    return "borderline", f"coherence {coherence:.2f}"


def audit(
    path: Path,
    theme: Optional[str] = None,
    sample: Optional[int] = None,
    min_fitness: float = 0.0,
) -> Dict[str, Any]:
    """Run audit and return stats."""
    theme_keywords: Optional[Set[str]] = None
    if theme:
        theme_keywords = set(w.strip().lower() for w in theme.split(",") if w.strip())

    candidates = _load_candidates(path, sample)
    if min_fitness > 0:
        candidates = [c for c in candidates if (c.get("fitness") or 0) >= min_fitness]

    counts: Dict[str, int] = {"coherent": 0, "mixed_topic": 0, "garbled": 0, "borderline": 0, "unknown": 0}
    examples: Dict[str, List[Dict[str, Any]]] = {
        "coherent": [],
        "mixed_topic": [],
        "garbled": [],
    }
    coherence_vals: List[float] = []

    for entry in candidates:
        label, reason = _classify_verse(entry, theme_keywords)
        counts[label] = counts.get(label, 0) + 1
        coherence_vals.append((entry.get("scores") or {}).get("coherence", 0.0))
        if label in examples and len(examples[label]) < 3:
            examples[label].append({
                "lines": entry.get("lines", [])[:2],
                "coherence": (entry.get("scores") or {}).get("coherence"),
                "fitness": entry.get("fitness"),
                "reason": reason,
            })

    n = len(candidates)
    report = {
        "path": str(path),
        "total_verses": n,
        "theme": theme,
        "min_fitness": min_fitness,
        "counts": counts,
        "pct_coherent": 100 * counts.get("coherent", 0) / n if n else 0,
        "pct_mixed_topic": 100 * counts.get("mixed_topic", 0) / n if n else 0,
        "pct_garbled": 100 * counts.get("garbled", 0) / n if n else 0,
        "pct_borderline": 100 * counts.get("borderline", 0) / n if n else 0,
        "mean_coherence": sum(coherence_vals) / len(coherence_vals) if coherence_vals else 0,
        "examples": examples,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit template contamination in verse archive/top_candidates"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to top_candidates.json or archive.json",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Comma-separated theme keywords (e.g. crown,empire)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Max verses to sample (default: all)",
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.0,
        help="Only include verses with fitness >= this",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} not found", file=sys.stderr)
        return 1

    report = audit(args.path, args.theme, args.sample, args.min_fitness)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    # Human-readable
    print("=" * 60)
    print("Template Contamination Audit")
    print("=" * 60)
    print(f"Path: {report['path']}")
    print(f"Total verses: {report['total_verses']}")
    if report.get("theme"):
        print(f"Theme: {report['theme']}")
    print()
    print("Classification:")
    print(f"  Coherent:    {report['counts'].get('coherent', 0):4d} ({report['pct_coherent']:.1f}%)")
    print(f"  Mixed-topic: {report['counts'].get('mixed_topic', 0):4d} ({report['pct_mixed_topic']:.1f}%)")
    print(f"  Garbled:     {report['counts'].get('garbled', 0):4d} ({report['pct_garbled']:.1f}%)")
    print(f"  Borderline:  {report['counts'].get('borderline', 0):4d} ({report['pct_borderline']:.1f}%)")
    print()
    print(f"Mean coherence: {report['mean_coherence']:.3f}")
    print()
    if report.get("examples", {}).get("mixed_topic"):
        print("Example mixed-topic verses:")
        for ex in report["examples"]["mixed_topic"][:2]:
            print(f"  - {ex.get('reason', '')}")
            for line in ex.get("lines", [])[:2]:
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
