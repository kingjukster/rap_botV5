"""
Penalty scoring for weak line endings and repetition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

WORD_RE = re.compile(r"[A-Za-z']+")


def _load_weak_endings() -> Set[str]:
    """Load weak endings from data/evo_rhyme/weak_endings.txt."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "evo_rhyme" / "weak_endings.txt"
    if not path.exists():
        return {"the", "and", "it", "me", "you", "that", "with", "on"}
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                out.add(w)
    return out


DEFAULT_WEAK_ENDINGS = _load_weak_endings()


def weak_tail_penalty(
    line: str,
    weak_endings: Set[str] | None = None,
) -> float:
    """
    Penalty for lines ending with weak words (e.g. the, and, it).

    Returns a penalty in [0, 1]; 0 = no penalty, 1 = full penalty.
    """
    weak_endings = weak_endings or DEFAULT_WEAK_ENDINGS
    tokens = WORD_RE.findall(line.lower())
    if not tokens:
        return 0.0
    last = re.sub(r"[^a-z']", "", tokens[-1]).strip("'")
    if last in weak_endings:
        return 1.0
    return 0.0


def repetition_penalty(lines: List[str]) -> float:
    """
    Penalty for repeated lines (identical or near-identical).

    Returns a penalty in [0, 1]; 0 = no penalty, 1 = full penalty.
    """
    if not lines or len(lines) < 2:
        return 0.0
    seen: Set[str] = set()
    dupes = 0
    for ln in lines:
        norm = " ".join(ln.lower().split()).strip()
        if not norm:
            continue
        if norm in seen:
            dupes += 1
        seen.add(norm)
    if not seen:
        return 0.0
    return min(1.0, dupes / max(1, len(seen)))
