"""
Deterministic synthetic bad verses for metric benchmark (plan spec).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Tuple


def _sid(seed: int, offset: int) -> str:
    h = hashlib.sha256(f"{seed}:{offset}:bad_verse_v1".encode()).hexdigest()[:16]
    return f"syn_bad_{h}"


def generate_no_rhythm_verse(seed: int, offset: int) -> Tuple[str, List[str], Dict[str, Any]]:
    lines = [
        "the of at in on for to a an or but",
        "is was were be been being am are",
        "very quite rather somewhat fairly",
        "then thus hence whereby wherein wherein",
    ]
    return _sid(seed, offset), lines, {"generator": "bad_verse_v1", "kind": "no_rhythm", "seed_offset": offset}


def generate_repeat_spam_verse(seed: int, offset: int) -> Tuple[str, List[str], Dict[str, Any]]:
    phrase = "repeat the same phrase again"
    lines = [phrase, phrase, phrase, phrase]
    return _sid(seed, offset), lines, {"generator": "bad_verse_v1", "kind": "repeat_spam", "seed_offset": offset}


def generate_keyword_stuffing_verse(seed: int, offset: int) -> Tuple[str, List[str], Dict[str, Any]]:
    lines = [
        "money money hustle grind stack cash",
        "street street real real top top boss",
        "flow bars bars mic mic crown crown crown",
        "win win win never lose never stop",
    ]
    return _sid(seed, offset), lines, {"generator": "bad_verse_v1", "kind": "keyword_stuffing", "seed_offset": offset}


def generate_broken_syllable_verse(seed: int, offset: int) -> Tuple[str, List[str], Dict[str, Any]]:
    lines = [
        "I",
        "unexpectedly phenomenologically counterintuitively",
        "a",
        "supercalifragilisticexpialidocious ramifications notwithstanding",
    ]
    return _sid(seed, offset), lines, {"generator": "bad_verse_v1", "kind": "broken_syllable", "seed_offset": offset}


def generate_antiflow_list_verse(seed: int, offset: int) -> Tuple[str, List[str], Dict[str, Any]]:
    lines = [
        "item one: purchase receipt",
        "item two: parking validation",
        "item three: terms and conditions",
        "item four: appendix subsection b",
    ]
    return _sid(seed, offset), lines, {"generator": "bad_verse_v1", "kind": "anti_flow_list", "seed_offset": offset}


_GENERATORS = [
    generate_no_rhythm_verse,
    generate_repeat_spam_verse,
    generate_keyword_stuffing_verse,
    generate_broken_syllable_verse,
    generate_antiflow_list_verse,
]


def generate_synthetic_verses(count: int, seed: int) -> List[Dict[str, Any]]:
    """Return list of verse dicts ready for JSONL (no labels)."""
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for i in range(count):
        g = _GENERATORS[rng.randrange(len(_GENERATORS))]
        vid, lines, harvest_extra = g(seed, i)
        out.append(
            {
                "id": vid,
                "lyrics": lines,
                "n_bars": len(lines),
                "source": "synthetic",
                "harvest": harvest_extra,
                "unlabeled": True,
            }
        )
    return out
