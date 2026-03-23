"""
Block builder: assembles 4-bar verses from a CoupletArchive.

Selects couplets from the archive by rhyme scheme (AABB, ABAB, etc.)
and combines them into 4-line verses for hierarchical evolution.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from evo_rhyme.couplet_archive import CoupletArchive, ScoredCouplet
from evo_rhyme.individual import VerseIndividual

logger = logging.getLogger(__name__)


def _select_couplet_pair(
    archive: CoupletArchive,
    scheme: str = "AABB",
    used_pairs: Optional[Set[frozenset]] = None,
) -> Optional[tuple[ScoredCouplet, ScoredCouplet]]:
    """Select two couplets from different rhyme groups for 4-bar assembly.

    For AABB: need 2 couplets from 2 different groups (A-A and B-B).
    For ABAB: same, but lines will be interleaved.
    Returns (couplet_a, couplet_b) or None if not enough variety.
    """
    if used_pairs is None:
        used_pairs = set()

    pairs = archive.eligible_pairs(min_per_group=1)
    if not pairs:
        return None

    def pair_quality(gk1: tuple, gk2: tuple) -> float:
        g1 = archive.get_group(gk1)
        g2 = archive.get_group(gk2)
        if not g1 or not g2:
            return 0.0
        f1 = sum(c.fitness for c in g1[:3]) / min(3, len(g1))
        f2 = sum(c.fitness for c in g2[:3]) / min(3, len(g2))
        return f1 + f2

    pairs = [(p, pair_quality(p[0], p[1])) for p in pairs]
    pairs.sort(key=lambda x: x[1], reverse=True)

    for (gk1, gk2), _ in pairs:
        pair_key = frozenset([gk1, gk2])
        if pair_key in used_pairs and len(pairs) > 1:
            continue

        c1_list = archive.sample_from_group(gk1, k=1)
        c2_list = archive.sample_from_group(gk2, k=1)
        if not c1_list or not c2_list:
            continue

        return (c1_list[0], c2_list[0])

    return None


def build_4bar_from_couplets(
    couplet_archive: CoupletArchive,
    scheme: str = "AABB",
    theme_keywords: Optional[List[str]] = None,
    used_pairs: Optional[Set[frozenset]] = None,
) -> Optional[VerseIndividual]:
    """Assemble a 4-bar verse from two couplets in the archive.

    For AABB: [c1.line1, c1.line2, c2.line1, c2.line2]
    For ABAB: [c1.line1, c2.line1, c1.line2, c2.line2]
    For ABBA: [c1.line1, c2.line1, c2.line2, c1.line2]
    For ABCB, AABA, AAAA: extended from scheme pattern.
    """
    if used_pairs is None:
        used_pairs = set()

    scheme = (scheme * 2)[:4]  # ensure at least 4 chars

    result = _select_couplet_pair(couplet_archive, scheme, used_pairs)
    if result is None:
        return None

    c1, c2 = result

    # Track used pair for diversity
    pair_key = frozenset([c1.group_key(), c2.group_key()])
    used_pairs.add(pair_key)

    unique_letters = list(dict.fromkeys(scheme[:4]))
    if len(unique_letters) == 1:
        # AAAA: all same rhyme, use one couplet twice or both - use both
        lines = [c1.line1, c1.line2, c2.line1, c2.line2]
    elif scheme[:4] == "AABB":
        lines = [c1.line1, c1.line2, c2.line1, c2.line2]
    elif scheme[:4] == "ABAB":
        lines = [c1.line1, c2.line1, c1.line2, c2.line2]
    elif scheme[:4] == "ABBA":
        lines = [c1.line1, c2.line1, c2.line2, c1.line2]
    elif scheme[:4] == "ABCB":
        # A B C B: c1 gives A,C; c2 gives B,B
        lines = [c1.line1, c2.line1, c1.line2, c2.line2]
    elif scheme[:4] == "AABA":
        # A A B A: c1 gives A,A; c2 gives B - need 3 from c1, 1 from c2
        lines = [c1.line1, c1.line2, c2.line1, c1.line2]
    else:
        lines = [c1.line1, c1.line2, c2.line1, c2.line2]

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={
            "origin": "couplet_assembly",
            "parent_couplet_fitnesses": [c1.fitness, c2.fitness],
        },
    )


def build_4bar_batch(
    couplet_archive: CoupletArchive,
    count: int,
    scheme: str = "AABB",
    theme_keywords: Optional[List[str]] = None,
) -> List[VerseIndividual]:
    """Build multiple 4-bar verses from the archive, maximizing diversity.

    Tracks used couplet pairs to avoid assembling identical-sounding verses.
    """
    verses: List[VerseIndividual] = []
    used_pairs: Set[frozenset] = set()
    attempts = 0
    max_attempts = count * 5

    while len(verses) < count and attempts < max_attempts:
        attempts += 1
        verse = build_4bar_from_couplets(
            couplet_archive,
            scheme=scheme,
            theme_keywords=theme_keywords,
            used_pairs=used_pairs,
        )
        if verse is not None:
            verses.append(verse)

    logger.info(
        "Built %d 4-bar verses from couplet archive (%d attempts, %d pairs used)",
        len(verses),
        attempts,
        len(used_pairs),
    )
    return verses
