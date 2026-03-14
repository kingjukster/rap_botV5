"""
Verse builder: assembles verses from a LineArchive by rhyme scheme.

Selects lines from the archive organized by rhyme group, arranges them
according to the scheme (AABB, ABAB, etc.), and optionally smooths
transitions between couplets via LM rewriting.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from evo_rhyme.individual import VerseIndividual
from evo_rhyme.line_archive import LineArchive, ScoredLine
from evo_rhyme.phonetics import tokenize_line

logger = logging.getLogger(__name__)


def _select_rhyme_groups(
    archive: LineArchive,
    scheme: str,
    used_pairs: Optional[set] = None,
) -> Optional[Dict[str, str]]:
    """Select rhyme groups for each letter in the scheme.

    For AABB: need 2 groups (A and B), each with >= 2 lines.
    For ABAB: need 2 groups (A and B), each with >= 2 lines.
    Returns mapping like {"A": "AY1 T", "B": "EH1 D"} or None if not enough groups.
    """
    unique_letters = list(dict.fromkeys(scheme))
    eligible = archive.eligible_tails(min_size=2)
    if len(eligible) < len(unique_letters):
        return None

    if used_pairs is None:
        used_pairs = set()

    random.shuffle(eligible)

    def group_quality(tail: str) -> float:
        group = archive.get_group(tail)
        if not group:
            return 0.0
        return sum(l.composite for l in group[:5]) / min(5, len(group))

    eligible.sort(key=group_quality, reverse=True)

    selected: Dict[str, str] = {}
    for letter in unique_letters:
        for tail in eligible:
            if tail in selected.values():
                continue
            pair_key = frozenset(list(selected.values()) + [tail])
            if pair_key in used_pairs and len(eligible) > len(unique_letters) + 2:
                continue
            selected[letter] = tail
            break

    if len(selected) != len(unique_letters):
        return None

    return selected


def build_verse_from_archive(
    archive: LineArchive,
    scheme: str = "AABB",
    num_lines: int = 4,
    theme_keywords: Optional[List[str]] = None,
    used_pairs: Optional[set] = None,
) -> Optional[VerseIndividual]:
    """Assemble a verse from elite lines in the archive.

    For AABB: [A1, A2, B1, B2] where A1,A2 rhyme and B1,B2 rhyme.
    For ABAB: [A1, B1, A2, B2] with interleaved rhyme groups.
    """
    if len(scheme) < num_lines:
        scheme = (scheme * ((num_lines // len(scheme)) + 1))[:num_lines]

    groups = _select_rhyme_groups(archive, scheme, used_pairs)
    if groups is None:
        return None

    lines: List[str] = []
    used_lines: set = set()

    for i in range(num_lines):
        letter = scheme[i]
        tail = groups[letter]
        group = archive.get_group(tail)
        if not group:
            return None

        available = [l for l in group if l.text not in used_lines]
        if not available:
            available = group

        weights = [max(0.01, l.composite) for l in available[:20]]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(len(available[:20])), weights=probs, k=1)[0]
        selected = available[min(idx, len(available) - 1)]

        lines.append(selected.text)
        used_lines.add(selected.text)

    if len(lines) != num_lines:
        return None

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={"origin": "line_archive_assembly"},
    )


def _optimize_transition(
    lines: List[str],
    boundary_idx: int,
    lm_budget: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Smooth the transition between two couplets using LM rewrite.

    If lines[boundary_idx] and lines[boundary_idx+1] have low coherence,
    rewrite lines[boundary_idx] to flow better into lines[boundary_idx+1].
    """
    if lm_budget is None or lm_budget.get("remaining", 0) <= 0:
        return lines
    if boundary_idx < 0 or boundary_idx + 1 >= len(lines):
        return lines

    try:
        from evo_rhyme.scoring.coherence import score_coherence
        coherence = score_coherence([lines[boundary_idx], lines[boundary_idx + 1]])
        if coherence >= 0.4:
            return lines

        from evo_rhyme.mutation import get_rewriter
        rewriter = get_rewriter()
        tokens = tokenize_line(lines[boundary_idx])
        if not tokens:
            return lines
        rhyme_target = tokens[-1]

        candidates = rewriter.paraphrase(
            lines[boundary_idx],
            preserve_end_rhyme=True,
            syllable_range=(6, 18),
        )
        lm_budget["remaining"] = lm_budget.get("remaining", 0) - 1

        if candidates:
            best_line = None
            best_score = coherence
            for cand in candidates:
                try:
                    s = score_coherence([cand, lines[boundary_idx + 1]])
                    if s > best_score:
                        best_score = s
                        best_line = cand
                except Exception:
                    pass
            if best_line:
                new_lines = list(lines)
                new_lines[boundary_idx] = best_line
                return new_lines

        return lines
    except Exception:
        logger.warning("_optimize_transition failed", exc_info=True)
        return lines


def build_verse_batch(
    archive: LineArchive,
    count: int,
    scheme: str = "AABB",
    num_lines: int = 4,
    optimize_transitions: bool = True,
    lm_budget: Optional[Dict[str, int]] = None,
) -> List[VerseIndividual]:
    """Build multiple verses from the archive, maximizing diversity.

    Tracks used rhyme group pairs to avoid assembling identical-sounding verses.
    Optionally optimizes transitions between couplets.
    """
    verses: List[VerseIndividual] = []
    used_pairs: set = set()
    attempts = 0
    max_attempts = count * 5

    while len(verses) < count and attempts < max_attempts:
        attempts += 1
        verse = build_verse_from_archive(
            archive, scheme, num_lines,
            used_pairs=used_pairs,
        )
        if verse is None:
            continue

        if len(verse.lines) == num_lines:
            tails = []
            for line in verse.lines:
                tokens = tokenize_line(line)
                if tokens:
                    from evo_rhyme.phonetics import extract_rhyme_tail
                    tail = extract_rhyme_tail(tokens[-1])
                    if tail:
                        tails.append(tail)
            if tails:
                used_pairs.add(frozenset(tails))

        if optimize_transitions and num_lines >= 4:
            new_lines = _optimize_transition(
                verse.lines, 1, lm_budget,
            )
            if new_lines != verse.lines:
                verse = VerseIndividual(
                    lines=new_lines,
                    features=None,
                    scores=None,
                    fitness=None,
                    metadata={"origin": "line_archive_assembly_optimized"},
                )

        verses.append(verse)

    logger.info(
        "Built %d verses from archive (%d attempts, %d rhyme pairs used)",
        len(verses), attempts, len(used_pairs),
    )
    return verses
