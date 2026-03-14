"""
Line archive for two-tier evolution: stores scored lines organized by rhyme group.

Each line is scored independently on fluency, semantic relevance, novelty, and
rhyme chain potential. Lines are grouped by end-word rhyme tail, enabling fast
lookup of rhyming pairs for verse assembly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from evo_rhyme.phonetics import (
    extract_rhyme_tail,
    syllable_count_line,
    tokenize_line,
    phones_for_word,
)


@dataclass
class ScoredLine:
    """A single scored line with all quality dimensions."""
    text: str
    end_tail: str
    end_word: str
    syllables: int
    fluency: float = 0.0
    lm_fluency: float = 0.0
    semantic: float = 0.0
    novelty: float = 0.0
    chain_potential: float = 0.0
    composite: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_text(text: str) -> "ScoredLine":
        """Create an unscored ScoredLine from raw text."""
        tokens = tokenize_line(text)
        end_word = tokens[-1] if tokens else ""
        tail = extract_rhyme_tail(end_word) if end_word else None
        return ScoredLine(
            text=text,
            end_tail=tail or "__unknown__",
            end_word=end_word.lower(),
            syllables=syllable_count_line(text),
        )


LINE_COMPOSITE_WEIGHTS: Dict[str, float] = {
    "fluency": 0.15,
    "lm_fluency": 0.20,
    "semantic": 0.15,
    "novelty": 0.15,
    "chain_potential": 0.10,
}


def compute_line_composite(line: ScoredLine, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute weighted composite score for a line."""
    w = weights or LINE_COMPOSITE_WEIGHTS
    total = 0.0
    for key, weight in w.items():
        total += weight * getattr(line, key, 0.0)
    return total


class LineArchive:
    """Archive of scored lines organized by rhyme group (end tail).

    Each rhyme group stores up to max_per_group lines, sorted by composite score.
    Supports sampling rhyming pairs for verse assembly.
    """

    def __init__(self, max_per_group: int = 200):
        self._groups: Dict[str, List[ScoredLine]] = defaultdict(list)
        self._max_per_group = max_per_group
        self._seen_texts: Set[str] = set()

    def add(self, line: ScoredLine) -> bool:
        """Add a line to the archive. Returns True if it was kept."""
        key = line.text.strip().lower()
        if key in self._seen_texts:
            return False
        self._seen_texts.add(key)

        group = self._groups[line.end_tail]
        if len(group) < self._max_per_group:
            group.append(line)
            group.sort(key=lambda l: l.composite, reverse=True)
            return True
        elif line.composite > group[-1].composite:
            evicted = group.pop()
            self._seen_texts.discard(evicted.text.strip().lower())
            group.append(line)
            group.sort(key=lambda l: l.composite, reverse=True)
            return True
        return False

    def add_batch(self, lines: List[ScoredLine]) -> int:
        """Add multiple lines. Returns count of lines actually kept."""
        return sum(1 for line in lines if self.add(line))

    def get_group(self, end_tail: str) -> List[ScoredLine]:
        """Get all lines in a rhyme group, sorted by composite score."""
        return list(self._groups.get(end_tail, []))

    def sample_rhyming_pair(
        self,
        tail: Optional[str] = None,
        min_group_size: int = 2,
    ) -> Optional[Tuple[ScoredLine, ScoredLine]]:
        """Sample two lines from the same rhyme group.

        If tail is None, picks a random group with >= min_group_size lines.
        Uses fitness-proportional selection within the group.
        """
        if tail is not None:
            group = self._groups.get(tail, [])
            if len(group) < min_group_size:
                return None
        else:
            eligible = [
                t for t, g in self._groups.items()
                if len(g) >= min_group_size and t != "__unknown__"
            ]
            if not eligible:
                return None
            tail = random.choice(eligible)
            group = self._groups[tail]

        if len(group) < 2:
            return None

        weights = [max(0.01, l.composite) for l in group]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx1 = random.choices(range(len(group)), weights=probs, k=1)[0]

        remaining_probs = list(probs)
        remaining_probs[idx1] = 0.0
        total2 = sum(remaining_probs)
        if total2 < 1e-9:
            idx2 = (idx1 + 1) % len(group)
        else:
            remaining_probs = [p / total2 for p in remaining_probs]
            idx2 = random.choices(range(len(group)), weights=remaining_probs, k=1)[0]

        return (group[idx1], group[idx2])

    def top_k_per_group(self, k: int) -> Dict[str, List[ScoredLine]]:
        """Return top k lines from each rhyme group."""
        return {
            tail: group[:k]
            for tail, group in self._groups.items()
            if group
        }

    def all_tails(self) -> List[str]:
        """Return all rhyme group keys with at least one line."""
        return [t for t, g in self._groups.items() if g and t != "__unknown__"]

    def eligible_tails(self, min_size: int = 2) -> List[str]:
        """Return tails with at least min_size lines (suitable for pair sampling)."""
        return [
            t for t, g in self._groups.items()
            if len(g) >= min_size and t != "__unknown__"
        ]

    def size(self) -> int:
        """Total number of lines in the archive."""
        return sum(len(g) for g in self._groups.values())

    def group_count(self) -> int:
        """Number of non-empty rhyme groups."""
        return len([g for g in self._groups.values() if g])

    def top_k_global(self, k: int) -> List[ScoredLine]:
        """Return the top k lines across all groups by composite score."""
        all_lines: List[ScoredLine] = []
        for group in self._groups.values():
            all_lines.extend(group)
        all_lines.sort(key=lambda l: l.composite, reverse=True)
        return all_lines[:k]

    def sample_diverse_lines(self, n: int) -> List[ScoredLine]:
        """Sample n lines from different rhyme groups for diversity."""
        eligible = self.eligible_tails(min_size=1)
        if not eligible:
            return []
        random.shuffle(eligible)
        result: List[ScoredLine] = []
        for tail in eligible:
            if len(result) >= n:
                break
            group = self._groups[tail]
            if group:
                result.append(random.choice(group[:5]))
        return result

    def to_json(self) -> Dict[str, Any]:
        """Serialize archive to JSON-compatible dict."""
        return {
            "total_lines": self.size(),
            "group_count": self.group_count(),
            "groups": {
                tail: [
                    {
                        "text": l.text,
                        "end_tail": l.end_tail,
                        "composite": l.composite,
                        "fluency": l.fluency,
                        "semantic": l.semantic,
                        "novelty": l.novelty,
                    }
                    for l in lines[:10]
                ]
                for tail, lines in self._groups.items()
                if lines
            },
        }
