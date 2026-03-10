"""
Thin wrapper for syllable counting.

Imports count_syllables_word and count_syllables_line from
scripts.training.build_elite_kaggle_corpus_multi_stage when available,
otherwise reimplements using pronouncing.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from scripts.training.build_elite_kaggle_corpus_multi_stage import (
        count_syllables_line as _count_syllables_line,
        count_syllables_word as _count_syllables_word,
    )

    def count_syllables_word(word: str) -> int:
        return _count_syllables_word(word)

    def count_syllables_line(line: str) -> int:
        return _count_syllables_line(line)

except ImportError:
    import pronouncing

    _WORD_RE = re.compile(r"[a-zA-Z']+")

    @lru_cache(maxsize=50000)
    def count_syllables_word(word: str) -> int:
        phones = pronouncing.phones_for_word(word)
        if phones:
            return pronouncing.syllable_count(phones[0])
        groups = re.findall(r"[aeiouy]+", word.lower())
        return max(1, len(groups))

    def count_syllables_line(line: str) -> int:
        words = _WORD_RE.findall(line.lower())
        if not words:
            return 0
        return sum(count_syllables_word(w) for w in words)
