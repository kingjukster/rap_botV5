"""
evo_rhyme/ngram_fluency.py

Corpus-based n-gram fluency scoring. Builds bigram (and optionally trigram)
counts from a rap corpus and scores lines by phrase plausibility.
Reduces phonetically rewarded nonsense like "damn the cup" or "grad list chazer plan".
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens from line."""
    return [w.lower() for w in WORD_RE.findall(text.lower()) if w]


class CorpusNgramModel:
    """
    Bigram/trigram model built from corpus lines.
    Scores lines by fraction of adjacent word pairs (and triples) that appear in corpus.
    """

    def __init__(
        self,
        bigram_counts: Dict[Tuple[str, str], int],
        trigram_counts: Dict[Tuple[str, str, str], int],
        unigram_vocab: Set[str],
        min_bigram_count: int = 2,
    ):
        # Only count bigrams that appear min_bigram_count+ times (stricter fluency)
        self._bigrams: FrozenSet[Tuple[str, str]] = frozenset(
            b for b, c in bigram_counts.items() if c >= min_bigram_count
        )
        self._trigrams: FrozenSet[Tuple[str, str, str]] = frozenset(trigram_counts.keys())
        self._unigram_vocab = unigram_vocab
        self._bigram_counts = bigram_counts
        self._trigram_counts = trigram_counts

    @classmethod
    def from_lines(
        cls,
        corpus_lines: List[str],
        max_lines: int = 5000,
        min_bigram_count: int = 2,
    ) -> "CorpusNgramModel":
        """
        Build n-gram model from corpus. Uses first max_lines for efficiency.
        """
        bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        trigram_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        unigram_vocab: Set[str] = set()

        for line in corpus_lines[:max_lines]:
            tokens = _tokenize(line)
            if len(tokens) < 2:
                continue
            for w in tokens:
                unigram_vocab.add(w)
            for i in range(len(tokens) - 1):
                big = (tokens[i], tokens[i + 1])
                bigram_counts[big] += 1
            for i in range(len(tokens) - 2):
                tri = (tokens[i], tokens[i + 1], tokens[i + 2])
                trigram_counts[tri] += 1

        return cls(
            bigram_counts=dict(bigram_counts),
            trigram_counts=dict(trigram_counts),
            unigram_vocab=unigram_vocab,
            min_bigram_count=min_bigram_count,
        )

    def score_line(
        self,
        text: str,
        bigram_weight: float = 0.35,
        trigram_weight: float = 0.65,
    ) -> float:
        """
        Score a single line [0,1] by phrase plausibility.
        Trigrams weighted more heavily - "truck duck fuck" is less likely than
        "truck duck" + "duck fuck" separately. Bigrams require min_count in corpus.
        """
        tokens = _tokenize(text)
        if len(tokens) < 2:
            return 0.5  # neutral for single-word lines

        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        bigram_hits = sum(1 for b in bigrams if b in self._bigrams)
        bigram_score = bigram_hits / len(bigrams) if bigrams else 0.5

        if len(tokens) >= 3 and trigram_weight > 0:
            trigrams = [
                (tokens[i], tokens[i + 1], tokens[i + 2])
                for i in range(len(tokens) - 2)
            ]
            trigram_hits = sum(1 for t in trigrams if t in self._trigrams)
            trigram_score = trigram_hits / len(trigrams) if trigrams else 0.5
        else:
            trigram_score = 0.5

        return bigram_weight * bigram_score + trigram_weight * trigram_score

    def score_couplet(self, line1: str, line2: str) -> float:
        """Average score across both lines."""
        s1 = self.score_line(line1)
        s2 = self.score_line(line2)
        return (s1 + s2) / 2.0

    @property
    def vocab(self) -> Set[str]:
        """Corpus unigram vocabulary for lexical validity checks."""
        return self._unigram_vocab


# Module-level cache: avoid rebuilding on every score_couplet call
_CACHED_MODEL: Optional[CorpusNgramModel] = None
_CACHED_LINES_REF: Optional[object] = None


def get_ngram_model(corpus_lines: Optional[List[str]]) -> Optional[CorpusNgramModel]:
    """
    Get or build cached n-gram model from corpus_lines.
    Returns None if corpus_lines is empty or None.
    """
    global _CACHED_MODEL, _CACHED_LINES_REF
    if not corpus_lines or len(corpus_lines) < 2:
        return None
    if _CACHED_LINES_REF is not corpus_lines:
        _CACHED_MODEL = CorpusNgramModel.from_lines(corpus_lines)
        _CACHED_LINES_REF = corpus_lines
    return _CACHED_MODEL


def clear_ngram_cache() -> None:
    """Clear cached model (e.g. between runs with different corpus)."""
    global _CACHED_MODEL, _CACHED_LINES_REF
    _CACHED_MODEL = None
    _CACHED_LINES_REF = None
