"""
Line-level penalty for banned or discouraged phrases.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set


def _load_penalty_phrases() -> Set[str]:
    """Load penalty phrases from data/evo_rhyme/penalty_phrases.txt."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "evo_rhyme" / "penalty_phrases.txt"
    if not path.exists():
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().lower()
            if p and not p.startswith("#"):
                out.add(p)
    return out


DEFAULT_PENALTY_PHRASES = _load_penalty_phrases()


def score_line_penalty(
    line: str,
    penalty_phrases: Set[str] | None = None,
) -> float:
    """
    Penalty for lines containing banned or discouraged phrases.

    Returns a penalty in [0, 1]; 0 = no penalty, 1 = full penalty.
    """
    penalty_phrases = penalty_phrases or DEFAULT_PENALTY_PHRASES
    if not penalty_phrases:
        return 0.0
    lower = line.lower()
    for phrase in penalty_phrases:
        if phrase in lower:
            return 1.0
    return 0.0


def apply_line_penalty(
    lines: List[str],
    penalty_phrases: Set[str] | None = None,
) -> float:
    """
    Aggregate penalty across multiple lines.

    Returns the fraction of lines that contain a penalty phrase.
    """
    penalty_phrases = penalty_phrases or DEFAULT_PENALTY_PHRASES
    if not lines or not penalty_phrases:
        return 0.0
    hit_count = sum(
        1 for ln in lines
        if score_line_penalty(ln, penalty_phrases) > 0
    )
    return hit_count / len(lines)


_NGRAM_INDEX: Optional[Dict[str, float]] = None


def _build_ngram_index(corpus_path: Optional[str] = None) -> Dict[str, float]:
    """Build normalized n-gram frequency index from corpus file.

    Loads data/elite_kaggle_corpus_clean.txt and computes 3-gram and 4-gram
    frequencies. Returns dict mapping n-gram -> frequency (0-1 normalized).
    """
    global _NGRAM_INDEX
    if _NGRAM_INDEX is not None:
        return _NGRAM_INDEX

    if corpus_path is None:
        root = Path(__file__).resolve().parents[2]
        corpus_path = str(root / "data" / "elite_kaggle_corpus_clean.txt")

    path = Path(corpus_path)
    if not path.exists():
        _NGRAM_INDEX = {}
        return _NGRAM_INDEX

    word_re = re.compile(r"[a-z']+")
    counter: Counter = Counter()
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = word_re.findall(line.lower())
            for n in (3, 4):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    counter[gram] += 1
                    total += 1

    if total == 0:
        _NGRAM_INDEX = {}
        return _NGRAM_INDEX

    max_count = max(counter.values()) if counter else 1
    _NGRAM_INDEX = {gram: count / max_count for gram, count in counter.items()}
    return _NGRAM_INDEX


def get_ngram_index(corpus_path: Optional[str] = None) -> Dict[str, float]:
    """Return the n-gram index (built on first call)."""
    return _build_ngram_index(corpus_path)


_RAP_CLICHE_FRAGMENTS: Set[str] = {
    "survival in my blood",
    "survival of the fittest",
    "king of the jungle",
    "phoenix in flames",
    "rise from the ashes",
    "blood on my hands",
    "pressure on my chest",
    "mask on my face",
    "crown on my head",
    "fire in my veins",
    "weight on my shoulders",
    "heart of the struggle",
    "diamond in the rough",
    "lion in the jungle",
    "streets made me",
    "came from nothing",
    "started from the bottom",
    "real ones know",
    "they don't understand",
    "born to be king",
    "throne is mine",
    "can't stop won't stop",
    "hustle never sleeps",
    "grind don't stop",
    "never fold under pressure",
    "eyes on the prize",
    "keep it real",
    "spitting fire",
    "i'm the one they fear",
    "never taste defeat",
    "built different",
    "cut from a different cloth",
}


def score_cliche_penalty(
    lines: List[str],
    ngram_index: Optional[Dict[str, float]] = None,
    threshold: float = 0.3,
) -> float:
    """Continuous cliche penalty [0,1] combining corpus n-gram frequency
    and a hardcoded set of overused rap phrases.

    Returns the average per-line cliche score (0 = novel, 1 = all cliches).
    """
    if ngram_index is None:
        ngram_index = get_ngram_index()
    if not lines:
        return 0.0

    word_re = re.compile(r"[a-z']+")
    line_scores: List[float] = []

    for line in lines:
        line_lower = line.lower()
        words = word_re.findall(line_lower)

        # Hardcoded cliché fragment match
        frag_hit = any(frag in line_lower for frag in _RAP_CLICHE_FRAGMENTS)

        if len(words) < 3:
            line_scores.append(0.8 if frag_hit else 0.0)
            continue

        # Corpus n-gram frequency
        freqs: List[float] = []
        if ngram_index:
            for n in (3, 4):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    freq = ngram_index.get(gram, 0.0)
                    freqs.append(freq)

        if freqs:
            avg_freq = sum(freqs) / len(freqs)
            corpus_score = min(1.0, avg_freq / max(threshold, 0.01))
        else:
            corpus_score = 0.0

        score = max(corpus_score, 0.8 if frag_hit else 0.0)
        line_scores.append(score)

    return sum(line_scores) / len(line_scores) if line_scores else 0.0
