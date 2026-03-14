"""
evo_rhyme/constraints.py

Constraint checking for couplet individuals. Rejects individuals that
violate syllable bounds, weak endings, repetition, or phonetic validity.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from evo_rhyme.individual import CoupletIndividual, LineFeatures, VerseIndividual
from evo_rhyme.phonetics import phones_for_word

# Weak end words (unstressed / filler) - reject as line endings
DEFAULT_WEAK_WORDS: Set[str] = frozenset({
    "a", "an", "and", "at", "be", "but", "by", "da",
    "for", "from", "go", "had", "he", "her", "him", "his",
    "i", "id", "im", "in", "is", "it", "me", "my",
    "no", "of", "on", "or", "our", "out", "she", "so",
    "that", "the", "them", "then", "there", "they", "this",
    "to", "was", "we", "ya", "yo", "you", "yall",
})

# Stopwords excluded from repetition check (only content words counted)
DEFAULT_REPETITION_STOPWORDS: Set[str] = frozenset({
    "a", "an", "and", "at", "be", "but", "by", "da",
    "for", "from", "go", "had", "he", "her", "him", "his",
    "i", "id", "im", "in", "is", "it", "me", "my",
    "no", "of", "on", "or", "our", "out", "she", "so",
    "that", "the", "them", "then", "there", "they", "this",
    "to", "was", "we", "ya", "yo", "you", "yall",
    "got", "like", "just", "all", "say", "said",
})


@dataclass
class ConstraintConfig:
    """Configuration for constraint checks."""
    min_syllables: int = 6
    max_syllables: int = 18
    min_words_per_line: int = 4
    weak_words: Set[str] = field(default_factory=lambda: set(DEFAULT_WEAK_WORDS))
    max_token_repeats: int = 2
    require_stressed_end: bool = True
    repetition_stopwords: Set[str] = field(
        default_factory=lambda: set(DEFAULT_REPETITION_STOPWORDS)
    )
    require_theme_presence: bool = False
    prompt_keywords: Optional[Set[str]] = None


def _end_word(line: str) -> str:
    """Extract final lexical word from a line."""
    import re
    words = re.findall(r"[A-Za-z']+", line.lower())
    return words[-1] if words else ""


def _check_min_words(
    f1: Optional[LineFeatures], f2: Optional[LineFeatures], config: ConstraintConfig
) -> Optional[str]:
    """Reject if either line has fewer than min_words_per_line tokens."""
    if not f1 or not f2:
        return "missing features"
    min_w = config.min_words_per_line
    if len(f1.tokens) < min_w:
        return f"line1 has {len(f1.tokens)} words (min {min_w})"
    if len(f2.tokens) < min_w:
        return f"line2 has {len(f2.tokens)} words (min {min_w})"
    return None


def _check_syllable_bounds(
    f1: Optional[LineFeatures], f2: Optional[LineFeatures], config: ConstraintConfig
) -> Optional[str]:
    """Reject if syllable count outside [min, max]."""
    if not f1 or not f2:
        return "missing features"
    if f1.syllable_count < config.min_syllables:
        return f"line1 syllable count {f1.syllable_count} < {config.min_syllables}"
    if f1.syllable_count > config.max_syllables:
        return f"line1 syllable count {f1.syllable_count} > {config.max_syllables}"
    if f2.syllable_count < config.min_syllables:
        return f"line2 syllable count {f2.syllable_count} < {config.min_syllables}"
    if f2.syllable_count > config.max_syllables:
        return f"line2 syllable count {f2.syllable_count} > {config.max_syllables}"
    return None


def _check_valid_pronunciation(
    individual: CoupletIndividual, config: ConstraintConfig
) -> Optional[str]:
    """Reject if end word has no valid pronunciation."""
    w1 = _end_word(individual.line1)
    w2 = _end_word(individual.line2)
    for label, word in [("line1", w1), ("line2", w2)]:
        if not word:
            return f"{label} has no end word"
        phones = phones_for_word(word)
        if not phones:
            return f"no valid pronunciation for end word '{word}' in {label}"
    return None


def _check_weak_end_words(
    individual: CoupletIndividual, config: ConstraintConfig
) -> Optional[str]:
    """Reject if end word is in weak list."""
    w1 = _end_word(individual.line1).lower()
    w2 = _end_word(individual.line2).lower()
    weak = config.weak_words
    if w1 in weak:
        return f"end word '{w1}' in weak list"
    if w2 in weak:
        return f"end word '{w2}' in weak list"
    return None


def _check_consecutive_duplicates(tokens: list, max_run: int = 1) -> Optional[str]:
    """Reject if any word (including stopwords) appears more than max_run times
    consecutively. Default max_run=1 means no consecutive duplicates allowed."""
    if len(tokens) < 2:
        return None
    run_len = 1
    for i in range(1, len(tokens)):
        if tokens[i].lower() == tokens[i - 1].lower():
            run_len += 1
            if run_len > max_run:
                return f"word '{tokens[i].lower()}' repeated {run_len}x consecutively"
        else:
            run_len = 1
    return None


def _check_repetition(
    individual: CoupletIndividual, config: ConstraintConfig
) -> Optional[str]:
    """Reject if same content word appears more than max_token_repeats times.
    Stopwords (the, a, in, etc.) are excluded from the content-word count,
    but consecutive duplicates of ANY word (including stopwords) are caught."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return None  # skip if not analyzed

    for tokens in (f1.tokens, f2.tokens):
        err = _check_consecutive_duplicates(tokens)
        if err:
            return err

    stopwords = config.repetition_stopwords
    content_tokens = [
        t.lower() for t in (f1.tokens + f2.tokens)
        if t.lower() not in stopwords
    ]
    counts = Counter(content_tokens)
    for tok, cnt in counts.items():
        if cnt > config.max_token_repeats:
            return f"content word '{tok}' appears {cnt} times (max {config.max_token_repeats})"
    return None


def _check_identical_lines(individual: CoupletIndividual) -> Optional[str]:
    """Reject if line1 and line2 are identical (trivial rhyme)."""
    if individual.line1.strip().lower() == individual.line2.strip().lower():
        return "line1 and line2 are identical"
    return None


def _check_near_duplicate(
    individual: CoupletIndividual, max_word_overlap: float = 0.78
) -> Optional[str]:
    """Reject if line1 and line2 share >max_word_overlap of words (near-duplicate)."""
    import re
    word_re = re.compile(r"[A-Za-z']+")
    t1 = set(word_re.findall(individual.line1.lower()))
    t2 = set(word_re.findall(individual.line2.lower()))
    if not t1 or not t2:
        return None
    overlap = len(t1 & t2) / max(len(t1), len(t2))
    if overlap > max_word_overlap:
        return f"lines are near-duplicate (word overlap {overlap:.2f} > {max_word_overlap})"
    return None


def _check_stressed_end_tail(
    individual: CoupletIndividual, config: ConstraintConfig
) -> Optional[str]:
    """Reject if no stressed vowel in end tail."""
    if not config.require_stressed_end:
        return None
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return None
    for label, feat in [("line1", f1.end_tail), ("line2", f2.end_tail)]:
        if not feat:
            return f"no end tail for {label}"
        if feat.stress == 0:
            return f"no stressed vowel in end tail of {label}"
    return None


def _resolve_config(config: Optional[Any]) -> ConstraintConfig:
    """Build ConstraintConfig from None, ConstraintConfig, or dict."""
    if config is None:
        return ConstraintConfig()
    if isinstance(config, ConstraintConfig):
        return config
    if isinstance(config, dict):
        cfg = ConstraintConfig()
        if "min_syllables" in config:
            cfg.min_syllables = int(config["min_syllables"])
        if "max_syllables" in config:
            cfg.max_syllables = int(config["max_syllables"])
        if "weak_words" in config:
            cfg.weak_words = set(config["weak_words"])
        if "max_token_repeats" in config:
            cfg.max_token_repeats = int(config["max_token_repeats"])
        if "require_stressed_end" in config:
            cfg.require_stressed_end = bool(config["require_stressed_end"])
        if "repetition_stopwords" in config:
            cfg.repetition_stopwords = set(config["repetition_stopwords"])
        if "min_words_per_line" in config:
            cfg.min_words_per_line = int(config["min_words_per_line"])
        if "require_theme_presence" in config:
            cfg.require_theme_presence = bool(config["require_theme_presence"])
        if "prompt_keywords" in config and config["prompt_keywords"]:
            cfg.prompt_keywords = set(w.lower() for w in config["prompt_keywords"])
        return cfg
    return ConstraintConfig()


def passes_constraints(
    individual: CoupletIndividual,
    config: Optional[Any] = None,
) -> bool:
    """
    Return True if individual passes all constraints, False otherwise.

    Rejects if:
    - syllable_count < 6 or > 18 for either line
    - no valid pronunciation for end word
    - end word in weak list
    - same token > 2 times
    - no stressed vowel in end tail
    """
    cfg = _resolve_config(config)
    # Ensure features are populated
    if not individual.features1 or not individual.features2:
        from evo_rhyme.individual import analyze_individual
        analyze_individual(individual)

    err = _check_syllable_bounds(individual.features1, individual.features2, cfg)
    if err:
        return False

    err = _check_min_words(individual.features1, individual.features2, cfg)
    if err:
        return False

    err = _check_valid_pronunciation(individual, cfg)
    if err:
        return False

    err = _check_weak_end_words(individual, cfg)
    if err:
        return False

    err = _check_identical_lines(individual)
    if err:
        return False

    err = _check_near_duplicate(individual)
    if err:
        return False

    err = _check_repetition(individual, cfg)
    if err:
        return False

    err = _check_stressed_end_tail(individual, cfg)
    if err:
        return False

    # Theme presence: at least one line must contain at least one keyword
    if cfg.require_theme_presence and cfg.prompt_keywords:
        words = set(re.findall(r"[A-Za-z']+", (individual.line1 + " " + individual.line2).lower()))
        if not (words & cfg.prompt_keywords):
            return False

    return True


# ---------------------------------------------------------------------------
# Verse constraints (4-line)
# ---------------------------------------------------------------------------


def _verse_end_word(line: str) -> str:
    """Extract final lexical word from a line."""
    import re
    words = re.findall(r"[A-Za-z']+", line.lower())
    return words[-1] if words else ""


def _check_verse_syllable_bounds(
    features: Optional[Any],
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any line has syllable count outside [min, max]."""
    if not features or len(features.syllable_counts) != 4:
        return "missing features or wrong line count"
    for i, sc in enumerate(features.syllable_counts):
        if sc < config.min_syllables:
            return f"line{i+1} syllable count {sc} < {config.min_syllables}"
        if sc > config.max_syllables:
            return f"line{i+1} syllable count {sc} > {config.max_syllables}"
    return None


def _check_verse_min_words(
    features: Optional[Any],
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any line has fewer than min_words_per_line tokens."""
    if not features or len(features.tokens_per_line) != 4:
        return "missing features or wrong line count"
    min_w = config.min_words_per_line
    for i, tokens in enumerate(features.tokens_per_line):
        if len(tokens) < min_w:
            return f"line{i+1} has {len(tokens)} words (min {min_w})"
    return None


def _check_verse_valid_pronunciation(
    individual: VerseIndividual,
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any end word has no valid pronunciation."""
    for i, line in enumerate(individual.lines):
        word = _verse_end_word(line)
        if not word:
            return f"line{i+1} has no end word"
        phones = phones_for_word(word)
        if not phones:
            return f"no valid pronunciation for end word '{word}' in line{i+1}"
    return None


def _check_verse_weak_end_words(
    individual: VerseIndividual,
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any end word is in weak list."""
    weak = config.weak_words
    for i, line in enumerate(individual.lines):
        w = _verse_end_word(line).lower()
        if w in weak:
            return f"end word '{w}' in weak list (line{i+1})"
    return None


def _check_verse_identical_lines(individual: VerseIndividual) -> Optional[str]:
    """Reject if any two lines are identical."""
    lines = [l.strip().lower() for l in individual.lines]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i] == lines[j]:
                return f"line{i+1} and line{j+1} are identical"
    return None


def _check_verse_stressed_end_tail(
    individual: VerseIndividual,
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any line has no stressed vowel in end tail."""
    if not config.require_stressed_end:
        return None
    f = individual.features
    if not f or len(f.end_tails) != 4:
        return None
    for i, feat in enumerate(f.end_tails):
        if not feat:
            return f"no end tail for line{i+1}"
        if feat.stress == 0:
            return f"no stressed vowel in end tail of line{i+1}"
    return None


def _check_verse_near_duplicate_lines(
    individual: VerseIndividual,
    max_word_overlap: float = 0.78,
) -> Optional[str]:
    """Reject if any two lines share >max_word_overlap of words (near-duplicate)."""
    import re
    word_re = re.compile(r"[A-Za-z']+")
    line_sets: list = []
    for line in individual.lines:
        words = set(word_re.findall(line.lower()))
        line_sets.append(words)
    for i in range(len(line_sets)):
        for j in range(i + 1, len(line_sets)):
            t1, t2 = line_sets[i], line_sets[j]
            if not t1 or not t2:
                continue
            overlap = len(t1 & t2) / max(len(t1), len(t2))
            if overlap > max_word_overlap:
                return f"lines {i+1} and {j+1} are near-duplicate (word overlap {overlap:.2f} > {max_word_overlap})"
    return None


def _check_verse_content_repetition(
    individual: VerseIndividual,
    config: ConstraintConfig,
) -> Optional[str]:
    """Reject if any content word appears more than max_token_repeats times across
    4 lines, or if any line has consecutive duplicate words."""
    f = individual.features
    if not f or len(f.tokens_per_line) != 4:
        return None

    for i, tokens in enumerate(f.tokens_per_line):
        err = _check_consecutive_duplicates(tokens)
        if err:
            return f"line{i+1}: {err}"

    stopwords = config.repetition_stopwords
    content_tokens = []
    for tokens in f.tokens_per_line:
        content_tokens.extend(
            t.lower() for t in tokens if t.lower() not in stopwords
        )
    counts = Counter(content_tokens)
    for tok, cnt in counts.items():
        if cnt > config.max_token_repeats:
            return f"content word '{tok}' appears {cnt} times (max {config.max_token_repeats})"
    return None


def passes_verse_constraints(
    individual: VerseIndividual,
    config: Optional[Any] = None,
) -> bool:
    """
    Return True if verse passes all constraints, False otherwise.

    Rejects if:
    - syllable_count outside [min, max] for any line
    - min_words_per_line not met for any line
    - no valid pronunciation for any end word
    - any end word in weak list
    - any two lines identical
    - no stressed vowel in any end tail
    """
    cfg = _resolve_config(config)
    if not individual.features or len(individual.features.syllable_counts) != 4:
        from evo_rhyme.individual import analyze_verse_individual
        analyze_verse_individual(individual)

    f = individual.features
    if not f or len(f.syllable_counts) != 4:
        return False

    err = _check_verse_syllable_bounds(f, cfg)
    if err:
        return False

    err = _check_verse_min_words(f, cfg)
    if err:
        return False

    err = _check_verse_valid_pronunciation(individual, cfg)
    if err:
        return False

    err = _check_verse_weak_end_words(individual, cfg)
    if err:
        return False

    err = _check_verse_identical_lines(individual)
    if err:
        return False

    err = _check_verse_stressed_end_tail(individual, cfg)
    if err:
        return False

    err = _check_verse_near_duplicate_lines(individual)
    if err:
        return False

    err = _check_verse_content_repetition(individual, cfg)
    if err:
        return False

    if cfg.require_theme_presence and cfg.prompt_keywords:
        all_words: set = set()
        for line in individual.lines:
            all_words.update(re.findall(r"[A-Za-z']+", line.lower()))
        if not (all_words & cfg.prompt_keywords):
            return False

    return True
