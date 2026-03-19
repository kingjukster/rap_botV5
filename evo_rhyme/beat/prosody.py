from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class SyllableUnit:
    text: str
    word: str
    stress: float  # 1.0 strong, 0.6 secondary, 0.25 weak fallback
    syllable_index_in_word: int
    is_word_final: bool
    is_line_final: bool
    is_rhyme_zone: bool  # useful for bar/line ending rewards


def tokenize_words(line: str) -> List[str]:
    return _WORD_RE.findall(line.lower())


def simple_fallback_syllable_count(word: str) -> int:
    """
    Very rough fallback if CMU pronouncing is unavailable or word missing.
    """
    word = word.lower()
    if not word:
        return 1

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1

    return max(1, count)


def get_pronunciations(word: str) -> List[str]:
    if pronouncing is None:
        return []
    try:
        return pronouncing.phones_for_word(word.lower())
    except Exception:
        return []


def stresses_from_pronunciation(phone_string: str) -> List[float]:
    """
    CMU stress digits:
    1 = primary stress
    2 = secondary stress
    0 = unstressed

    Convert them to numeric weights.
    """
    if pronouncing is None:
        return []

    raw = pronouncing.stresses(phone_string)
    out: List[float] = []
    for ch in raw:
        if ch == "1":
            out.append(1.0)
        elif ch == "2":
            out.append(0.6)
        else:
            out.append(0.25)
    return out


def word_to_syllable_units(word: str) -> List[tuple[str, float]]:
    """
    Returns a list of (syllable_text_placeholder, stress_weight).

    We are not doing true grapheme-to-syllable splitting here.
    For alignment, approximate syllable placeholders are sufficient.
    """
    prons = get_pronunciations(word)
    if prons:
        stresses = stresses_from_pronunciation(prons[0])
        return [(f"{word}_{i}", s) for i, s in enumerate(stresses)]

    count = simple_fallback_syllable_count(word)
    if count == 1:
        return [(word, 1.0)]

    # Heuristic: last syllable stronger (often end word carries emphasis).
    return [(f"{word}_{i}", 0.25 if i < count - 1 else 1.0) for i in range(count)]


def extract_syllable_units(line: str, rhyme_zone_last_n_syllables: int = 2) -> List[SyllableUnit]:
    words = tokenize_words(line)
    expanded: List[SyllableUnit] = []

    for word_idx, word in enumerate(words):
        sylls = word_to_syllable_units(word)
        for i, (syll_text, stress) in enumerate(sylls):
            expanded.append(
                SyllableUnit(
                    text=syll_text,
                    word=word,
                    stress=float(stress),
                    syllable_index_in_word=i,
                    is_word_final=(i == len(sylls) - 1),
                    is_line_final=False,
                    is_rhyme_zone=False,
                )
            )

    if not expanded:
        return []

    # Mark final syllable of the line.
    expanded = list(expanded)
    last = expanded[-1]
    expanded[-1] = SyllableUnit(
        text=last.text,
        word=last.word,
        stress=last.stress,
        syllable_index_in_word=last.syllable_index_in_word,
        is_word_final=last.is_word_final,
        is_line_final=True,
        is_rhyme_zone=False,
    )

    # Mark rhyme zone (last N syllables).
    start_rhyme_zone = max(0, len(expanded) - rhyme_zone_last_n_syllables)
    for i in range(start_rhyme_zone, len(expanded)):
        unit = expanded[i]
        expanded[i] = SyllableUnit(
            text=unit.text,
            word=unit.word,
            stress=unit.stress,
            syllable_index_in_word=unit.syllable_index_in_word,
            is_word_final=unit.is_word_final,
            is_line_final=unit.is_line_final,
            is_rhyme_zone=True,
        )

    return expanded


def count_syllables_in_line(line: str) -> int:
    return len(extract_syllable_units(line))

