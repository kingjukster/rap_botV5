"""
evo_rhyme phonetics: rhyme tail extraction, pronunciations, syllable counting.

Extracts rhyme tail from last stressed vowel onward using pronouncing.
Loads custom_pronunciation.json from data/evo_rhyme/ for OOV words (tryna, fiya, opp).
Reuses logic from scripts.tools.update_rhyme_groups (PhoneticFeature, extract_last_syllable, phonetic_similarity).
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pronouncing
except ImportError:
    pronouncing = None  # type: ignore

WORD_RE = re.compile(r"[A-Za-z']+")
ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

_CUSTOM_PRONUNCIATIONS: Optional[dict[str, List[str]]] = None
_CUSTOM_PRONUNCIATIONS_LOADED: bool = False  # True when real JSON loaded (not git-lfs pointer)


def _get_custom_pronunciations_path() -> Path:
    """Path to custom_pronunciation.json in data/evo_rhyme/."""
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "evo_rhyme" / "custom_pronunciation.json"


def _load_custom_pronunciations() -> dict[str, List[str]]:
    """Load custom pronunciations for OOV words (tryna, fiya, opp, etc.)."""
    global _CUSTOM_PRONUNCIATIONS, _CUSTOM_PRONUNCIATIONS_LOADED
    if _CUSTOM_PRONUNCIATIONS is not None:
        return _CUSTOM_PRONUNCIATIONS
    path = _get_custom_pronunciations_path()
    if not path.exists():
        _CUSTOM_PRONUNCIATIONS = {}
        return _CUSTOM_PRONUNCIATIONS
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Git-LFS pointer: file not pulled yet; treat as empty
    if raw.strip().startswith("version https://git-lfs.github.com/spec/v1"):
        _CUSTOM_PRONUNCIATIONS = {}
        return _CUSTOM_PRONUNCIATIONS
    data = json.loads(raw)
    _CUSTOM_PRONUNCIATIONS_LOADED = True
    result: dict[str, List[str]] = {}
    for k, v in data.items():
        key = str(k).lower()
        if isinstance(v, list):
            parts = [str(p) for p in v]
            # If first element has no space, treat as list of individual phones - join to space-separated string
            if parts and " " not in parts[0]:
                result[key] = [" ".join(parts)]
            else:
                result[key] = parts
        else:
            result[key] = [str(v)]
    _CUSTOM_PRONUNCIATIONS = result
    return _CUSTOM_PRONUNCIATIONS


def has_custom_pronunciations() -> bool:
    """True if custom_pronunciation.json was loaded (not git-lfs pointer or missing)."""
    _load_custom_pronunciations()
    return _CUSTOM_PRONUNCIATIONS_LOADED


def strip_stress(phone: str) -> str:
    """Remove stress digit from ARPA phone."""
    return re.sub(r"\d", "", phone.upper())


@dataclass
class PhoneticFeature:
    """Last-syllable phonetic features for rhyme matching. From scripts.tools.update_rhyme_groups."""
    vowel: str
    stress: int
    coda: Tuple[str, ...]
    vowel_group: str
    raw: Tuple[str, ...]


def get_pronunciations(word: str) -> List[str]:
    """
    Get pronunciations for a word. Uses custom_pronunciation.json for OOV,
    otherwise pronouncing.phones_for_word.
    Returns list of phone strings (e.g. ["T R AY1 N AH0"]).
    """
    word = str(word).lower()
    custom = _load_custom_pronunciations()
    if word in custom:
        return custom[word]
    if pronouncing is None:
        return []
    phones = pronouncing.phones_for_word(word)
    return list(phones) if phones else []


def phones_for_word(word: str) -> List[str]:
    """Alias for get_pronunciations. Returns empty list if unknown."""
    return get_pronunciations(word)


def extract_rhyme_tail(word: str) -> Optional[str]:
    """
    Extract rhyme tail from last stressed vowel onward using pronouncing.
    Returns the rhyming part (e.g. "AY1 N AH0" for "tryna") or None if unknown.
    """
    phones_list = get_pronunciations(word)
    if not phones_list or pronouncing is None:
        return None
    return pronouncing.rhyming_part(phones_list[0]) or None


def multisyllable_overlap(tail1: Optional[str], tail2: Optional[str]) -> int:
    """
    Count matching phonemes from the end of two rhyme tails.

    Takes two rhyme tails from extract_rhyme_tail (e.g. "IY1 N", "EY1 SH AH0 N"),
    splits each by space to get phoneme lists, and counts matching phonemes
    from the end (reversed comparison).

    Returns:
        Overlap count, or 0 if either tail is None or empty.
    """
    if tail1 is None or tail2 is None:
        return 0
    tail1 = tail1.strip()
    tail2 = tail2.strip()
    if not tail1 or not tail2:
        return 0
    p1 = tail1.split()
    p2 = tail2.split()
    count = 0
    for i in range(min(len(p1), len(p2))):
        if p1[-(i + 1)] == p2[-(i + 1)]:
            count += 1
        else:
            break
    return count


def count_syllables(word: str) -> int:
    """Count syllables in a word. Returns 0 if pronunciation unknown."""
    phones_list = get_pronunciations(word)
    if not phones_list or pronouncing is None:
        return 0
    return pronouncing.syllable_count(phones_list[0])


def syllable_count_word(word: str) -> int:
    """Count syllables in a word. Falls back to vowel heuristic if unknown."""
    n = count_syllables(word)
    if n > 0:
        return n
    groups = re.findall(r"[aeiouy]+", str(word).lower())
    return max(1, len(groups))


def extract_last_syllable(word: str) -> Optional[PhoneticFeature]:
    """
    Extract last-syllable phonetic features for rhyme matching.
    Reused from scripts.tools.update_rhyme_groups.
    Returns None if word has no valid pronunciation.
    """
    word = str(word).lower()
    phones_list = get_pronunciations(word)
    if not phones_list:
        return None

    feature: Optional[PhoneticFeature] = None
    for ph in phones_list:
        tokens = ph.split()
        vowel = None
        stress = 0
        coda: List[str] = []
        for idx in range(len(tokens) - 1, -1, -1):
            token = tokens[idx]
            base = strip_stress(token)
            if base in ARPA_VOWELS and vowel is None:
                vowel = base
                if token[-1].isdigit():
                    stress = int(token[-1])
                coda = [
                    strip_stress(t) for t in tokens[idx + 1:]
                    if strip_stress(t) not in ARPA_VOWELS
                ]
                break
        if vowel:
            feature = PhoneticFeature(
                vowel=vowel,
                stress=stress,
                coda=tuple(coda) if coda else tuple(),
                vowel_group=_vowel_group_key(vowel),
                raw=tuple(tokens),
            )
            break
    return feature


def stress_pattern_from_phones(phones: str) -> Tuple[int, ...]:
    """Extract stress pattern (0=unstressed, 1=primary, 2=secondary) from ARPA phones."""
    tokens = phones.split()
    pattern: List[int] = []
    for t in tokens:
        base = strip_stress(t)
        if base in ARPA_VOWELS:
            stress = int(t[-1]) if t[-1].isdigit() else 0
            pattern.append(stress)
    return tuple(pattern)


def tokenize_line(line: str) -> List[str]:
    """Extract word tokens from a line."""
    return WORD_RE.findall(line.lower())


def extract_stressed_vowels_from_phones(phones: str) -> List[str]:
    """
    Extract stressed vowel phonemes from CMU/ARPA phones.
    Vowel phonemes end in 0 (unstressed), 1 (primary), or 2 (secondary).
    Returns list of full vowel tokens, e.g. ["AE1", "AH0"] for "DH AE1 K AH0".
    """
    tokens = phones.split()
    result: List[str] = []
    for t in tokens:
        base = strip_stress(t)
        if base in ARPA_VOWELS and len(t) >= 2 and t[-1] in "012":
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# vowel_to_words index: stressed vowel phoneme -> list of words containing it
# ---------------------------------------------------------------------------

_VOWEL_TO_WORDS: Optional[Dict[str, List[str]]] = None


def _get_rhymes_csv_path() -> Path:
    """Path to rhymes_grouped.csv (same source as tail_to_words)."""
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "rhymes_grouped.csv"


def _build_vowel_to_words() -> Dict[str, List[str]]:
    """
    Build vowel_to_words index from rhymes_grouped.csv.
    Maps stressed vowel phoneme (e.g. "AA1", "EH0") -> list of words that contain it.
    Uses same word set as tail_to_words for valid dictionary words.
    """
    global _VOWEL_TO_WORDS
    if _VOWEL_TO_WORDS is not None:
        return _VOWEL_TO_WORDS

    path = _get_rhymes_csv_path()
    vowel_to_words: Dict[str, List[str]] = defaultdict(list)
    seen: set[tuple] = set()  # (vowel, word) to avoid duplicates

    if not path.exists():
        _VOWEL_TO_WORDS = dict(vowel_to_words)
        return _VOWEL_TO_WORDS

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip().lower()
            if not word:
                continue

            for phones in get_pronunciations(word):
                for vowel in extract_stressed_vowels_from_phones(phones):
                    key = (vowel, word)
                    if key not in seen:
                        seen.add(key)
                        vowel_to_words[vowel].append(word)

    _VOWEL_TO_WORDS = {k: list(v) for k, v in vowel_to_words.items()}
    return _VOWEL_TO_WORDS


def get_vowel_to_words() -> Dict[str, List[str]]:
    """Return the vowel_to_words index (built on first call)."""
    return _build_vowel_to_words()


def syllable_count_line(line: str) -> int:
    """Count syllables in a line."""
    words = tokenize_line(line)
    return sum(syllable_count_word(w) for w in words)


# ---------------------------------------------------------------------------
# Rhyme similarity (for fitness scoring)
# ---------------------------------------------------------------------------

VOWEL_GROUPS = [
    {"AA", "AO", "AH"},
    {"AE", "EH", "EY"},
    {"IH", "IY"},
    {"OW", "UW", "UH"},
    {"ER"},
    {"AW", "AY", "OY"},
]

CONSONANT_GROUPS = [
    {"B", "P"},
    {"D", "T"},
    {"G", "K"},
    {"S", "Z", "SH", "ZH"},
    {"F", "V"},
    {"CH", "JH"},
    {"M", "N", "NG"},
    {"L", "R"},
]


def _vowel_group_key(vowel: str) -> str:
    upper = str(vowel).upper()
    for idx, group in enumerate(VOWEL_GROUPS):
        if upper in group:
            return f"VG{idx}"
    return upper


vowel_group_key = _vowel_group_key  # Public alias for scoring module


def _consonant_group_key(phone: str) -> str:
    upper = strip_stress(phone)
    for idx, group in enumerate(CONSONANT_GROUPS):
        if upper in group:
            return f"CG{idx}"
    return upper


def _vowel_similarity(v1: Optional[str], v2: Optional[str]) -> float:
    if not v1 or not v2:
        return 0.0
    if v1 == v2:
        return 1.0
    if _vowel_group_key(v1) == _vowel_group_key(v2):
        return 0.75
    return 0.25


def _stress_similarity(s1: int, s2: int) -> float:
    if s1 == s2:
        return 1.0 if s1 > 0 else 0.7
    if s1 > 0 and s2 > 0:
        return 0.8
    if s1 == 0 and s2 == 0:
        return 0.6
    return 0.35


def _consonant_similarity(c1: Tuple[str, ...], c2: Tuple[str, ...]) -> float:
    if not c1 and not c2:
        return 0.5
    if not c1 or not c2:
        return 0.3
    if c1 == c2:
        return 1.0
    h1, h2 = c1[0], c2[0]
    if h1 == h2:
        return 0.8
    if _consonant_group_key(h1) == _consonant_group_key(h2):
        return 0.6
    return 0.3


def phonetic_similarity(
    f1: Optional[PhoneticFeature], f2: Optional[PhoneticFeature]
) -> float:
    """Rhyme similarity [0,1] between two last-syllable features."""
    if not f1 or not f2:
        return 0.0
    sigma_v = _vowel_similarity(f1.vowel, f2.vowel)
    if sigma_v <= 0:
        return 0.0
    sigma_s = _stress_similarity(f1.stress, f2.stress)
    sigma_c = _consonant_similarity(f1.coda, f2.coda)
    base = 0.5 * sigma_v + 0.2 * sigma_s + 0.3 * sigma_c
    return float(base * sigma_v)
