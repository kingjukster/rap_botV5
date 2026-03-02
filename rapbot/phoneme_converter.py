"""
phoneme_converter.py

Phonetics-first rhyme engine: full phoneme sequence extraction using pronouncing/CMUdict.
Provides syllable boundary detection, G2P fallback for OOV words, and rap contraction handling.
Integrates with extract_last_syllable logic from update_rhyme_groups.py for rhyme nucleus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import pronouncing

# ARPA vowel set (CMU Pronouncing Dictionary)
ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

# Rap/hip-hop contractions: informal form -> standard dictionary form
RAP_CONTRACTIONS = {
    "'em": "them",
    "em": "them",
    "droppin'": "dropping",
    "droppin": "dropping",
    "bustin'": "busting",
    "bustin": "busting",
    "flowin'": "flowing",
    "flowin": "flowing",
    "goin'": "going",
    "goin": "going",
    "comin'": "coming",
    "comin": "coming",
    "runnin'": "running",
    "runnin": "running",
    "gettin'": "getting",
    "gettin": "getting",
    "makin'": "making",
    "makin": "making",
    "takin'": "taking",
    "takin": "taking",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "musta": "must have",
    "hafta": "have to",
    "hasta": "has to",
    "oughta": "ought to",
    "used-ta": "used to",
    "outta": "out of",
    "lotsa": "lots of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "gotcha": "got you",
    "prolly": "probably",
    "supposed": "supposed",
    "supposin'": "supposing",
    "rollin'": "rolling",
    "rollin": "rolling",
    "holdin'": "holding",
    "holdin": "holding",
    "tellin'": "telling",
    "tellin": "telling",
    "callin'": "calling",
    "callin": "calling",
    "fallin'": "falling",
    "fallin": "falling",
    "ballin'": "balling",
    "ballin": "balling",
    "playin'": "playing",
    "playin": "playing",
    "sayin'": "saying",
    "sayin": "saying",
    "doin'": "doing",
    "doin": "doing",
    "bein'": "being",
    "bein": "being",
    "seein'": "seeing",
    "seein": "seeing",
    "tryin'": "trying",
    "tryin": "trying",
    "lyin'": "lying",
    "lyin": "lying",
    "cryin'": "crying",
    "cryin": "crying",
    "livin'": "living",
    "livin": "living",
    "givin'": "giving",
    "givin": "giving",
    "drivin'": "driving",
    "drivin": "driving",
    "takin'": "taking",
    "takin": "taking",
    "breakin'": "breaking",
    "breakin": "breaking",
    "shakin'": "shaking",
    "shakin": "shaking",
}


def strip_stress(phone: str) -> str:
    """Remove stress digit (0/1/2) from ARPA phoneme."""
    return re.sub(r"\d", "", phone.upper())


def _expand_rap_contraction(word: str) -> str:
    """Map rap contraction to standard dictionary form for lookup."""
    w = word.lower()
    return RAP_CONTRACTIONS.get(w, word)


def _g2p_fallback(word: str) -> Optional[str]:
    """Attempt G2P conversion for OOV words. Returns ARPA string or None."""
    try:
        import g2p_en
        g2p = g2p_en.G2p()
        phones = g2p(word)
        if phones:
            return " ".join(phones)
    except ImportError:
        pass
    except Exception:
        pass
    return None


@dataclass
class PhonemeSequence:
    """Full phoneme sequence with syllable and rhyme analysis."""

    phonemes: Tuple[str, ...]
    stressed_vowels: List[int]  # Indices into phonemes where stressed vowels occur
    syllable_boundaries: List[int]  # Indices marking syllable starts (0 = first phoneme)
    rhyme_nucleus: Tuple[str, ...]  # Last stressed vowel to end (for rhyme matching)
    has_stressed_rhyme_vowel: bool = True  # True if nucleus starts at stressed vowel, False if fell back to unstressed
    raw_word: str = ""
    source: str = "pronouncing"  # "pronouncing" | "g2p" | "contraction"

    def __post_init__(self):
        if isinstance(self.phonemes, list):
            object.__setattr__(self, "phonemes", tuple(self.phonemes))
        if isinstance(self.rhyme_nucleus, list):
            object.__setattr__(self, "rhyme_nucleus", tuple(self.rhyme_nucleus))

    @property
    def last_stressed_vowel_idx(self) -> Optional[int]:
        """Index of last stressed vowel in phonemes, or None."""
        if not self.stressed_vowels:
            return None
        return self.stressed_vowels[-1]

    def rhyme_tail(self, num_syllables: int = 1) -> Tuple[str, ...]:
        """
        Return tail phonemes for multi-syllable rhyme comparison.
        Takes last N syllable boundaries and returns phonemes from that point.
        """
        if num_syllables <= 0 or not self.syllable_boundaries:
            return self.rhyme_nucleus
        n = min(num_syllables, len(self.syllable_boundaries))
        start_idx = self.syllable_boundaries[-n]
        return tuple(self.phonemes[start_idx:])


def tail_phones(seq: PhonemeSequence) -> int:
    """Number of phonemes in the rhyme tail (rhyme_nucleus)."""
    return len(seq.rhyme_nucleus)


def tail_syllables(seq: PhonemeSequence) -> int:
    """Number of vowel nuclei (syllables) in the rhyme tail."""
    return sum(1 for p in seq.rhyme_nucleus if strip_stress(p) in ARPA_VOWELS)


def nucleus_vowel(seq: PhonemeSequence) -> Optional[str]:
    """First vowel phone in the rhyme nucleus, or None."""
    for p in seq.rhyme_nucleus:
        if strip_stress(p) in ARPA_VOWELS:
            return p
    return None


def _get_syllable_boundaries(tokens: List[str]) -> List[int]:
    """
    Detect syllable boundaries from ARPA phoneme sequence.
    Syllables are marked by onset of each vowel (nucleus). Stress digits indicate
    primary (1), secondary (2), or unstressed (0) syllables.
    """
    boundaries = [0]
    for i, tok in enumerate(tokens):
        base = strip_stress(tok)
        if base in ARPA_VOWELS and i > 0:
            boundaries.append(i)
    return boundaries


def _get_stressed_vowel_indices(tokens: List[str]) -> List[int]:
    """Indices where stressed vowels occur (stress digit 1 or 2)."""
    indices = []
    for i, tok in enumerate(tokens):
        base = strip_stress(tok)
        if base in ARPA_VOWELS:
            if len(tok) > 0 and tok[-1].isdigit():
                stress = int(tok[-1])
                if stress in (1, 2):
                    indices.append(i)
            else:
                # No stress digit (e.g. G2P) - treat as stressed for indexing
                indices.append(i)
    return indices


def _extract_rhyme_nucleus(tokens: List[str], stressed_indices: List[int]) -> Tuple[Tuple[str, ...], bool]:
    """
    Extract rhyme nucleus: from last stressed vowel (stress 1 or 2) to end.
    If no stressed vowel exists, fall back to last vowel to end.
    Returns (nucleus, has_stressed_rhyme_vowel).
    """
    if stressed_indices:
        start_idx = stressed_indices[-1]
        nucleus = tuple(t.strip() for t in tokens[start_idx:])
        return nucleus, True
    for idx in range(len(tokens) - 1, -1, -1):
        token = tokens[idx]
        base = strip_stress(token)
        if base in ARPA_VOWELS:
            nucleus = tuple(t.strip() for t in tokens[idx:])
            return nucleus, False
    return (), False


def _merge_phoneme_sequences(seqs: List[PhonemeSequence], raw_word: str) -> PhonemeSequence:
    """
    Merge PhonemeSequences for hyphenated compounds (e.g. jack-offs -> jack + offs).
    Rhyme nucleus = last part's nucleus (compound rhymes on the final element).
    """
    if not seqs:
        raise ValueError("merge requires at least one sequence")
    if len(seqs) == 1:
        return PhonemeSequence(
            phonemes=seqs[0].phonemes,
            stressed_vowels=list(seqs[0].stressed_vowels),
            syllable_boundaries=list(seqs[0].syllable_boundaries),
            rhyme_nucleus=seqs[0].rhyme_nucleus,
            has_stressed_rhyme_vowel=seqs[0].has_stressed_rhyme_vowel,
            raw_word=raw_word,
            source=seqs[0].source,
        )
    all_phonemes: List[str] = []
    all_stressed: List[int] = []
    all_boundaries: List[int] = [0]
    offset = 0
    for s in seqs:
        all_phonemes.extend(s.phonemes)
        all_stressed.extend(offset + i for i in s.stressed_vowels)
        for b in s.syllable_boundaries[1:]:  # skip leading 0
            all_boundaries.append(offset + b)
        offset += len(s.phonemes)
    # Rhyme nucleus = last part's nucleus (e.g. Daniel-san rhymes on "san")
    last = seqs[-1]
    return PhonemeSequence(
        phonemes=tuple(all_phonemes),
        stressed_vowels=all_stressed,
        syllable_boundaries=sorted(set(all_boundaries)),
        rhyme_nucleus=last.rhyme_nucleus,
        has_stressed_rhyme_vowel=last.has_stressed_rhyme_vowel,
        raw_word=raw_word,
        source="g2p" if any(s.source == "g2p" for s in seqs) else "pronouncing",
    )


@lru_cache(maxsize=10000)
def word_to_phonemes(word: str) -> Optional[PhonemeSequence]:
    """
    Convert word to full PhonemeSequence using pronouncing/CMUdict.
    Handles rap contractions, hyphenated compounds (jack-offs, Daniel-san), and G2P fallback for OOV.
    """
    w = str(word).strip()
    if not w:
        return None

    # Contraction expansion first (e.g. "used-ta" -> "used to")
    expanded = _expand_rap_contraction(w)
    parts = expanded.split()
    lookup_word = parts[0].lower() if parts else w.lower()

    # OOV hyphen handling: split hyphenated compounds before phoneme lookup
    if "-" in lookup_word:
        hyphen_parts = [p.strip() for p in lookup_word.split("-") if p.strip()]
        if len(hyphen_parts) > 1:
            seqs: List[PhonemeSequence] = []
            for part in hyphen_parts:
                s = word_to_phonemes(part)
                if not s:
                    return None
                seqs.append(s)
            return _merge_phoneme_sequences(seqs, raw_word=w)

    # Pronouncing / CMUdict
    phones_list = pronouncing.phones_for_word(lookup_word)

    if not phones_list:
        # G2P fallback
        arpa_str = _g2p_fallback(lookup_word)
        if arpa_str:
            tokens = arpa_str.strip().split()
            boundaries = _get_syllable_boundaries(tokens)
            stressed = _get_stressed_vowel_indices(tokens)
            nucleus, has_stressed = _extract_rhyme_nucleus(tokens, stressed)
            return PhonemeSequence(
                phonemes=tuple(tokens),
                stressed_vowels=stressed,
                syllable_boundaries=boundaries,
                rhyme_nucleus=nucleus,
                has_stressed_rhyme_vowel=has_stressed,
                raw_word=w,
                source="g2p",
            )
        return None

    # Use first pronunciation
    ph = phones_list[0]
    tokens = ph.split()
    boundaries = _get_syllable_boundaries(tokens)
    stressed = _get_stressed_vowel_indices(tokens)
    nucleus, has_stressed = _extract_rhyme_nucleus(tokens, stressed)

    source = "contraction" if expanded != w else "pronouncing"
    return PhonemeSequence(
        phonemes=tuple(tokens),
        stressed_vowels=stressed,
        syllable_boundaries=boundaries,
        rhyme_nucleus=nucleus,
        has_stressed_rhyme_vowel=has_stressed,
        raw_word=w,
        source=source,
    )


def extract_last_syllable(word: str) -> Optional[Tuple[str, int, Tuple[str, ...], str, Tuple[str, ...]]]:
    """
    Compatible with update_rhyme_groups PhoneticFeature: returns
    (vowel, stress, coda, vowel_group, raw) or None.
    Kept for integration with existing extract_last_syllable logic.
    """
    seq = word_to_phonemes(word.lower())
    if not seq or not seq.phonemes:
        return None

    tokens = list(seq.phonemes)
    vowel = None
    stress = 0
    coda: List[str] = []

    for idx in range(len(tokens) - 1, -1, -1):
        token = tokens[idx]
        base = strip_stress(token)
        if base in ARPA_VOWELS and vowel is None:
            vowel = base
            if token and token[-1].isdigit():
                stress = int(token[-1])
            coda = [strip_stress(t) for t in tokens[idx + 1:] if strip_stress(t) not in ARPA_VOWELS]
            break

    if not vowel:
        return None

    # Vowel group (from update_rhyme_groups VOWEL_GROUPS)
    VOWEL_GROUPS = [
        {"AA", "AO", "AH"},
        {"AE", "EH", "EY"},
        {"IH", "IY"},
        {"OW", "UW", "UH"},
        {"ER"},
        {"AW", "AY", "OY"},
    ]
    vowel_group = vowel
    for idx, group in enumerate(VOWEL_GROUPS):
        if vowel in group:
            vowel_group = f"VG{idx}"
            break

    return (vowel, stress, tuple(coda), vowel_group, tuple(tokens))


def phrase_to_phonemes(text: str) -> List[PhonemeSequence]:
    """
    Convert a phrase (multiple words) to list of PhonemeSequences.
    Word tokenization via simple regex for letters and apostrophes.
    """
    WORD_RE = re.compile(r"[A-Za-z']+")
    words = WORD_RE.findall(text)
    sequences: List[PhonemeSequence] = []
    for w in words:
        seq = word_to_phonemes(w)
        if seq:
            sequences.append(seq)
    return sequences
