"""
rhyme_chain_detector.py

Detect rhyme chains (A→A→A) across multiple bars.
Track rhyme families across a verse and output chain patterns.

Supports cluster-based detection (preferred) via detect_rhyme_chains_from_clusters
and legacy end-word detection via detect_rhyme_chains.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from rapbot.phoneme_converter import ARPA_VOWELS, PhonemeSequence, strip_stress, word_to_phonemes
from rapbot.rhyme_similarity import compute_rhyme_similarity, extract_stressed_nucleus

if TYPE_CHECKING:
    from rapbot.internal_rhyme_detector import RhymeCluster


# Stopwords: weak rhyme anchors (common function words)
# Matches internal_rhyme_detector plus da, go, had, ya'll
RHYME_ANCHOR_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "at", "be", "but", "by", "da", "for", "from", "go",
    "had", "he", "her", "him", "his", "i", "id", "im", "i'm", "in", "is", "it", "it's",
    "me", "my", "no", "of", "on", "or", "our", "out", "she", "so", "that", "that's",
    "the", "them", "then", "there", "they", "they're", "this", "to", "was", "we", "we're",
    "when", "with", "ya", "ya'll", "yo", "you",
    "don't", "dont", "can't", "cant", "ain't", "aint",
    "'em", "em", "'cause", "cause", "'bout", "bout",
})


@dataclass
class RhymeChain:
    """A chain of rhymes across bars (e.g. A→A→A)."""
    chain_id: int
    pattern: str  # e.g. "AAA", "AABB"
    positions: List[Tuple[int, str]]  # (line_idx, end_word)
    family_id: int
    confidence: float = 0.0
    chain_continuity_score: float = 0.0  # proportion of consecutive lines participating
    chain_type: str = "tail"  # "tail" | "nucleus"
    representative_nucleus: Optional[str] = None  # e.g. "UW1" (stressed vowel phoneme)
    representative_tail: Optional[str] = None  # space-joined tail phonemes, e.g. "UW1 N Z"
    positions_with_phonemes: Optional[List[Dict[str, Any]]] = None  # [{line_idx, word, phonemes, tail, nucleus}, ...]


def _extract_end_word(line: str) -> Optional[str]:
    """Extract last lexical word from a bar/line."""
    import re
    WORD_RE = re.compile(r"[A-Za-z']+")
    words = WORD_RE.findall(line.lower())
    return words[-1] if words else None


def _assign_rhyme_families(
    lines: List[str],
    similarity_threshold: float = 0.65,
) -> Tuple[Dict[int, List[Tuple[int, str]]], int]:
    """
    Assign each line's end word to a rhyme family.
    Returns (family_id -> [(line_idx, word)], next_family_id).
    """
    end_words: List[Tuple[int, Optional[str]]] = []
    for i, line in enumerate(lines):
        w = _extract_end_word(line)
        end_words.append((i, w))

    word_to_family: Dict[str, int] = {}
    family_to_lines: Dict[int, List[Tuple[int, str]]] = {}
    next_fam = 0

    for line_idx, word in end_words:
        if not word:
            continue
        seq = word_to_phonemes(word)
        if not seq:
            continue

        assigned = False
        for existing_word, fam_id in list(word_to_family.items()):
            existing_seq = word_to_phonemes(existing_word)
            if not existing_seq:
                continue
            sim = compute_rhyme_similarity(seq, existing_seq, tail_syllables=2)
            if sim >= similarity_threshold:
                word_to_family[word] = fam_id
                family_to_lines.setdefault(fam_id, []).append((line_idx, word))
                assigned = True
                break

        if not assigned:
            word_to_family[word] = next_fam
            family_to_lines[next_fam] = [(line_idx, word)]
            next_fam += 1

    return family_to_lines, next_fam


def _extract_chains(
    family_to_lines: Dict[int, List[Tuple[int, str]]],
) -> List[RhymeChain]:
    """
    Extract chains from families. A chain is consecutive or near-consecutive
    lines in the same family (A→A→A pattern).
    """
    chains: List[RhymeChain] = []
    for fam_id, entries in family_to_lines.items():
        if len(entries) < 2:
            continue
        entries = sorted(entries, key=lambda x: x[0])
        positions = entries
        pattern = "A" * len(positions)
        chain = RhymeChain(
            chain_id=len(chains),
            pattern=pattern,
            positions=positions,
            family_id=fam_id,
            confidence=0.8,
        )
        chains.append(chain)
    return chains


def _primary_rhyme_word(span: "Span") -> str:
    """Extract primary rhyme word (last word) from a span."""
    words = span.text.split()
    return words[-1].lower() if words else ""


def _compute_canonical_family(cluster: "RhymeCluster") -> None:
    """
    Populate cluster's canonical family: representative_tail, top_anchors, example_spans.
    representative_tail = most frequent tail (or longest on tie).
    top_anchors = unigram tokens from span last words.
    example_spans = up to 5 example span texts.
    """
    tail_counts: Dict[Tuple[str, ...], int] = {}
    anchors: List[str] = []
    for span in cluster.spans:
        words = span.text.split()
        if not words:
            continue
        last_word = words[-1].lower().rstrip(".,!?;:'\"")
        anchors.append(last_word)
        seq = word_to_phonemes(last_word)
        if not seq or not seq.rhyme_nucleus:
            continue
        tail = seq.rhyme_nucleus
        tail_counts[tail] = tail_counts.get(tail, 0) + 1

    # representative_tail: most frequent, or longest on tie
    if tail_counts:
        max_count = max(tail_counts.values())
        candidates = [t for t, c in tail_counts.items() if c == max_count]
        rep_tail = max(candidates, key=len)
        cluster.representative_tail = rep_tail
    cluster.top_anchors = list(dict.fromkeys(anchors))[:20]
    cluster.example_spans = [s.text for s in cluster.spans[:5]]


def _is_stopword_anchor(word: str) -> bool:
    """Check if word is in rhyme anchor stopword list. Strips punctuation."""
    return word.rstrip(".,!?;:'\"").lower() in RHYME_ANCHOR_STOPWORDS


def _compute_chain_continuity(line_indices: List[int]) -> float:
    """
    Compute proportion of consecutive line pairs that participate.
    Chains spanning >= 3 consecutive lines get full score (1.0).
    """
    if len(line_indices) < 2:
        return 1.0
    sorted_lines = sorted(set(line_indices))
    consecutive_pairs = sum(
        1 for i in range(len(sorted_lines) - 1)
        if sorted_lines[i + 1] - sorted_lines[i] == 1
    )
    max_possible = len(sorted_lines) - 1
    return consecutive_pairs / max_possible if max_possible > 0 else 0.0


def _pick_best_anchor_for_line(
    spans_on_line: List["Span"],
    stopwords: Set[str],
) -> Optional[Tuple[int, str]]:
    """
    Pick best anchor (line_idx, end_word) from spans on same line.
    Includes internal anchors (any span on the line, not only line-final).
    Prefer non-stopword anchors; else prefer rightmost/last word of span.
    Returns None if the best anchor's word is a stopword (after stripping punctuation).
    """
    line_idx = spans_on_line[0].line_idx
    candidates: List[Tuple[str, int]] = []  # (word, word_idx_preference)
    for span in spans_on_line:
        word = _primary_rhyme_word(span)
        if not word:
            continue
        word_stripped = word.rstrip(".,!?;:'\"").lower()
        # Prefer spans whose last word is not stopword; use word_indices[-1] for tie-break
        pref = (0 if word_stripped not in stopwords else 1, -(span.word_indices[-1] if span.word_indices else 0))
        candidates.append((word, pref))

    if not candidates:
        return None
    # Sort: non-stopword first, then by rightmost word
    candidates.sort(key=lambda x: (x[1],))
    best_word = candidates[0][0]
    # Do not add line to chain if best anchor is a stopword
    if _is_stopword_anchor(best_word):
        return None
    return (line_idx, best_word)


def _filter_chains_by_gap(
    line_anchors: List[Tuple[int, str]],
    max_gap: int = 2,
) -> List[Tuple[int, str]]:
    """
    From line_anchors (sorted by line_idx), keep only consecutive runs where
    gap between members <= max_gap. Take the longest such run.
    """
    if len(line_anchors) < 2:
        return line_anchors
    best: List[Tuple[int, str]] = []
    current: List[Tuple[int, str]] = [line_anchors[0]]
    for i in range(1, len(line_anchors)):
        prev_li = current[-1][0]
        curr_li = line_anchors[i][0]
        if curr_li - prev_li <= max_gap:
            current.append(line_anchors[i])
        else:
            if len(current) > len(best):
                best = current
            current = [line_anchors[i]]
    if len(current) > len(best):
        best = current
    return best


def detect_nucleus_chains(
    clusters: List["RhymeCluster"],
    lines: List[str],
    min_chain_length: int = 3,
    max_line_gap: int = 2,
) -> List[RhymeChain]:
    """
    Detect nucleus chains: same stressed vowel (e.g. UW1, AE1) repeated across
    multiple lines. Groups by nucleus from cluster span last words.

    A NucleusChain = nucleus group with occurrences on >= min_chain_length distinct
    lines, with gaps <= max_line_gap between consecutive members.

    Returns list of RhymeChain with chain_type="nucleus" and representative_nucleus set.
    """
    from rapbot.internal_rhyme_detector import RhymeCluster, Span

    stopwords = set(RHYME_ANCHOR_STOPWORDS)
    nucleus_to_anchors: Dict[str, List[Tuple[int, str]]] = {}

    for cluster in clusters:
        for span in cluster.spans:
            last_word = _primary_rhyme_word(span)
            if not last_word:
                continue
            word_stripped = last_word.rstrip(".,!?;:'\"").lower()
            if word_stripped in stopwords:
                continue
            seq = word_to_phonemes(last_word)
            if not seq:
                continue
            nucleus = extract_stressed_nucleus(seq)
            if not nucleus:
                continue
            word_idx = span.word_indices[-1] if span.word_indices else 0
            nucleus_to_anchors.setdefault(nucleus, []).append((span.line_idx, last_word, word_idx))

    chains: List[RhymeChain] = []
    chain_id = 0
    for nucleus_phone, anchors in nucleus_to_anchors.items():
        # One anchor per line: keep rightmost word (max word_idx)
        line_to_entry: Dict[int, Tuple[int, str, int]] = {}
        for li, w, wi in anchors:
            if li not in line_to_entry or wi > line_to_entry[li][2]:
                line_to_entry[li] = (li, w, wi)
        line_anchors = [t[:2] for t in sorted(line_to_entry.values(), key=lambda x: x[0])]
        if len(line_anchors) < min_chain_length:
            continue

        line_anchors = _filter_chains_by_gap(line_anchors, max_gap=max_line_gap)
        if len(line_anchors) < min_chain_length:
            continue

        line_indices = [p[0] for p in line_anchors]
        continuity = _compute_chain_continuity(line_indices)
        pattern = "A" * len(line_anchors)

        # Chain Inspector: positions_with_phonemes for nucleus chains
        positions_with_phonemes: List[Dict[str, Any]] = []
        for line_idx, word in line_anchors:
            clean = word.rstrip(".,!?;:'\"").lower()
            seq = word_to_phonemes(clean)
            phonemes_str = " ".join(seq.phonemes) if seq else ""
            tail_str = " ".join(seq.rhyme_nucleus) if seq and seq.rhyme_nucleus else ""
            positions_with_phonemes.append({
                "line_idx": line_idx,
                "word": word,
                "phonemes": phonemes_str,
                "tail": tail_str,
                "nucleus": nucleus_phone,
            })

        chains.append(RhymeChain(
            chain_id=chain_id,
            pattern=pattern,
            positions=line_anchors,
            family_id=-1,  # Nucleus chains don't map to cluster family_id
            confidence=round(0.75 * (1.0 + 0.2 * continuity), 4),
            chain_continuity_score=round(continuity, 4),
            chain_type="nucleus",
            representative_nucleus=nucleus_phone,
            representative_tail=None,  # nucleus chains focus on vowel
            positions_with_phonemes=positions_with_phonemes,
        ))
        chain_id += 1

    return chains


def detect_rhyme_chains_from_clusters(
    clusters: List["RhymeCluster"],
    lines: List[str],
    min_chain_length: int = 3,
    max_line_gap: int = 2,
) -> List[RhymeChain]:
    """
    Build rhyme chains from detected rhyme clusters.
    Uses cluster spans that span >= min_chain_length different lines.
    Includes internal anchors (any span on a line, not only line-final).
    Allows gaps <= max_line_gap lines between chain members.

    Logic:
    - For each cluster, compute canonical family: representative_tail, top_anchors, example_spans.
    - Chain eligibility: canonical tail >= 2 syllables OR >= 4 phonemes (not per-span).
    - Group ALL spans by line_idx (internal anchors included), pick best anchor per line.
    - Filter line_anchors so gaps between consecutive lines <= max_line_gap.
    - Only output chains with >= min_chain_length lines.

    Args:
        clusters: Rhyme clusters from detect_internal_rhymes
        lines: Verse lines (for context)
        min_chain_length: Minimum number of lines a chain must span
        max_line_gap: Max allowed lines between consecutive chain members (default 2)

    Returns:
        List of RhymeChain
    """
    from rapbot.internal_rhyme_detector import RhymeCluster, Span

    stopwords = set(RHYME_ANCHOR_STOPWORDS)
    chains: List[RhymeChain] = []
    chain_id = 0

    for cluster in clusters:
        # Compute canonical family
        _compute_canonical_family(cluster)

        # Tail-length filter: use canonical representative_tail (not per-span avg)
        rep_tail = cluster.representative_tail
        if not rep_tail:
            continue
        tail_phonemes = len(rep_tail)
        tail_syllables = sum(
            1 for p in rep_tail if strip_stress(p) in ARPA_VOWELS
        ) or 1
        if tail_syllables < 2 and tail_phonemes < 4:
            continue

        # Include ALL spans in cluster (internal anchors, not only line-final)
        line_to_spans: Dict[int, List[Span]] = {}
        for span in cluster.spans:
            line_to_spans.setdefault(span.line_idx, []).append(span)

        if len(line_to_spans) < min_chain_length:
            continue

        # Build best anchor per line (internal or line-final)
        line_anchors: List[Tuple[int, str]] = []
        for line_idx in sorted(line_to_spans.keys()):
            anchor = _pick_best_anchor_for_line(line_to_spans[line_idx], stopwords)
            if anchor is None:
                continue
            line_anchors.append(anchor)

        if len(line_anchors) < min_chain_length:
            continue

        # Order by line index and filter by gap: allow gaps <= max_line_gap
        line_anchors.sort(key=lambda x: x[0])
        line_anchors = _filter_chains_by_gap(line_anchors, max_gap=max_line_gap)

        if len(line_anchors) < min_chain_length:
            continue

        positions = line_anchors
        pattern = "A" * len(positions)

        # chain_continuity_score: proportion of consecutive lines
        line_indices = [p[0] for p in positions]
        continuity = _compute_chain_continuity(line_indices)
        # Boost confidence when chain spans >= 3 lines with good continuity
        base_conf = cluster.confidence
        conf = min(1.0, base_conf * (1.0 + 0.2 * continuity) if len(positions) >= 3 else base_conf)

        # Chain Inspector: chain_type, representative, positions_with_phonemes
        rep_tail_tuple = cluster.representative_tail or ()
        rep_nucleus = None
        rep_tail_str = None
        if rep_tail_tuple:
            rep_tail_str = " ".join(rep_tail_tuple)
            # Nucleus = first phoneme of tail (stressed vowel)
            for p in rep_tail_tuple:
                if strip_stress(p) in ARPA_VOWELS:
                    rep_nucleus = p
                    break
        chain_type = "tail"  # chains from clusters use full tail similarity

        positions_with_phonemes: List[Dict[str, Any]] = []
        for line_idx, word in positions:
            clean = word.rstrip(".,!?;:'\"").lower()
            seq = word_to_phonemes(clean)
            phonemes_str = " ".join(seq.phonemes) if seq else ""
            tail_str = " ".join(seq.rhyme_nucleus) if seq and seq.rhyme_nucleus else ""
            nucleus_phone = None
            if seq and seq.rhyme_nucleus:
                for p in seq.rhyme_nucleus:
                    if strip_stress(p) in ARPA_VOWELS:
                        nucleus_phone = p
                        break
            positions_with_phonemes.append({
                "line_idx": line_idx,
                "word": word,
                "phonemes": phonemes_str,
                "tail": tail_str,
                "nucleus": nucleus_phone or "",
            })

        chains.append(RhymeChain(
            chain_id=chain_id,
            pattern=pattern,
            positions=positions,
            family_id=cluster.family_id,
            confidence=round(conf, 4),
            chain_continuity_score=round(continuity, 4),
            chain_type=chain_type,
            representative_nucleus=rep_nucleus,
            representative_tail=rep_tail_str,
            positions_with_phonemes=positions_with_phonemes,
        ))
        chain_id += 1

    return chains


def detect_rhyme_chains(
    lines: List[str],
    similarity_threshold: float = 0.65,
) -> List[RhymeChain]:
    """
    Detect rhyme chains across verse bars (legacy end-word method).
    Tracks rhyme families and outputs chain patterns (A→A→A, etc.).

    .. deprecated::
        Prefer detect_rhyme_chains_from_clusters when clusters from
        detect_internal_rhymes are available.

    Args:
        lines: Verse lines (bars)
        similarity_threshold: Min similarity to group into same family

    Returns:
        List of RhymeChain
    """
    warnings.warn(
        "detect_rhyme_chains is deprecated; prefer detect_rhyme_chains_from_clusters "
        "when clusters from detect_internal_rhymes are available.",
        DeprecationWarning,
        stacklevel=2,
    )
    family_to_lines, _ = _assign_rhyme_families(lines, similarity_threshold)
    return _extract_chains(family_to_lines)


def get_rhyme_scheme(lines: List[str], similarity_threshold: float = 0.65) -> str:
    """
    Return compact rhyme scheme string (e.g. "AABB", "ABAB").
    Letters assigned by order of first appearance in the verse.
    """
    family_to_lines, _ = _assign_rhyme_families(lines, similarity_threshold)
    # Sort families by minimum line index (first occurrence)
    sorted_fams = sorted(
        family_to_lines.items(),
        key=lambda x: min(idx for idx, _ in x[1]),
    )
    line_to_letter: Dict[int, str] = {}
    for idx, (fam_id, entries) in enumerate(sorted_fams):
        letter = chr(ord("A") + idx)
        for line_idx, _ in entries:
            line_to_letter[line_idx] = letter

    scheme = []
    for i in range(len(lines)):
        scheme.append(line_to_letter.get(i, "X"))
    return "".join(scheme)
