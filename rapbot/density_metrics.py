"""
density_metrics.py

Technical density metrics for verse analysis: internal rhyme density,
multisyllable span lengths, chain continuity, rhyme entropy, and overall technicality score.

V3 Rhyme-Eligible Spans: Only spans passing _is_rhyme_eligible_span contribute
to density, multisyllable avg, and entropy. Weighted density uses
log(1 + phoneme_tail_length) * type_weight * stress_weight.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from rapbot.internal_rhyme_detector import RhymeCluster, RHYME_ANCHOR_STOPWORDS, Span
from rapbot.phoneme_converter import PhonemeSequence, strip_stress, word_to_phonemes
from rapbot.rhyme_chain_detector import RhymeChain

# Nucleus match contributes to density when anchor's stressed nucleus (first phoneme of rhyme_nucleus)
# appears in any other rhyme anchor in the verse. Tuned for mean density in 0.20-0.45 range.
NUCLEUS_WEIGHT = 3.1
NUCLEUS_BONUS = math.log(2)  # per nucleus match

# Short content words (length < 3) that are valid rhyme anchors
CONTENT_WORD_WHITELIST = frozenset({"rap", "bar", "bars", "flow", "go", "yo", "so"})


def _is_rhyme_eligible_span(span: Span, rhyme_clusters: List[RhymeCluster]) -> bool:
    """
    A span is rhyme-eligible if it passes:
    - Not stopword (last word not in RHYME_ANCHOR_STOPWORDS)
    - Tail length >= 3 phonemes OR >= 2 syllables
    - Content word: length >= 3 chars or in whitelist
    """
    words = span.text.lower().split()
    if not words:
        return False
    last_word = words[-1].rstrip(".,!?;:'\"")
    if last_word in RHYME_ANCHOR_STOPWORDS:
        return False
    seq = word_to_phonemes(last_word)
    if not seq or not seq.phonemes:
        return False
    tail_phonemes = (
        len(seq.rhyme_tail(2)) if seq.syllable_boundaries else len(seq.rhyme_nucleus or seq.phonemes)
    )
    tail_syllables = min(2, len(seq.syllable_boundaries)) if seq.syllable_boundaries else 1
    if tail_phonemes < 3 and tail_syllables < 2:
        return False
    if len(last_word) < 3 and last_word not in CONTENT_WORD_WHITELIST:
        return False
    return True


def _span_weight(
    span: Span,
    rhyme_type: str,
) -> float:
    """
    weight = log(1 + phoneme_tail_length) * type_weight * stress_weight
    type_weight: perfect 1.0, slant 0.7, assonance 0.4, consonance 0.3
    stress_weight: 1.0 if stressed, else 0.4
    """
    words = span.text.split()
    if not words:
        return 0.0
    last_word = words[-1].rstrip(".,!?;:'\"")
    seq = word_to_phonemes(last_word.lower())
    if not seq:
        return 0.0
    tail_len = len(seq.rhyme_nucleus) if seq.rhyme_nucleus else len(seq.phonemes)
    type_weights = {"perfect": 1.0, "slant": 0.7, "assonance": 0.4, "consonance": 0.3}
    type_weight = type_weights.get(rhyme_type, 0.5)
    stress_weight = 1.0 if seq.stressed_vowels else 0.4
    return math.log(1.0 + tail_len) * type_weight * stress_weight


def _syllable_count(word: str) -> int:
    """Return syllable count for a word. Returns 1 if lookup fails."""
    seq = word_to_phonemes(word.lower())
    if not seq or not seq.syllable_boundaries:
        return 1
    return len(seq.syllable_boundaries)


def _tail_syllable_count(word: str) -> int:
    """
    Syllable count in the RHYME TAIL (rhyme_nucleus) of the word only.
    Uses last word's phoneme sequence; counts vowels in rhyme_nucleus.
    Returns 1 if lookup fails or nucleus empty.
    """
    from rapbot.phoneme_converter import ARPA_VOWELS, strip_stress

    seq = word_to_phonemes(word.lower())
    if not seq or not seq.rhyme_nucleus:
        return 1
    n = sum(
        1 for p in seq.rhyme_nucleus
        if strip_stress(p) in ARPA_VOWELS
    )
    return n if n > 0 else 1


def _stripped_stressed_nucleus(span: Span) -> Optional[str]:
    """
    Return the stripped stressed nucleus (first phoneme of rhyme_nucleus, e.g. UW1 -> UW)
    for the span's last word. Returns None if no rhyme nucleus.
    """
    words = span.text.split()
    if not words:
        return None
    last_word = words[-1].rstrip(".,!?;:'\"")
    seq = word_to_phonemes(last_word.lower())
    if not seq or not seq.rhyme_nucleus:
        return None
    first_phone = seq.rhyme_nucleus[0]
    return strip_stress(first_phone)


def _internal_rhyme_density_per_bar(
    lines: List[str],
    rhyme_clusters: List[RhymeCluster],
) -> List[float]:
    """
    Weighted internal rhyme density per bar. Only rhyme-eligible spans count.
    density = (weighted tail matches) + (nucleus_weight * nucleus matches)
    Tail weight = log(1 + phoneme_tail_length) * type_weight * stress_weight
    Nucleus bonus: add NUCLEUS_WEIGHT * log(2) when anchor's stressed nucleus
    (first phoneme of rhyme_nucleus, stripped) appears in any other anchor in the verse.
    Deduplicate by anchor (last_word_idx): take max(tail_weight) + nucleus_bonus per anchor.
    """
    # Precompute: anchor (line_idx, last_wi) -> (max_tail_weight, stripped_nucleus)
    anchor_data: Dict[tuple, tuple] = {}  # (line_idx, last_wi) -> (max_tail_w, nucleus)
    for cluster in rhyme_clusters:
        for span in cluster.spans:
            if not _is_rhyme_eligible_span(span, rhyme_clusters):
                continue
            last_wi = span.word_indices[-1] if span.word_indices else 0
            key = (span.line_idx, last_wi)
            w = _span_weight(span, cluster.rhyme_type)
            nucleus = _stripped_stressed_nucleus(span)
            old = anchor_data.get(key, (0.0, None))
            anchor_data[key] = (max(old[0], w), nucleus or old[1])

    # Nucleus counts: how many distinct anchors share each stripped nucleus
    nucleus_to_count: Dict[str, int] = {}
    for (_, _), (_, nuc) in anchor_data.items():
        if nuc:
            nucleus_to_count[nuc] = nucleus_to_count.get(nuc, 0) + 1

    densities: List[float] = []
    for line_idx, line in enumerate(lines):
        words = line.split()
        total_tokens = len(words)
        if total_tokens == 0:
            densities.append(0.0)
            continue
        anchor_weights: Dict[int, float] = {}
        for (li, last_wi), (tail_w, nucleus) in anchor_data.items():
            if li != line_idx:
                continue
            nucleus_bonus = (
                NUCLEUS_WEIGHT * NUCLEUS_BONUS
                if nucleus and nucleus_to_count.get(nucleus, 0) >= 2
                else 0.0
            )
            total = tail_w + nucleus_bonus
            anchor_weights[last_wi] = max(anchor_weights.get(last_wi, 0.0), total)
        weight_sum = sum(anchor_weights.values())
        densities.append(weight_sum / total_tokens)
    return densities


def _multisyllable_length_average(
    rhyme_clusters: List[RhymeCluster],
) -> float:
    """
    Mean syllable count of the RHYME TAIL of anchor tokens (last word of each span).
    Only rhyme-eligible spans are included.
    Tail = rhyme_nucleus (last stressed vowel to end), NOT full span.
    """
    tail_syllable_counts: List[float] = []
    for cluster in rhyme_clusters:
        for span in cluster.spans:
            if not _is_rhyme_eligible_span(span, rhyme_clusters):
                continue
            words = span.text.split()
            if not words:
                continue
            last_word = words[-1].rstrip(".,!?;:'\"")
            count = _tail_syllable_count(last_word)
            tail_syllable_counts.append(float(count))
    if not tail_syllable_counts:
        return 0.0
    return sum(tail_syllable_counts) / len(tail_syllable_counts)


def _chain_continuity_score(
    rhyme_chains: List[RhymeChain],
    total_lines: int,
) -> float:
    """
    For each chain: lines_in_chain / total_lines.
    Returns mean across chains. Zero if no chains or no lines.
    """
    if total_lines == 0 or not rhyme_chains:
        return 0.0
    scores = []
    for chain in rhyme_chains:
        lines_in_chain = len(set(p[0] for p in chain.positions))
        scores.append(lines_in_chain / total_lines)
    return sum(scores) / len(scores)


def _cluster_chain_eligible(cluster: RhymeCluster) -> bool:
    """
    Chain-eligibility: include cluster if avg tail >= 2 syl OR >= 4 phones.
    Exclude if avg tail < 2 syl AND < 4 phones.
    """
    if not cluster.spans:
        return False
    total_phones = 0.0
    total_syl = 0.0
    n = 0
    for span in cluster.spans:
        words = span.text.split()
        if not words:
            continue
        last_word = words[-1].rstrip(".,!?;:'\"")
        seq = word_to_phonemes(last_word.lower())
        if not seq:
            continue
        tail_phones = len(seq.rhyme_tail(2)) if seq.syllable_boundaries else len(seq.rhyme_nucleus or seq.phonemes)
        tail_syl = min(2, len(seq.syllable_boundaries)) if seq.syllable_boundaries else 1
        total_phones += tail_phones
        total_syl += tail_syl
        n += 1
    if n == 0:
        return False
    avg_phones = total_phones / n
    avg_syl = total_syl / n
    return avg_syl >= 2 or avg_phones >= 4


def _rhyme_entropy(
    rhyme_clusters: List[RhymeCluster],
    exclude_chain_ineligible: bool = True,
) -> float:
    """
    Shannon entropy over rhyme family size distribution.
    Optionally exclude clusters that fail chain-eligibility
    (avg tail < 2 syl AND < 4 phones).
    """
    if not rhyme_clusters:
        return 0.0
    clusters = rhyme_clusters
    if exclude_chain_ineligible:
        clusters = [c for c in rhyme_clusters if _cluster_chain_eligible(c)]
    if not clusters:
        return 0.0
    size_counts: Dict[int, int] = {}
    for cluster in clusters:
        n = len(cluster.spans)
        size_counts[n] = size_counts.get(n, 0) + 1
    total = len(clusters)
    entropy = 0.0
    for count in size_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# Default weights for overall_technicality_score
DEFAULT_WEIGHTS = {
    "internal_rhyme_density": 0.25,
    "multisyllable_length": 0.25,
    "chain_continuity": 0.25,
    "rhyme_entropy": 0.25,
}


def compute_density_metrics(
    lines: List[str],
    rhyme_clusters: List[RhymeCluster],
    rhyme_chains: List[RhymeChain],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute all density metrics and overall technicality score.

    Args:
        lines: Verse lines.
        rhyme_clusters: Detected rhyme clusters (families).
        rhyme_chains: Detected rhyme chains.
        weights: Optional dict of metric -> weight for overall_technicality_score.
                 Default: equal 0.25 each.

    Returns:
        Dict with:
        - internal_rhyme_density_per_bar: List[float] (one per line)
        - internal_rhyme_density_mean: float (mean across lines)
        - multisyllable_length_average: float
        - chain_continuity_score: float
        - rhyme_entropy: float
        - overall_technicality_score: float (weighted combo, 0–1 scale)
    """
    weights = weights or DEFAULT_WEIGHTS

    densities = _internal_rhyme_density_per_bar(lines, rhyme_clusters)
    density_mean = sum(densities) / len(densities) if densities else 0.0

    multisyll = _multisyllable_length_average(rhyme_clusters)
    # Normalize multisyllable to ~0–1: cap at 4 syllables for scaling
    multisyll_norm = min(1.0, multisyll / 4.0) if multisyll else 0.0

    chain_cont = _chain_continuity_score(rhyme_chains, len(lines))

    entropy = _rhyme_entropy(rhyme_clusters)
    # Normalize entropy: max for uniform over ~10 sizes is ~3.3, scale to 0–1
    entropy_norm = min(1.0, entropy / 3.5) if entropy else 0.0

    w = weights
    overall = (
        w.get("internal_rhyme_density", 0.25) * density_mean
        + w.get("multisyllable_length", 0.25) * multisyll_norm
        + w.get("chain_continuity", 0.25) * chain_cont
        + w.get("rhyme_entropy", 0.25) * entropy_norm
    )
    overall = min(1.0, max(0.0, overall))

    return {
        "internal_rhyme_density_per_bar": [round(d, 4) for d in densities],
        "internal_rhyme_density_mean": round(density_mean, 4),
        "multisyllable_length_average": round(multisyll, 4),
        "chain_continuity_score": round(chain_cont, 4),
        "rhyme_entropy": round(entropy, 4),
        "overall_technicality_score": round(overall, 4),
    }
