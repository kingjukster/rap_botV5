"""
fusion_engine.py

Late fusion for the Two-Track Verse Analysis System.
Combines outputs from phonetics engine (rhyme clusters, chains) and
semantics engine (discourse, metaphors, punchlines, cohesion).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rapbot.internal_rhyme_detector import RHYME_ANCHOR_STOPWORDS
from rapbot.phoneme_converter import ARPA_VOWELS, strip_stress, word_to_phonemes
from rapbot.verse_annotation import (
    Token,
    VerseAnnotation,
    RhymeCluster,
    RhymeChain,
    MetaphorFrame,
    Punchline,
    DiscourseUnit,
)
from rapbot.density_metrics import compute_density_metrics


WORD_RE = re.compile(r"[A-Za-z']+")


def _enrich_tokens_with_anchor_metadata(
    tokens: List[Token],
    rhyme_clusters: List[RhymeCluster],
) -> None:
    """
    Set anchor_status and source for each token.
    anchor_status (anchors only): "eligible" | "stopword" | "weak-tail" | "no-phonemes"
    Non-anchors: anchor_status = None (unset).
    source: "cmu" | "g2p" (phoneme lookup hint for OOV)
    Phonemes (phonemes, tail, nucleus) are computed for ALL tokens.
    Trace from internal_rhyme_detector spans: anchor = last word of span.
    """
    # (line_idx, word_idx) of rhyme anchors (last word of each span)
    anchor_positions: Set[Tuple[int, int]] = set()
    for cluster in rhyme_clusters:
        for span in cluster.spans:
            if span.word_indices:
                last_wi = span.word_indices[-1]
                anchor_positions.add((span.line_idx, last_wi))

    for t in tokens:
        clean = t.word.rstrip(".,!?;:'\"").lower()
        seq = word_to_phonemes(clean)

        # 1. Phoneme info for ALL tokens (run first)
        if seq:
            t.phonemes = " ".join(seq.phonemes)
            t.tail = " ".join(seq.rhyme_nucleus) if seq.rhyme_nucleus else None
            nucleus_phone = None
            if seq.rhyme_nucleus:
                for p in seq.rhyme_nucleus:
                    if strip_stress(p) in ARPA_VOWELS:
                        nucleus_phone = p
                        break
            t.nucleus = nucleus_phone
            t.source = "g2p" if seq.source == "g2p" else "cmu"
        else:
            t.source = None

        # 2. anchor_status: only anchors get a status; non-anchors get None
        is_anchor = (t.line_idx, t.word_idx) in anchor_positions
        if not is_anchor:
            t.anchor_status = None
            continue
        if clean in RHYME_ANCHOR_STOPWORDS:
            t.anchor_status = "stopword"
            continue
        # Anchors only from here
        if not seq or not seq.rhyme_nucleus:
            t.anchor_status = "no-phonemes"
        else:
            tail_phones = len(seq.rhyme_nucleus)
            tail_syl = sum(1 for p in seq.rhyme_nucleus if strip_stress(p) in ARPA_VOWELS)
            stressed = bool(seq.has_stressed_rhyme_vowel)
            if stressed:
                # Eligible if tail_phones >= 2 or tail_syl >= 1, else weak-tail
                t.anchor_status = "eligible" if (tail_phones >= 2 or tail_syl >= 1) else "weak-tail"
            else:
                # Eligible if tail_phones >= 4 and tail_syl >= 2, else weak-tail
                t.anchor_status = "eligible" if (tail_phones >= 4 and tail_syl >= 2) else "weak-tail"


def _tokenize_verse(lines: List[str]) -> List[Token]:
    """Build Token list from verse lines with global character offsets."""
    tokens: List[Token] = []
    char_offset = 0
    for line_idx, line in enumerate(lines):
        for word_idx, m in enumerate(WORD_RE.finditer(line)):
            tokens.append(
                Token(
                    word=m.group(0),
                    line_idx=line_idx,
                    word_idx=word_idx,
                    char_start=char_offset + m.start(),
                    char_end=char_offset + m.end(),
                    token_id=f"l{line_idx}.t{word_idx}",
                )
            )
        char_offset += len(line) + 1  # +1 for newline
    return tokens


def _compute_unified_confidence(
    rhyme_clusters: List[RhymeCluster],
    rhyme_chains: List[RhymeChain],
    metaphor_frames: List[MetaphorFrame],
    punchlines: List[Punchline],
    discourse_units: List[DiscourseUnit],
) -> Dict[str, float]:
    """Generate unified confidence scores across tracks."""
    scores: Dict[str, float] = {}

    # Rhyme track
    if rhyme_clusters:
        scores["rhyme_clusters"] = float(sum(c.confidence for c in rhyme_clusters) / len(rhyme_clusters))
    else:
        scores["rhyme_clusters"] = 0.0

    if rhyme_chains:
        scores["rhyme_chains"] = float(sum(c.confidence for c in rhyme_chains) / len(rhyme_chains))
    else:
        scores["rhyme_chains"] = 0.0

    scores["rhyme_overall"] = (scores["rhyme_clusters"] + scores["rhyme_chains"]) / 2.0 if (rhyme_clusters or rhyme_chains) else 0.0

    # Semantics track
    if metaphor_frames:
        scores["metaphor_frames"] = float(sum(m.confidence for m in metaphor_frames) / len(metaphor_frames))
    else:
        scores["metaphor_frames"] = 0.0

    if punchlines:
        scores["punchlines"] = float(sum(p.confidence for p in punchlines) / len(punchlines))
    else:
        scores["punchlines"] = 0.0

    if discourse_units:
        scores["discourse_units"] = float(sum(d.confidence for d in discourse_units) / len(discourse_units))
    else:
        scores["discourse_units"] = 0.0

    sem_parts = [scores["metaphor_frames"], scores["punchlines"], scores["discourse_units"]]
    scores["semantics_overall"] = sum(sem_parts) / 3.0 if any(p > 0 for p in sem_parts) else 0.0

    # Combined
    scores["overall"] = (scores["rhyme_overall"] + scores["semantics_overall"]) / 2.0

    return {k: round(v, 4) for k, v in scores.items()}


def fuse_verse_analysis(
    lines: List[str],
    rhyme_results: Optional[Dict[str, Any]] = None,
    semantics_results: Optional[Dict[str, Any]] = None,
) -> VerseAnnotation:
    """
    Late fusion: combine outputs from rhyme engine + semantics engine.

    Args:
        lines: Verse lines (cleaned, no empty strings).
        rhyme_results: Dict with keys:
            - rhyme_clusters: List[RhymeCluster]
            - rhyme_chains: List[RhymeChain]
        semantics_results: Dict with keys:
            - discourse_units: List[DiscourseUnit]
            - metaphor_frames: List[MetaphorFrame]
            - punchlines: List[Punchline]
            - cohesion_metrics: Dict
            - stance_labels: Dict[int, Dict[str, float]]
            - entity_chains: List (optional)

    Returns:
        Unified VerseAnnotation.
    """
    rhyme_results = rhyme_results or {}
    semantics_results = semantics_results or {}

    # Extract with graceful handling of missing components
    rhyme_clusters: List[RhymeCluster] = rhyme_results.get("rhyme_clusters") or []
    rhyme_chains: List[RhymeChain] = rhyme_results.get("rhyme_chains") or []
    discourse_units: List[DiscourseUnit] = semantics_results.get("discourse_units") or []
    metaphor_frames: List[MetaphorFrame] = semantics_results.get("metaphor_frames") or []
    punchlines: List[Punchline] = semantics_results.get("punchlines") or []
    cohesion_metrics: Dict[str, Any] = semantics_results.get("cohesion_metrics") or {}
    stance_labels_raw = semantics_results.get("stance_labels") or {}

    # Normalize stance_labels: might come as str keys from cohesion output
    stance_labels: Dict[int, Dict[str, float]] = {}
    for k, v in stance_labels_raw.items():
        try:
            idx = int(k)
            stance_labels[idx] = v if isinstance(v, dict) else {"brag": 0.0, "insult": 0.0, "threat": 0.0}
        except (ValueError, TypeError):
            pass

    # Tokenize
    tokens = _tokenize_verse(lines)
    _enrich_tokens_with_anchor_metadata(tokens, rhyme_clusters)
    verse = "\n".join(lines)

    # Unified confidence
    confidence_scores = _compute_unified_confidence(
        rhyme_clusters, rhyme_chains, metaphor_frames, punchlines, discourse_units
    )

    # Density metrics
    density_metrics = compute_density_metrics(lines, rhyme_clusters, rhyme_chains)

    return VerseAnnotation(
        verse=verse,
        lines=lines,
        tokens=tokens,
        rhyme_clusters=rhyme_clusters,
        rhyme_chains=rhyme_chains,
        metaphor_frames=metaphor_frames,
        punchlines=punchlines,
        discourse_units=discourse_units,
        stance_labels=stance_labels,
        cohesion_metrics=cohesion_metrics,
        density_metrics=density_metrics,
        confidence_scores=confidence_scores,
    )
