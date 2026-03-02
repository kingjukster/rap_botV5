"""
verse_analyzer.py

Main interface for the Two-Track Verse Analysis System.
Orchestrates phonetics engine + semantics engine and fuses results.
"""

from __future__ import annotations

from typing import Optional

from rapbot.internal_rhyme_detector import detect_internal_rhymes
from rapbot.rhyme_chain_detector import (
    detect_nucleus_chains,
    detect_rhyme_chains_from_clusters,
    get_rhyme_scheme,
)
from rapbot.discourse_analyzer import analyze_discourse
from rapbot.metaphor_detector import detect_metaphors
from rapbot.punchline_detector import detect_punchlines
from rapbot.semantic_cohesion import analyze_cohesion
from rapbot.fusion_engine import fuse_verse_analysis
from rapbot.verse_annotation import VerseAnnotation


def analyze_verse(
    verse: str,
    use_embeddings: bool = True,
    similarity_threshold: float = 0.6,
    window_lines: int = 2,
) -> VerseAnnotation:
    """
    Full verse analysis: phonetics + semantics, fused into VerseAnnotation.

    Args:
        verse: Raw verse text (multi-line string).
        use_embeddings: If True, use sentence-transformers where available.
        similarity_threshold: Min rhyme similarity for clustering.
        window_lines: Search window for internal rhymes (same line ± N).

    Returns:
        VerseAnnotation with tokens, rhyme_clusters, rhyme_chains,
        metaphor_frames, punchlines, discourse_units, stance_labels,
        cohesion_metrics, confidence_scores.
    """
    lines = [ln.strip() for ln in verse.strip().split("\n") if ln.strip()]
    if not lines:
        return fuse_verse_analysis(
            [],
            rhyme_results={},
            semantics_results={},
        )

    # Phonetics engine
    rhyme_clusters = detect_internal_rhymes(
        lines,
        similarity_threshold=similarity_threshold,
        window_lines=window_lines,
    )
    tail_chains = detect_rhyme_chains_from_clusters(rhyme_clusters, lines)
    nucleus_chains = detect_nucleus_chains(rhyme_clusters, lines)
    all_chains = tail_chains + nucleus_chains
    rhyme_chains = sorted(
        all_chains,
        key=lambda c: (
            0 if getattr(c, "chain_type", "tail") == "nucleus" else 1,
            0 if (getattr(c, "representative_nucleus", None) or "") == "UW1" else 1,
            -len(c.positions),
            -c.confidence,
        ),
    )
    rhyme_scheme = get_rhyme_scheme(lines, similarity_threshold=similarity_threshold)

    rhyme_results = {
        "rhyme_clusters": rhyme_clusters,
        "rhyme_chains": rhyme_chains,
        "rhyme_scheme": rhyme_scheme,
    }

    # Semantics engine
    discourse_units = analyze_discourse(lines, use_embeddings=use_embeddings)
    metaphor_frames = detect_metaphors(lines, use_embeddings=use_embeddings)
    punchlines = detect_punchlines(lines, use_embeddings=use_embeddings)
    cohesion_output = analyze_cohesion(lines, use_embeddings=use_embeddings)

    semantics_results = {
        "discourse_units": discourse_units,
        "metaphor_frames": metaphor_frames,
        "punchlines": punchlines,
        "cohesion_metrics": cohesion_output.get("cohesion_metrics", {}),
        "stance_labels": cohesion_output.get("stance_labels", {}),
        "entity_chains": cohesion_output.get("entity_chains", []),
    }

    return fuse_verse_analysis(lines, rhyme_results, semantics_results)
