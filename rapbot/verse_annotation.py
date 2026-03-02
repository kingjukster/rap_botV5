"""
verse_annotation.py

Unified annotation schema for the Two-Track Verse Analysis System.
Combines phonetics (rhyme clusters, chains) and semantics (metaphors, punchlines, discourse).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Re-exports from phonetics engine
from rapbot.internal_rhyme_detector import RhymeCluster, Span
from rapbot.rhyme_chain_detector import RhymeChain

# Re-exports from semantics engine
from rapbot.discourse_analyzer import DiscourseUnit
from rapbot.metaphor_detector import MetaphorFrame
from rapbot.punchline_detector import Punchline


@dataclass
class Token:
    """A single word token with position metadata."""
    word: str
    line_idx: int
    word_idx: int
    char_start: int
    char_end: int
    token_id: Optional[str] = None
    anchor_status: Optional[str] = None  # "eligible" | "stopword" | "ignored" | "weak-tail"
    source: Optional[str] = None  # "cmu" | "g2p" for phoneme lookup hint (OOV)
    phonemes: Optional[str] = None  # space-joined full phonemes (rhyme anchors only)
    tail: Optional[str] = None  # space-joined tail phonemes (rhyme anchors only)
    nucleus: Optional[str] = None  # single phoneme e.g. UW1 (rhyme anchors only)


@dataclass
class VerseAnnotation:
    """
    Unified verse analysis combining phonetics + semantics tracks.
    """
    verse: str
    lines: List[str]
    tokens: List[Token]
    rhyme_clusters: List[RhymeCluster]
    rhyme_chains: List[RhymeChain]
    metaphor_frames: List[MetaphorFrame]
    punchlines: List[Punchline]
    discourse_units: List[DiscourseUnit]
    stance_labels: Dict[int, Dict[str, float]]
    cohesion_metrics: Dict[str, Any]
    density_metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "verse": self.verse,
            "lines": self.lines,
            "tokens": [
                {
                    "word": t.word,
                    "line_idx": t.line_idx,
                    "word_idx": t.word_idx,
                    "char_start": t.char_start,
                    "char_end": t.char_end,
                    "token_id": t.token_id,
                    "anchor_status": t.anchor_status,
                    "source": t.source,
                    "phonemes": t.phonemes,
                    "tail": t.tail,
                    "nucleus": t.nucleus,
                }
                for t in self.tokens
            ],
            "rhyme_clusters": [
                {
                    "family_id": c.family_id,
                    "spans": [
                        {
                            "text": s.text,
                            "start_char": s.start_char,
                            "end_char": s.end_char,
                            "line_idx": s.line_idx,
                            "word_indices": list(s.word_indices),
                        }
                        for s in c.spans
                    ],
                    "confidence": c.confidence,
                    "rhyme_type": c.rhyme_type,
                    "representative_tail": list(c.representative_tail) if getattr(c, "representative_tail", None) else None,
                    "top_anchors": getattr(c, "top_anchors", None),
                    "example_spans": getattr(c, "example_spans", None),
                }
                for c in self.rhyme_clusters
            ],
            "rhyme_chains": [
                {
                    "chain_id": c.chain_id,
                    "pattern": c.pattern,
                    "positions": list(c.positions),
                    "family_id": c.family_id,
                    "confidence": c.confidence,
                    "chain_continuity_score": getattr(c, "chain_continuity_score", 0.0),
                    "chain_type": getattr(c, "chain_type", "tail"),
                    "representative_nucleus": getattr(c, "representative_nucleus", None),
                    "representative_tail": getattr(c, "representative_tail", None),
                    "positions_with_phonemes": getattr(c, "positions_with_phonemes", None),
                }
                for c in self.rhyme_chains
            ],
            "metaphor_frames": [
                {
                    "source_domain": m.source_domain,
                    "target_domain": m.target_domain,
                    "spans": list(m.spans),
                    "confidence": m.confidence,
                    "cue_words": m.cue_words,
                    "frame_coherence_score": getattr(m, "frame_coherence_score", None),
                }
                for m in self.metaphor_frames
            ],
            "punchlines": [
                {
                    "pivot_word": p.pivot_word,
                    "line": p.line,
                    "line_index": p.line_index,
                    "original_meaning": p.original_meaning,
                    "double_meaning": p.double_meaning,
                    "confidence": p.confidence,
                }
                for p in self.punchlines
            ],
            "discourse_units": [
                {
                    "type": d.type,
                    "span": list(d.span),
                    "confidence": d.confidence,
                }
                for d in self.discourse_units
            ],
            "stance_labels": {str(k): v for k, v in self.stance_labels.items()},
            "cohesion_metrics": self.cohesion_metrics,
            "density_metrics": self.density_metrics,
            "confidence_scores": self.confidence_scores,
        }

    def to_json(self, indent: Optional[int] = 2, ensure_ascii: bool = False) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)
