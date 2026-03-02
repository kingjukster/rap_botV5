"""Core Rap Bot modules."""

__all__ = [
    "meter_utils",
    "rhyme_planner",
    "rhyme_scorer",
    "rhyme_detector",
    "phoneme_converter",
    "rhyme_similarity",
    "internal_rhyme_detector",
    "rhyme_chain_detector",
    "discourse_analyzer",
    "metaphor_detector",
    "punchline_detector",
    "semantic_cohesion",
    "verse_annotation",
    "fusion_engine",
    "verse_analyzer",
]

# Import main classes for convenience
from rapbot.rhyme_detector import RhymeDetector, RhymeType, RhymeResult, InternalRhyme
from rapbot.phoneme_converter import PhonemeSequence, word_to_phonemes
from rapbot.rhyme_similarity import compute_rhyme_similarity, word_rhyme_similarity
from rapbot.internal_rhyme_detector import (
    RhymeCluster,
    RepetitionGroup,
    Span,
    detect_internal_rhymes,
    detect_repetitions,
)
from rapbot.rhyme_chain_detector import RhymeChain, detect_rhyme_chains, get_rhyme_scheme

try:
    from rapbot.discourse_analyzer import DiscourseUnit, analyze_discourse
    from rapbot.metaphor_detector import MetaphorFrame, detect_metaphors
    from rapbot.punchline_detector import Punchline, detect_punchlines
    from rapbot.semantic_cohesion import analyze_cohesion
except ImportError:
    pass

# Two-Track Verse Analysis System
try:
    from rapbot.verse_annotation import Token, VerseAnnotation
    from rapbot.fusion_engine import fuse_verse_analysis
    from rapbot.verse_analyzer import analyze_verse
except ImportError:
    pass
