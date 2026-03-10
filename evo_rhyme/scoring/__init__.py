"""evo_rhyme scoring modules."""

from evo_rhyme.scoring.internal_rhyme import (
    LineFeatures,
    build_line_features,
    score_internal_rhyme,
)
from evo_rhyme.scoring.line_penalty import apply_line_penalty, score_line_penalty
from evo_rhyme.scoring.penalties import repetition_penalty, weak_tail_penalty
from evo_rhyme.scoring.syllable_balance import score_syllable_balance

try:
    from evo_rhyme.scoring.end_rhyme import score_end_rhyme
    __all__ = [
        "LineFeatures",
        "apply_line_penalty",
        "build_line_features",
        "score_end_rhyme",
        "score_internal_rhyme",
        "score_line_penalty",
        "score_syllable_balance",
        "repetition_penalty",
        "weak_tail_penalty",
    ]
except ImportError:
    __all__ = [
        "LineFeatures",
        "apply_line_penalty",
        "build_line_features",
        "score_internal_rhyme",
        "score_line_penalty",
        "score_syllable_balance",
        "repetition_penalty",
        "weak_tail_penalty",
    ]
