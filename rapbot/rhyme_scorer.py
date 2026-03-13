"""
Deprecated. Use evo_rhyme.rhyme_resources and evo_rhyme.siamese_scorer instead.
"""

import warnings

warnings.warn(
    "rapbot.rhyme_scorer is deprecated. Use evo_rhyme.rhyme_resources and evo_rhyme.siamese_scorer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from evo_rhyme.rhyme_resources import (
    RHYME_GROUPS,
    RHYME_CSV_PATH,
    load_rhyme_groups,
    get_rhyme_group,
    rhyme_key,
    rhyme_similarity,
)
from evo_rhyme.siamese_scorer import (
    SiameseRhymeScorer,
    get_siamese_model_dir,
    SIAMESE_MODEL_DIR,
)

__all__ = [
    "RHYME_GROUPS",
    "RHYME_CSV_PATH",
    "load_rhyme_groups",
    "get_rhyme_group",
    "rhyme_key",
    "rhyme_similarity",
    "SiameseRhymeScorer",
    "get_siamese_model_dir",
    "SIAMESE_MODEL_DIR",
]
