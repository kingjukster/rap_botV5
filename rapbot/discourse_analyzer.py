"""
Discourse analyzer for the Two-Track Verse Analysis System.

Segments verses into discourse units: setup, elaboration, punchline,
metaphor_extension, transition. Supports line-level or couplet-level
classification using rule-based heuristics and optional embedding similarity.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import re


@dataclass
class DiscourseUnit:
    """A span of lines classified as a discourse unit."""
    type: str  # setup | elaboration | punchline | metaphor_extension | transition
    span: tuple  # (start_line_idx, end_line_idx) 0-based
    confidence: float = 0.0


# ---------------------------------------------------------------------
# Optional sentence-transformers for embedding-based similarity
# ---------------------------------------------------------------------

_SENTENCE_TRANSFORMER_AVAILABLE = False
_embed_model = None


def _load_embed_model():
    """Lazy load sentence-transformers; return None if unavailable."""
    global _SENTENCE_TRANSFORMER_AVAILABLE, _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _SENTENCE_TRANSFORMER_AVAILABLE = True
        return _embed_model
    except ImportError:
        _SENTENCE_TRANSFORMER_AVAILABLE = False
        return None


def _embed_similarity(lines: List[str], model) -> List[List[float]]:
    """Compute pairwise embedding similarities for lines."""
    if model is None or not lines:
        return []
    try:
        import numpy
    except ImportError:
        return []
    embs = model.encode(lines)
    from numpy import dot
    from numpy.linalg import norm
    n = len(embs)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                s = float(dot(embs[i], embs[j]) / (norm(embs[i]) * norm(embs[j]) + 1e-9))
                sim_matrix[i][j] = (s + 1) / 2  # scale to [0,1]
    return sim_matrix


# ---------------------------------------------------------------------
# Rule-based discourse heuristics
# ---------------------------------------------------------------------

# Patterns suggesting setup (introductory)
SETUP_PATTERNS = [
    r"\b(you|they|we|i)\s+(make|got|can't|need)\b",
    r"\b(come|get|bring)\s+(to|here|it)\b",
    r"^(let me|watch |listen |see )",
]

# Patterns suggesting punchline (wordplay, twist, strong assertion)
PUNCHLINE_PATTERNS = [
    r"\b(hand job|handjob)\b",  # double meaning
    r"\b(loosen|lumberjack|hacksaw)\b",
    r"\b(pseudonym|number two|droppin['']?\s*a\s*deuce)\b",
    r"\b(cocoon|womb)\b",
    r"\b(pivots?|twist|switch)\b",
    r"(\w+)\s+like\s+\w+",  # simile at end
    r"\blike\s+\w+\s+(when|and|but)\b",
]

# Patterns suggesting metaphor extension
METAPHOR_EXTENSION_PATTERNS = [
    r"\b(beats?|bars?|stools?|bowel|saloon|Metamucil)\b",
    r"\b(poop|deuce|lyrical)\b",
    r"\b(producing|movements)\b",
    r"\b(john|prostitute)\b",
]

# Patterns suggesting transition
TRANSITION_PATTERNS = [
    r"^(and |but |so |then |now |'cause |'cause)\b",
    r"\b(and\s+i|but\s+i|so\s+i)\b",
    r"^I['']?m\s+(about|'bout|goin|gonna)\s+",
    r"^(when|after|before)\s+",
]


def _score_line_for_type(line: str, patterns: List[str]) -> float:
    """Return a score in [0,1] for how well line matches pattern type."""
    text = line.lower().strip()
    if not text:
        return 0.0
    matches = sum(1 for p in patterns if re.search(p, text, re.I))
    return min(1.0, matches * 0.4)  # cap at 1.0


def _classify_line_rules(line: str, line_idx: int, total: int) -> tuple:
    """Return (unit_type, confidence) for a single line using rules."""
    scores = {
        "setup": _score_line_for_type(line, SETUP_PATTERNS),
        "punchline": _score_line_for_type(line, PUNCHLINE_PATTERNS),
        "metaphor_extension": _score_line_for_type(line, METAPHOR_EXTENSION_PATTERNS),
        "transition": _score_line_for_type(line, TRANSITION_PATTERNS),
    }

    # Position heuristics
    if line_idx == 0:
        scores["setup"] = max(scores["setup"], 0.5)
    if line_idx == total - 1:
        scores["punchline"] = max(scores["punchline"], 0.3)
    if line_idx > 0 and line_idx < total - 1:
        scores["elaboration"] = 0.3 if not any(s > 0.5 for s in scores.values()) else 0.0
    else:
        scores["elaboration"] = 0.0

    best = max(scores.items(), key=lambda x: x[1])
    return (best[0], best[1] if best[1] > 0 else 0.2)


def analyze_discourse(
    lines: List[str],
    use_embeddings: bool = True,
    couplet_level: bool = False,
) -> List[DiscourseUnit]:
    """
    Segment verse into discourse units.

    Args:
        lines: List of line strings (e.g. from verse.split('\\n')).
        use_embeddings: If True and sentence-transformers available, use embedding
            similarity to refine boundaries.
        couplet_level: If True, merge into couplets (2-line units) where possible.

    Returns:
        List of DiscourseUnit with type, span, confidence.
    """
    if not lines:
        return []

    # Clean lines
    lines = [ln.strip() for ln in lines if ln.strip()]

    # Per-line rule-based classification
    per_line: List[tuple] = []
    for i, line in enumerate(lines):
        utype, conf = _classify_line_rules(line, i, len(lines))
        per_line.append((utype, conf))

    # Optional embedding refinement: if adjacent lines have high similarity,
    # keep same unit type; if low, likely transition
    if use_embeddings and len(lines) >= 2:
        model = _load_embed_model()
        if model is not None:
            sim_matrix = _embed_similarity(lines, model)
            for i in range(len(lines) - 1):
                sim = sim_matrix[i][i + 1] if i + 1 < len(sim_matrix[i]) else 0.5
                if sim < 0.4 and per_line[i + 1][0] == "elaboration":
                    per_line[i + 1] = ("transition", max(0.4, per_line[i + 1][1]))

    # Build units by merging consecutive same-type lines
    units: List[DiscourseUnit] = []
    i = 0
    while i < len(lines):
        utype, conf = per_line[i]
        j = i + 1
        if couplet_level and j < len(lines):
            # Optionally merge couplets: two lines same or compatible type
            ntype, nconf = per_line[j]
            if ntype == utype or (utype in ("elaboration", "metaphor_extension") and ntype in ("elaboration", "metaphor_extension")):
                j += 1
                conf = (conf + nconf) / 2

        while j < len(lines) and per_line[j][0] == utype:
            conf = (conf * (j - i) + per_line[j][1]) / (j - i + 1)
            j += 1

        units.append(DiscourseUnit(type=utype, span=(i, j - 1), confidence=min(1.0, conf)))
        i = j

    return units
