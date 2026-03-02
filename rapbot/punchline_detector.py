"""
Punchline detector for the Two-Track Verse Analysis System.

Detects idiom → double meaning pivots (e.g., "hand job" = manual work vs sexual).
Uses pattern matching and optional embedding shifts to identify punchlines
where a word/phrase pivots between literal and figurative meanings.
"""

from dataclasses import dataclass
from typing import List, Optional

import re

# ---------------------------------------------------------------------
# Optional NLTK WordNet for word-sense ambiguity
# ---------------------------------------------------------------------

_nltk_wn_available = False


def _ensure_wn():
    """Check if NLTK WordNet is available."""
    global _nltk_wn_available
    if _nltk_wn_available:
        return True
    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        _nltk_wn_available = True
        return True
    except Exception:
        _nltk_wn_available = False
        return False


def _word_sense_ambiguity_count(word: str) -> int:
    """Return number of WordNet synsets for word (0 if NLTK unavailable)."""
    if not _ensure_wn():
        return 0
    try:
        from nltk.corpus import wordnet as wn
        return len(wn.synsets(word))
    except Exception:
        return 0


# ---------------------------------------------------------------------
# Optional sentence-transformers for embedding shift detection
# ---------------------------------------------------------------------

_embed_model = None
_EMBED_AVAILABLE = False


def _load_embed_model():
    """Lazy load sentence-transformers; return None if unavailable."""
    global _embed_model, _EMBED_AVAILABLE
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBED_AVAILABLE = True
        return _embed_model
    except ImportError:
        _EMBED_AVAILABLE = False
        return None


@dataclass
class Punchline:
    """A detected punchline with pivot word and double meaning."""
    pivot_word: str
    line: str
    line_index: int
    original_meaning: str
    double_meaning: str
    confidence: float


# ---------------------------------------------------------------------
# Idiom / double-meaning lexicons
# ---------------------------------------------------------------------

# Format: (pattern, literal_meaning, figurative_meaning) or (pattern, literal, figurative, confidence_override)
# Demoted: hold a candle - not a true punchline, low confidence
PIVOT_PATTERNS = [
    (r"\bhand\s*job\b", "manual labor / task done by hand", "sexual act"),
    (r"\bhold\s*a\s*candle\b", "hold a literal candle", "measure up / compare", 0.3),
    (r"\bwax\s*off\b", "remove wax (e.g. karate)", "leave / get lost"),
    (r"\bjack[\s-]?off(s)?\b", "masturbation", "jerk / foolish person"),
    (r"\bcome\s*to\s*grips\b", "grip physically", "accept or deal with"),
    (r"\bnumber\s*one\b", "ranking first", "urine"),
    (r"\bnumber\s*two\b", "ranking second", "feces"),
    (r"\bdroppin['']?\s*a\s*deuce\b", "dropping the number two", "defecating"),
    (r"\bon\s*the\s*john\b", "on the toilet", "name John / sex work"),
    (r"\bbars\b", "physical bars / prison", "rap verses"),
    (r"\bstools\b", "chairs / seating", "feces"),
    (r"\bbeat(s)?\b", "rhythm / music", "hit / assault"),
    (r"\bchops\b", "cutting meat", "rapping skills"),
    (r"\bspin\s*(another)?\s*cocoon\b", "form cocoon", "create / produce again"),
    (r"\bcuttin['']?\s*you\s*from\s*.*\s*womb\b", "cut from womb", "eliminate / destroy"),
    (r"\bbustin['']?\s*you\b", "busting open", "defeating / attacking"),
    (r"\bpoop\s*is\s*my\s*pseudonym\b", "feces as fake name", "wordplay on feces/rap"),
    (r"\bpencils\s*are\s*number\s*two\b", "pencils ranking second", "pencils = feces"),
    (r"\bgot\s*my\s*stools\s*in\s*['']?em\b", "stools inside", "bars contain feces pun"),
]

# Single-word pivots: word -> (literal, figurative)
SINGLE_PIVOTS = {
    "bars": ("metal bars, prison", "rap verses"),
    "beats": ("rhythm, music", "hitting, assault"),
    "flow": ("liquid movement", "rap delivery"),
    "drop": ("let fall", "release music"),
    "spit": ("saliva", "rap/perform"),
    "burn": ("fire damage", "insult heavily"),
    "murder": ("kill", "dominate lyrically"),
    "kill": ("cause death", "perform excellently"),
    "slay": ("kill violently", "perform excellently"),
}


def _literal_figurative_distinct(literal: str, figurative: str) -> bool:
    """Require dual interpretation: literal and figurative must be distinct."""
    lit_lower = literal.lower()
    fig_lower = figurative.lower()
    # Overlap check: if one contains the other or they share key tokens, may be too similar
    lit_tokens = set(re.findall(r"[a-z']+", lit_lower))
    fig_tokens = set(re.findall(r"[a-z']+", fig_lower))
    overlap = lit_tokens & fig_tokens
    # Allow some overlap (common words like "the", "a") but reject when meanings are near-identical
    meaningful = {w for w in lit_tokens | fig_tokens if len(w) > 2}
    if len(meaningful) <= 2:
        return True
    overlap_ratio = len(overlap & meaningful) / max(1, len(meaningful))
    return overlap_ratio < 0.7


def _match_punchline(line: str, line_idx: int) -> Optional[Punchline]:
    """Check if line contains a known pivot pattern; return Punchline if so."""
    line_lower = line.lower()

    for item in PIVOT_PATTERNS:
        if len(item) == 4:
            pattern, orig, double, conf_override = item
        else:
            pattern, orig, double = item
            conf_override = None

        m = re.search(pattern, line_lower, re.I)
        if m:
            if not _literal_figurative_distinct(orig, double):
                continue
            pivot = m.group(0).strip()
            if conf_override is not None:
                conf = conf_override
            else:
                conf = 0.7 + (0.2 if len(pattern) > 20 else 0.1)
            # Confidence boost for word-sense ambiguity (multiple WordNet synsets)
            if _ensure_wn() and conf_override is None:
                pivot_words = re.findall(r"[a-z']+", pivot)
                for pw in pivot_words:
                    n_synsets = _word_sense_ambiguity_count(pw)
                    if n_synsets >= 3:
                        conf = min(1.0, conf + 0.1)
                        break
            return Punchline(
                pivot_word=pivot,
                line=line.strip(),
                line_index=line_idx,
                original_meaning=orig,
                double_meaning=double,
                confidence=min(1.0, conf),
            )

    # Single-word pivots when in suggestive context
    words = re.findall(r"[A-Za-z']+", line_lower)
    for w in words:
        clean = w.rstrip(".,!?;:'\"")
        if clean in SINGLE_PIVOTS:
            orig, double = SINGLE_PIVOTS[clean]
            if not _literal_figurative_distinct(orig, double):
                continue
            if len(words) >= 3:
                conf = 0.5
                # Confidence boost for word-sense ambiguity (multiple WordNet synsets)
                if _ensure_wn():
                    n_synsets = _word_sense_ambiguity_count(clean)
                    if n_synsets >= 3:
                        conf = min(1.0, conf + 0.15)
                return Punchline(
                    pivot_word=clean,
                    line=line.strip(),
                    line_index=line_idx,
                    original_meaning=orig,
                    double_meaning=double,
                    confidence=conf,
                )

    return None


def _embedding_shift_score(line: str, context_lines: List[str], model) -> float:
    """
    Optional: detect if line has high semantic shift from context
    (suggesting punchline/twist).
    """
    if model is None or not line or not context_lines:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return 0.0
    embs = model.encode([line] + context_lines)
    line_emb = embs[0]
    ctx_emb = np.mean(embs[1:], axis=0)
    sim = np.dot(line_emb, ctx_emb) / (np.linalg.norm(line_emb) * np.linalg.norm(ctx_emb) + 1e-9)
    # Low similarity = high shift = possibly punchline
    shift = 1.0 - (sim + 1) / 2
    return float(shift)


MIN_PUNCHLINE_CONFIDENCE = 0.75


def detect_punchlines(
    lines: List[str],
    use_embeddings: bool = True,
    min_confidence: float = MIN_PUNCHLINE_CONFIDENCE,
) -> List[Punchline]:
    """
    Detect idiom → double meaning pivots in verse lines.

    Args:
        lines: List of line strings.
        use_embeddings: If True and sentence-transformers available, use
            embedding shift to boost confidence for lines that diverge from context.
        min_confidence: Only return punchlines with confidence >= this (default 0.75).

    Returns:
        List of Punchline with pivot_word, line, original_meaning, double_meaning, confidence.
    """
    if not lines:
        return []

    lines = [ln.strip() for ln in lines if ln.strip()]
    results: List[Punchline] = []

    for i, line in enumerate(lines):
        pl = _match_punchline(line, i)
        if pl:
            if use_embeddings:
                model = _load_embed_model()
                if model is not None:
                    ctx = [lines[j] for j in range(max(0, i - 2), min(len(lines), i + 3)) if j != i]
                    shift = _embedding_shift_score(line, ctx, model)
                    if shift > 0.5:
                        pl = Punchline(
                            pivot_word=pl.pivot_word,
                            line=pl.line,
                            line_index=pl.line_index,
                            original_meaning=pl.original_meaning,
                            double_meaning=pl.double_meaning,
                            confidence=min(1.0, pl.confidence + 0.15),
                        )
            if pl.confidence >= min_confidence:
                results.append(pl)

    return results
