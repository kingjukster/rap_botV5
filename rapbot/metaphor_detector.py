"""
Metaphor detector for the Two-Track Verse Analysis System.

Two-stage detection:
1. Metaphor cue detection (literal vs figurative language)
2. Concept linking into metaphor frames (source → target domain mapping)

Clusters related imagery (e.g., poop, deuce, stools, Metamucil → bodily functions)
and maps source domain to target domain (bodily functions → rap skill).

Uses NLTK WordNet for categories and sentence-transformers for embeddings
when available (graceful fallback when missing).
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import re

# ---------------------------------------------------------------------
# NLTK WordNet (required for metaphor detection)
# ---------------------------------------------------------------------

_wn_available = False
_wn = None


def _ensure_wn():
    """Download and load NLTK WordNet if needed."""
    global _wn_available, _wn
    if _wn is not None:
        return _wn_available
    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")  # ensure data exists
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn
        _wn = wn
        _wn_available = True
        return True
    except Exception:
        _wn_available = False
        return False


# ---------------------------------------------------------------------
# Optional sentence-transformers
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


def _mean_pairwise_cosine(cue_words: List[str], model) -> Optional[float]:
    """Compute mean pairwise cosine similarity of cue words. Returns None if < 2 cues or model unavailable."""
    if not cue_words or len(cue_words) < 2 or model is None:
        return None
    try:
        import numpy as np
        embs = model.encode(cue_words)
        n = len(embs)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embs[i], embs[j]) / (
                    np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-9
                )
                total += float(sim)
                count += 1
        return total / count if count > 0 else None
    except Exception:
        return None


def _cues_coherent(cue_words: List[str], model, min_sim: float = 0.5) -> bool:
    """Check if all pairwise cosine similarities of cue words exceed min_sim."""
    if not cue_words or len(cue_words) < 2 or model is None:
        return True
    try:
        import numpy as np
        embs = model.encode(cue_words)
        n = len(embs)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embs[i], embs[j]) / (
                    np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-9
                )
                if float(sim) <= min_sim:
                    return False
        return True
    except Exception:
        return True


@dataclass
class MetaphorFrame:
    """A detected metaphor mapping source domain to target domain."""
    source_domain: str
    target_domain: str
    spans: List[Tuple[int, int]]  # list of (line_start, line_end) 0-based
    confidence: float
    cue_words: Optional[List[str]] = None  # words that triggered the metaphor
    frame_coherence_score: Optional[float] = None  # mean pairwise cosine sim of cues (when embeddings available)


# ---------------------------------------------------------------------
# Domain clusters (source domains common in rap)
# ---------------------------------------------------------------------

# Bodily functions / bathroom (includes saloon/saloons - stool/bar pun, not craft)
BODILY_CUES = {
    "poop", "deuce", "stool", "stools", "bowel", "metamucil", "loosen",
    "john", "bathroom", "toilet", "dump", "crap", "shit", "number two",
    "movements", "droppin", "dropping", "producing", "lyrical bowel",
    "saloon", "saloons",
}

# Violence / combat
VIOLENCE_CUES = {
    "axe", "hacksaw", "lumberjack", "cut", "cutting", "bust", "busting",
    "kill", "murder", "slay", "attack", "war", "fight",
}

# Violence / elimination (womb, cutting, busting in destructive metaphors)
VIOLENCE_ELIMINATION_CUES = {"womb", "cutting", "busting"}

# Prostitution / sex work (often metaphorical for "selling")
SEX_WORK_CUES = {
    "prostitute", "john", "ho", "hooker", "trick",
}

# Craft / construction (womb moved to VIOLENCE_ELIMINATION_CUES)
# saloon/saloons removed: part of stool/bar pun, mapped to BODILY
CRAFT_CUES = {
    "wax", "wax off", "chops", "grind", "build", "construct",
    "cocoon",
}

# Rap-specific target domains
RAP_TARGET_DOMAINS = {"rap", "bars", "beats", "lyrics", "flow", "skill"}

# Mapping: source cluster name -> typical target in rap
DOMAIN_MAP = {
    "bodily_functions": "rap_skill",
    "violence": "rap_skill",
    "sex_work": "selling_out",
    "craft": "rap_craft",
}


def _get_wordnet_lexname(word: str) -> Optional[str]:
    """Get highest-level lexical category for a word from WordNet."""
    if not _ensure_wn():
        return None
    synsets = _wn.synsets(word)
    if not synsets:
        return None
    s = synsets[0]
    lexname = getattr(s, "lexname", None)
    if lexname:
        # lexname is e.g. "noun.animal" -> return "animal"
        parts = lexname.split(".")
        return parts[-1] if len(parts) > 1 else lexname
    return None


def _match_cue_cluster(tokens: List[str]) -> Optional[Tuple[str, Set[str]]]:
    """Match tokens against cue clusters; return first (cluster_name, matched_words)."""
    matches = _match_all_cue_clusters(tokens)
    return matches[0] if matches else None


def _match_all_cue_clusters(tokens: List[str]) -> List[Tuple[str, Set[str]]]:
    """
    Match tokens against all cue clusters; return list of (cluster_name, matched_words).
    Enables multi-frame lines: a line with violence + bodily cues returns both.
    """
    lower = [t.lower().rstrip(".,!?;:'\"") for t in tokens]
    violence_all = VIOLENCE_CUES | VIOLENCE_ELIMINATION_CUES
    results: List[Tuple[str, Set[str]]] = []
    for cluster, cues in [
        ("bodily_functions", BODILY_CUES),
        ("violence", violence_all),
        ("sex_work", SEX_WORK_CUES),
        ("craft", CRAFT_CUES),
    ]:
        matched = {w for w in lower if w in cues}
        for i in range(len(lower) - 1):
            bigram = f"{lower[i]} {lower[i+1]}"
            if bigram in cues or (lower[i] + lower[i+1].replace("'", "")) in cues:
                matched.add(bigram)
        if matched:
            results.append((cluster, matched))
    return results


def _is_likely_figurative(line: str, cue_words: Set[str]) -> bool:
    """
    Stage 1: Cue detection. If line contains metaphor cues in a context
    that suggests figurative use (e.g., "bars" + "stools" = metaphor).
    """
    lower = line.lower()
    words = set(re.findall(r"[a-z']+", lower))

    # Direct cues
    if cue_words & words:
        return True

    # Co-occurrence: "beats", "bars", "lyrics" + bodily/violence cues
    rap_terms = {"beats", "bars", "lyrics", "flow", "rap", "verse"}
    all_cues = BODILY_CUES | VIOLENCE_CUES | VIOLENCE_ELIMINATION_CUES | CRAFT_CUES
    if (rap_terms & words) and (all_cues & words):
        return True

    return False


def _infer_target_domain(lines: List[str], source_domain: str) -> str:
    """Infer target domain from verse context (rap, skill, etc.)."""
    all_text = " ".join(lines).lower()
    if any(w in all_text for w in ["bars", "beats", "lyrics", "flow", "rap"]):
        return "rap_skill"
    return DOMAIN_MAP.get(source_domain, "abstract")


def detect_metaphors(
    lines: List[str],
    use_embeddings: bool = True,
) -> List[MetaphorFrame]:
    """
    Two-stage metaphor detection: cue detection then concept linking.

    Args:
        lines: List of line strings.
        use_embeddings: If True and sentence-transformers available, use
            embeddings to cluster related imagery across lines.

    Returns:
        List of MetaphorFrame with source_domain, target_domain, spans, confidence.
    """
    if not lines:
        return []

    lines = [ln.strip() for ln in lines if ln.strip()]
    frames: List[MetaphorFrame] = []
    seen_spans: Set[Tuple[int, int]] = set()

    # Stage 1: Find lines with metaphor cues (multi-frame: one line can match multiple domains)
    cue_lines: List[Tuple[int, Set[str], str]] = []  # (line_idx, cues, cluster)
    for i, line in enumerate(lines):
        tokens = re.findall(r"[A-Za-z']+", line)
        for cluster, matched in _match_all_cue_clusters(tokens):
            if _is_likely_figurative(line, matched):
                cue_lines.append((i, matched, cluster))

    # Stage 2: Cluster into contiguous spans and build frames
    i = 0
    while i < len(cue_lines):
        line_idx, cues, cluster = cue_lines[i]
        span_start = line_idx
        span_end = line_idx
        all_cues = set(cues)

        # Extend span to adjacent cue lines
        j = i + 1
        while j < len(cue_lines):
            nidx, ncues, ncluster = cue_lines[j]
            if ncluster == cluster and nidx <= span_end + 2:
                span_end = nidx
                all_cues |= ncues
                j += 1
            else:
                break

        span = (span_start, span_end)
        if span not in seen_spans:
            seen_spans.add(span)
            target = _infer_target_domain(lines[span_start : span_end + 1], cluster)
            conf = min(1.0, 0.5 + 0.1 * len(all_cues) + (0.2 if _ensure_wn() else 0))
            frames.append(
                MetaphorFrame(
                    source_domain=cluster,
                    target_domain=target,
                    spans=[span],
                    confidence=conf,
                    cue_words=sorted(all_cues),
                )
            )
        i = j

    # Optional: use embeddings to merge semantically similar frames (require cosine_sim > 0.5)
    if use_embeddings and len(frames) > 1:
        model = _load_embed_model()
        if model is not None:
            merged = []
            for f in frames:
                absorbed = False
                for m in merged:
                    if m.target_domain == f.target_domain and m.source_domain == f.source_domain:
                        combined_cues = list((m.cue_words or []) + (f.cue_words or []))
                        combined_cues = list(dict.fromkeys(combined_cues))
                        if _cues_coherent(combined_cues, model, min_sim=0.5):
                            for sp in f.spans:
                                if sp not in [(s, e) for s, e in m.spans]:
                                    m.spans.append(sp)
                            m.confidence = max(m.confidence, f.confidence)
                            if f.cue_words:
                                m.cue_words = combined_cues
                            absorbed = True
                            break
                if not absorbed:
                    merged.append(MetaphorFrame(
                        source_domain=f.source_domain,
                        target_domain=f.target_domain,
                        spans=list(f.spans),
                        confidence=f.confidence,
                        cue_words=list(f.cue_words) if f.cue_words else None,
                    ))
            frames = merged

    # Add frame_coherence_score when embeddings available
    if use_embeddings:
        model = _load_embed_model()
        if model is not None:
            for f in frames:
                if f.cue_words and len(f.cue_words) >= 2:
                    f.frame_coherence_score = _mean_pairwise_cosine(
                        list(f.cue_words), model
                    )

    return frames
