"""
Semantic cohesion analyzer for the Two-Track Verse Analysis System.

Provides:
- Entity chains (beats, bars, topic-related nouns)
- Frame consistency (does imagery repeat?)
- Stance detection: brag, insult, threat (embeddings or keyword heuristics)

Output: cohesion_metrics dict, entity_chains list, stance_labels dict.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import re
from collections import defaultdict

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


# ---------------------------------------------------------------------
# Stance keyword heuristics
# ---------------------------------------------------------------------

# Fine-grained stance patterns: dominance, aggression, superiority, humiliation
# Mapping: brag = dominance + superiority; insult = humiliation; threat = aggression

DOMINANCE_PATTERNS = [
    r"\b(murder|kill|slay|destroy|dominate)\b",
    r"\b(best|greatest|number\s*one|top|elite|king|boss)\b",
]

SUPERIORITY_PATTERNS = [
    r"\b(real|true|authentic)\s+(mc|emcee|rapper)\b",
    r"\b(can['']?t\s+hold\s+a\s*candle|ain['']?t\s+nobody)\b",
    r"\b(i['']?m\s+the|i\s+am\s+the)\b",
    r"\bbeats?\s+(are|is)\s+like\s+my\b",
]

HUMILIATION_PATTERNS = [
    r"\b(wack|weak|trash|garbage|wack)\b",
    r"\b(jack[\s-]?off|dumb|stupid|fool)\b",
    r"\b(can['']?t\s+hold|can['']?t\s+compare)\b",
    r"\b(mumble\s*rap|mumble)\b",
    r"\b(bitch|ass)\b",
]

AGGRESSION_PATTERNS = [
    r"\b(cut|cutting|bust|busting)\s+(you|them|him)\b",
    r"\b(murder|kill|slay|destroy)\s+(you|them)\b",
    r"\b(coming\s+for|coming\s+at)\b",
    r"\b(axe|hacksaw)\s+(to|for)\b",
    r"\b(eliminate|erase)\b",
]

# Backward compatibility: composite pattern sets
BRAG_PATTERNS = DOMINANCE_PATTERNS + SUPERIORITY_PATTERNS
INSULT_PATTERNS = HUMILIATION_PATTERNS
THREAT_PATTERNS = AGGRESSION_PATTERNS


def _score_stance(line: str) -> Dict[str, float]:
    """Return fine-grained stance scores (dominance, aggression, superiority, humiliation)
    plus composite brag, insult, threat for backward compatibility."""
    lower = line.lower()

    # Fine-grained labels
    dominance = sum(0.35 for p in DOMINANCE_PATTERNS if re.search(p, lower))
    superiority = sum(0.35 for p in SUPERIORITY_PATTERNS if re.search(p, lower))
    humiliation = sum(0.35 for p in HUMILIATION_PATTERNS if re.search(p, lower))
    aggression = sum(0.35 for p in AGGRESSION_PATTERNS if re.search(p, lower))

    scores = {
        "dominance": min(1.0, dominance),
        "aggression": min(1.0, aggression),
        "superiority": min(1.0, superiority),
        "humiliation": min(1.0, humiliation),
        # Composite outputs for backward compatibility
        "brag": min(1.0, dominance + superiority),
        "insult": min(1.0, humiliation),
        "threat": min(1.0, aggression),
    }

    return scores


# ---------------------------------------------------------------------
# Entity extraction (nouns, rap terms)
# ---------------------------------------------------------------------

RAP_ENTITIES = {"bars", "beats", "flow", "lyrics", "verse", "rap", "rhymes"}
BODILY_ENTITIES = {"poop", "deuce", "stools", "bowel", "john", "metamucil"}
VIOLENCE_ENTITIES = {"axe", "hacksaw", "lumberjack", "cocoon", "womb"}
CRAFT_ENTITIES = {"wax", "chops", "saloon", "pencils", "pseudonym"}

ALL_ENTITY_TERMS = RAP_ENTITIES | BODILY_ENTITIES | VIOLENCE_ENTITIES | CRAFT_ENTITIES


def _extract_entities(line: str) -> List[Tuple[str, str]]:
    """Extract (entity, category) from line."""
    words = re.findall(r"[A-Za-z']+", line.lower())
    out = []
    for w in words:
        clean = w.rstrip(".,!?;:'\"")
        if clean in RAP_ENTITIES:
            out.append((clean, "rap"))
        elif clean in BODILY_ENTITIES:
            out.append((clean, "bodily"))
        elif clean in VIOLENCE_ENTITIES:
            out.append((clean, "violence"))
        elif clean in CRAFT_ENTITIES:
            out.append((clean, "craft"))
    return out


@dataclass
class EntityChain:
    """A chain of related entities across the verse."""
    entities: List[str]
    category: str
    line_indices: List[int]


def _build_entity_chains(lines: List[str]) -> List[EntityChain]:
    """Build entity chains from verse lines."""
    chains_by_category: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for i, line in enumerate(lines):
        for entity, cat in _extract_entities(line):
            chains_by_category[cat].append((entity, i))

    result = []
    for cat, pairs in chains_by_category.items():
        entities = []
        indices = []
        for e, idx in pairs:
            if not entities or (entities[-1] != e or indices[-1] != idx):
                entities.append(e)
                indices.append(idx)
        if entities:
            result.append(EntityChain(entities=entities, category=cat, line_indices=indices))

    return result


def _frame_consistency(lines: List[str], entity_chains: List[EntityChain]) -> float:
    """Measure if imagery/frames repeat across the verse (0-1)."""
    if not lines or not entity_chains:
        return 0.0
    # Count unique entities per category
    total = sum(len(c.entities) for c in entity_chains)
    unique = sum(len(set(c.entities)) for c in entity_chains)
    if total == 0:
        return 0.0
    # High repetition = high consistency
    repetition = 1.0 - (unique / total) if total > 0 else 0.0
    # Also factor in span: if same category appears in multiple chains
    categories = [c.category for c in entity_chains]
    cat_diversity = len(set(categories)) / max(1, len(categories))
    return min(1.0, 0.5 * (1 - cat_diversity) + 0.5 * repetition + 0.2)


def analyze_cohesion(
    lines: List[str],
    use_embeddings: bool = True,
) -> Dict:
    """
    Analyze semantic cohesion of a verse.

    Returns dict with:
        - cohesion_metrics: {entity_density, frame_consistency, stance_strength}
        - entity_chains: list of EntityChain
        - stance_labels: {line_idx: {brag, insult, threat}}
    """
    if not lines:
        return {
            "cohesion_metrics": {},
            "entity_chains": [],
            "stance_labels": {},
        }

    lines = [ln.strip() for ln in lines if ln.strip()]
    entity_chains = _build_entity_chains(lines)

    # Entity density: entities per line
    total_entities = sum(len(c.entities) for c in entity_chains)
    entity_density = total_entities / len(lines) if lines else 0.0

    # Frame consistency
    frame_consistency = _frame_consistency(lines, entity_chains)

    # Stance per line
    stance_labels: Dict[int, Dict[str, float]] = {}
    for i, line in enumerate(lines):
        stance_labels[i] = _score_stance(line)

    # Aggregate stance strength
    brag_sum = sum(s["brag"] for s in stance_labels.values())
    insult_sum = sum(s["insult"] for s in stance_labels.values())
    threat_sum = sum(s["threat"] for s in stance_labels.values())
    stance_strength = max(brag_sum, insult_sum, threat_sum) / max(1, len(lines))

    cohesion_metrics = {
        "entity_density": round(entity_density, 4),
        "frame_consistency": round(frame_consistency, 4),
        "stance_strength": round(stance_strength, 4),
        "num_entity_chains": len(entity_chains),
    }

    # Optional: use embeddings for stance refinement
    if use_embeddings:
        model = _load_embed_model()
        if model is not None:
            # Could add embedding-based stance similarity to a reference
            # For now we rely on heuristics
            pass

    return {
        "cohesion_metrics": cohesion_metrics,
        "entity_chains": [
            {"entities": c.entities, "category": c.category, "line_indices": c.line_indices}
            for c in entity_chains
        ],
        "stance_labels": {str(k): v for k, v in stance_labels.items()},
    }
