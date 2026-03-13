"""Cross-bar coherence scoring using sentence embeddings."""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL: Optional["SentenceTransformer"] = None
_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _get_model(model_name: Optional[str] = None) -> "SentenceTransformer":
    """Lazy-load sentence transformer model."""
    global _MODEL, _MODEL_NAME
    target = model_name or _MODEL_NAME
    if _MODEL is not None and target == _MODEL_NAME:
        return _MODEL
    from sentence_transformers import SentenceTransformer

    _MODEL = SentenceTransformer(target)
    _MODEL_NAME = target
    logger.info("Loaded coherence model: %s", target)
    return _MODEL


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_coherence_with_embeddings(embeddings: np.ndarray) -> float:
    """
    Score cross-bar coherence from pre-computed embeddings.

    Args:
        embeddings: (N, D) array where each row is a line embedding.

    Returns:
        Mean cosine similarity between consecutive pairs, clamped to [0.0, 1.0].
        Returns 0.0 for fewer than 2 rows.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    sims: List[float] = []
    for i in range(embeddings.shape[0] - 1):
        sims.append(_cosine_similarity(embeddings[i], embeddings[i + 1]))
    raw = float(np.mean(sims))
    return max(0.0, min(1.0, raw))


def score_coherence(lines: List[str], model_name: Optional[str] = None) -> float:
    """
    Score cross-bar coherence for a list of lines.

    Computes cosine similarity between each consecutive pair of lines,
    then returns the mean. Higher = more coherent verse.

    Returns 0.0 for single-line input, 0.0 on error.
    Score range: [0.0, 1.0] (clamped)
    """
    if len(lines) < 2:
        return 0.0
    try:
        model = _get_model(model_name)
        embeddings = model.encode(lines, convert_to_numpy=True)
        return score_coherence_with_embeddings(embeddings)
    except Exception:
        logger.warning("Coherence scoring failed", exc_info=True)
        return 0.0
