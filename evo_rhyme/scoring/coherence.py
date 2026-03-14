"""Cross-bar coherence scoring using sentence embeddings."""

from __future__ import annotations

import logging
import math
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


def _consecutive_similarity(embeddings: np.ndarray) -> float:
    """Mean cosine similarity between consecutive line pairs."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    sims: List[float] = []
    for i in range(embeddings.shape[0] - 1):
        sims.append(_cosine_similarity(embeddings[i], embeddings[i + 1]))
    raw = float(np.mean(sims))
    return max(0.0, min(1.0, raw))


def _information_gain(embeddings: np.ndarray) -> float:
    """Score how much new info each line adds via cumulative centroid drift.

    For each line beyond the first, measures cosine distance from the line's
    embedding to the centroid of all preceding lines. Higher = more new info.
    Normalized via sigmoid centered at 0.3 distance.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    gains: List[float] = []
    for i in range(1, embeddings.shape[0]):
        centroid = embeddings[:i].mean(axis=0)
        dist = 1.0 - _cosine_similarity(embeddings[i], centroid)
        gains.append(dist)
    if not gains:
        return 0.0
    avg_dist = float(np.mean(gains))
    scaled = 1.0 / (1.0 + math.exp(-10.0 * (avg_dist - 0.3)))
    return max(0.0, min(1.0, scaled))


def _structural_coherence(embeddings: np.ndarray) -> float:
    """Score setup/payoff structure between first and second half of verse.

    Ideal: moderate embedding distance between (avg lines 0-1) and (avg lines 2-3).
    Too close = repetitive, too far = disconnected.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 4:
        return 0.5
    setup = embeddings[:2].mean(axis=0)
    payoff = embeddings[2:4].mean(axis=0)
    dist = 1.0 - _cosine_similarity(setup, payoff)
    ideal = 0.25
    sigma = 0.15
    score = math.exp(-((dist - ideal) ** 2) / (2 * sigma ** 2))
    return max(0.0, min(1.0, score))


def _structural_coherence_n(embeddings: np.ndarray, block_size: int = 4) -> float:
    """Score setup/payoff structure for verses of any length.

    Splits embeddings into blocks of block_size, computes centroid for
    first half and second half of blocks, then measures their distance.
    Ideal: moderate distance (~0.25) between setup and payoff halves.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < block_size:
        return 0.5
    num_blocks = embeddings.shape[0] // block_size
    if num_blocks < 2:
        return _structural_coherence(embeddings)

    mid = num_blocks // 2
    setup_lines = embeddings[:mid * block_size]
    payoff_lines = embeddings[mid * block_size:]

    setup_centroid = setup_lines.mean(axis=0)
    payoff_centroid = payoff_lines.mean(axis=0)

    dist = 1.0 - _cosine_similarity(setup_centroid, payoff_centroid)
    ideal = 0.25
    sigma = 0.15
    score = math.exp(-((dist - ideal) ** 2) / (2 * sigma ** 2))
    return max(0.0, min(1.0, score))


def score_coherence_with_embeddings(embeddings: np.ndarray) -> float:
    """
    Score cross-bar coherence from pre-computed embeddings.

    Blends three components:
      - consecutive similarity (0.5 weight)
      - information gain (0.3 weight)
      - structural coherence (0.2 weight)

    Args:
        embeddings: (N, D) array where each row is a line embedding.

    Returns:
        Blended coherence score clamped to [0.0, 1.0].
        Returns 0.0 for fewer than 2 rows.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    consecutive = _consecutive_similarity(embeddings)
    info_gain = _information_gain(embeddings)
    structural = _structural_coherence(embeddings)
    blended = 0.5 * consecutive + 0.3 * info_gain + 0.2 * structural
    return max(0.0, min(1.0, blended))


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
