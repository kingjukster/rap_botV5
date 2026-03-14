"""
Novelty scoring via embedding-based distance to k-nearest neighbors.

Maintains a ring buffer of embeddings from past candidates. Novelty for a
new candidate = average cosine distance to its k nearest neighbors in the
archive. Higher novelty = more different from what came before.

Reuses the sentence-transformers model from coherence scoring.
"""

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
    """Lazy-load sentence transformer model (shared with coherence scorer)."""
    global _MODEL, _MODEL_NAME
    target = model_name or _MODEL_NAME
    if _MODEL is not None and target == _MODEL_NAME:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(target)
    _MODEL_NAME = target
    logger.info("Loaded novelty model: %s", target)
    return _MODEL


def _cosine_distance_batch(query: np.ndarray, archive: np.ndarray) -> np.ndarray:
    """Cosine distance from query to each row in archive. Returns (N,) array."""
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-9:
        return np.ones(archive.shape[0])
    archive_norms = np.linalg.norm(archive, axis=1)
    archive_norms = np.maximum(archive_norms, 1e-9)
    similarities = archive @ query / (archive_norms * query_norm)
    return 1.0 - similarities


class NoveltyArchive:
    """Ring buffer of embeddings for novelty computation.
    
    Persists across generations. When full, overwrites the oldest entry.
    """

    def __init__(self, max_size: int = 5000, k_nearest: int = 10, embedding_dim: int = 384):
        self._max_size = max_size
        self._k = k_nearest
        self._buffer = np.zeros((max_size, embedding_dim), dtype=np.float32)
        self._size = 0
        self._write_idx = 0
        self._dim = embedding_dim

    def add(self, embedding: np.ndarray) -> None:
        """Add a single embedding to the archive."""
        if embedding.shape[0] != self._dim:
            if self._size == 0:
                self._dim = embedding.shape[0]
                self._buffer = np.zeros((self._max_size, self._dim), dtype=np.float32)
            else:
                return
        self._buffer[self._write_idx] = embedding
        self._write_idx = (self._write_idx + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(self, embeddings: np.ndarray) -> None:
        """Add multiple embeddings."""
        for i in range(embeddings.shape[0]):
            self.add(embeddings[i])

    def compute_novelty(self, embedding: np.ndarray) -> float:
        """Average cosine distance to k nearest neighbors in archive.
        
        Returns value in [0, 1]. Higher = more novel.
        Returns 1.0 if archive is empty (everything is novel initially).
        """
        if self._size == 0:
            return 1.0
        active = self._buffer[: self._size]
        distances = _cosine_distance_batch(embedding, active)
        k = min(self._k, self._size)
        nearest_k = np.partition(distances, k - 1)[:k]
        return float(np.mean(nearest_k))

    def compute_novelty_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute novelty for multiple embeddings. Returns (N,) array."""
        results = np.zeros(embeddings.shape[0])
        for i in range(embeddings.shape[0]):
            results[i] = self.compute_novelty(embeddings[i])
        return results

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._max_size


def embed_text(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """Encode a single text string into an embedding vector."""
    model = _get_model(model_name)
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
    """Encode multiple text strings into embeddings. Returns (N, D) array."""
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True)


def compute_verse_novelty(
    verse_text: str,
    novelty_archive: NoveltyArchive,
    model_name: Optional[str] = None,
) -> float:
    """Compute novelty score for a verse (full text)."""
    try:
        embedding = embed_text(verse_text, model_name)
        return novelty_archive.compute_novelty(embedding)
    except Exception:
        logger.warning("Verse novelty computation failed", exc_info=True)
        return 0.5


def compute_line_novelty(
    text: str,
    novelty_archive: NoveltyArchive,
    model_name: Optional[str] = None,
) -> float:
    """Compute novelty score for a single line."""
    try:
        embedding = embed_text(text, model_name)
        return novelty_archive.compute_novelty(embedding)
    except Exception:
        logger.warning("Line novelty computation failed", exc_info=True)
        return 0.5
