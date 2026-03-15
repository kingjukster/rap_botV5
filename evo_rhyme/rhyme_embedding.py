"""
Lightweight rhyme embedding space for smooth rhyme-neighborhood mutations.

This module is intentionally runtime-efficient:
- no heavy ML framework dependency
- deterministic vectors from rhyme tails + small lexical perturbation
- optional nearest-neighbor lookup for rhyme-aware mutation and graph edges
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evo_rhyme.phonetics import extract_rhyme_tail


def _stable_unit_vector(key: str, dim: int) -> List[float]:
    vals: List[float] = []
    for i in range(dim):
        digest = hashlib.sha256(f"{key}:{i}".encode("utf-8")).hexdigest()
        v = (int(digest[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        vals.append(v)
    norm = math.sqrt(sum(x * x for x in vals)) or 1.0
    return [x / norm for x in vals]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class RhymeEmbeddingConfig:
    dim: int = 64
    lexical_noise_scale: float = 0.08


class RhymeEmbeddingSpace:
    def __init__(self, config: Optional[RhymeEmbeddingConfig] = None) -> None:
        self.config = config or RhymeEmbeddingConfig()
        self.word_vectors: Dict[str, List[float]] = {}
        self._built = False

    def build_from_tail_index(self, tail_to_words: Dict[str, List[str]]) -> None:
        dim = self.config.dim
        scale = self.config.lexical_noise_scale

        for tail, words in tail_to_words.items():
            tail_key = str(tail)
            tail_vec = _stable_unit_vector(f"tail:{tail_key}", dim)
            for w in words:
                word = str(w).lower().strip()
                if not word:
                    continue
                lex = _stable_unit_vector(f"word:{word}", dim)
                vec = [tail_vec[i] + scale * lex[i] for i in range(dim)]
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                self.word_vectors[word] = [x / norm for x in vec]

        self._built = True

    def has_word(self, word: str) -> bool:
        return word.lower() in self.word_vectors

    def vector(self, word: str) -> Optional[List[float]]:
        return self.word_vectors.get(word.lower())

    def nearest_neighbors(
        self,
        word_or_vec: str | List[float],
        k: int = 12,
        min_cosine: float = 0.55,
    ) -> List[Tuple[str, float]]:
        if not self._built or not self.word_vectors:
            return []

        if isinstance(word_or_vec, str):
            query = self.vector(word_or_vec)
            if query is None:
                return []
        else:
            query = word_or_vec

        scored: List[Tuple[str, float]] = []
        for w, vec in self.word_vectors.items():
            c = _cosine(query, vec)
            if c >= min_cosine:
                scored.append((w, c))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, k)]

    def rhyme_neighbors(
        self,
        word: str,
        k: int = 12,
        min_cosine: float = 0.55,
    ) -> List[str]:
        out = []
        for w, _c in self.nearest_neighbors(word, k=k, min_cosine=min_cosine):
            if w != word.lower():
                out.append(w)
        return out


_EMBED_SPACE: Optional[RhymeEmbeddingSpace] = None


def get_rhyme_embedding_space() -> RhymeEmbeddingSpace:
    global _EMBED_SPACE
    if _EMBED_SPACE is not None:
        return _EMBED_SPACE
    from evo_rhyme.mutation import get_tail_to_words

    space = RhymeEmbeddingSpace()
    space.build_from_tail_index(get_tail_to_words())
    _EMBED_SPACE = space
    return _EMBED_SPACE


def word_pair_rhyme_similarity(a: str, b: str) -> float:
    space = get_rhyme_embedding_space()
    va = space.vector(a)
    vb = space.vector(b)
    if va is None or vb is None:
        # fallback to phonetic overlap signal
        ta = extract_rhyme_tail(a)
        tb = extract_rhyme_tail(b)
        return 1.0 if ta and tb and ta == tb else 0.0
    return _cosine(va, vb)

