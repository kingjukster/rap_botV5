"""Tests for evo_rhyme.scoring.coherence. Area 12: coherence scoring with embeddings."""

import numpy as np
import pytest

from evo_rhyme.scoring.coherence import score_coherence_with_embeddings


def test_score_coherence_with_embeddings_few_rows():
    assert score_coherence_with_embeddings(np.zeros((0, 4))) == 0.0
    assert score_coherence_with_embeddings(np.zeros((1, 4))) == 0.0


def test_score_coherence_with_embeddings_identical_lines():
    """Identical embeddings -> high consecutive similarity."""
    emb = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    score = score_coherence_with_embeddings(emb)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5


def test_score_coherence_with_embeddings_orthogonal():
    """Orthogonal consecutive lines -> low consecutive similarity."""
    emb = np.array(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0]],
        dtype=np.float32,
    )
    score = score_coherence_with_embeddings(emb)
    assert 0.0 <= score <= 1.0


def test_score_coherence_with_embeddings_four_lines():
    """4 lines exercises structural coherence path."""
    emb = np.ones((4, 8), dtype=np.float32) * 0.5
    emb[0, 0] = 1.0
    score = score_coherence_with_embeddings(emb)
    assert 0.0 <= score <= 1.0


def test_score_coherence_with_embeddings_eight_lines():
    """8+ lines exercises _structural_coherence_n with multiple blocks."""
    emb = np.ones((8, 8), dtype=np.float32) * 0.5
    emb[0, 0] = 1.0
    emb[4:, 1] = 0.3
    score = score_coherence_with_embeddings(emb)
    assert 0.0 <= score <= 1.0


def test_score_coherence_zero_vectors_no_crash():
    """Zero vectors do not cause division by zero; score is in [0, 1]."""
    emb = np.zeros((3, 4), dtype=np.float32)
    score = score_coherence_with_embeddings(emb)
    assert 0.0 <= score <= 1.0
