"""Tests for evo_rhyme.scoring.novelty. Area 12: novelty scoring (archive and cosine distance)."""

import numpy as np
import pytest

from evo_rhyme.scoring.novelty import NoveltyArchive


def test_novelty_archive_empty_returns_one():
    archive = NoveltyArchive(max_size=100, k_nearest=5, embedding_dim=4)
    assert archive.size == 0
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert archive.compute_novelty(q) == 1.0


def test_novelty_archive_add_and_compute():
    archive = NoveltyArchive(max_size=10, k_nearest=3, embedding_dim=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    archive.add(v1)
    archive.add(v2)
    assert archive.size == 2
    # Query same as v1 -> low distance to nearest
    nov = archive.compute_novelty(v1)
    assert 0.0 <= nov <= 1.0
    # Query orthogonal -> higher distance
    v_far = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    nov_far = archive.compute_novelty(v_far)
    assert nov_far >= nov


def test_novelty_archive_add_batch():
    archive = NoveltyArchive(max_size=20, k_nearest=2, embedding_dim=3)
    batch = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
    archive.add_batch(batch)
    assert archive.size == 3
