"""Tests for evo_rhyme.scoring.punchline. Area 12: punchline scoring (sigmoid, line weights)."""

import pytest

from evo_rhyme.scoring import punchline as punchline_mod


def test_sigmoid_scale():
    """_sigmoid_scale maps ratio to [0,1]; ratio=1 -> ~0.5."""
    sig = punchline_mod._sigmoid_scale(1.0)
    assert 0.4 <= sig <= 0.6
    assert punchline_mod._sigmoid_scale(0.5) < sig
    assert punchline_mod._sigmoid_scale(2.0) > sig
    assert 0.0 <= punchline_mod._sigmoid_scale(0.0) <= 1.0
    assert 0.0 <= punchline_mod._sigmoid_scale(3.0) <= 1.0


def test_get_line_weights():
    assert punchline_mod._get_line_weights(0) == []
    assert punchline_mod._get_line_weights(1) == [1.0]
    assert punchline_mod._get_line_weights(2) == [0.2, 0.8]
    assert punchline_mod._get_line_weights(4) == [0.2, 0.2, 0.2, 0.4]


def test_score_punchline_empty_lines():
    result = punchline_mod.score_punchline([])
    assert result == 0.0


def test_score_punchline_per_line_empty():
    result = punchline_mod.score_punchline_per_line([])
    assert result == []
