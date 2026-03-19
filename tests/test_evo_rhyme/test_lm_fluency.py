"""Tests for evo_rhyme.lm_fluency: _ppl_to_score, LMPerplexityScorer (short/empty), get_lm_scorer. Area 9."""

import pytest

from evo_rhyme.lm_fluency import (
    PPL_CAP,
    PPL_SCALE,
    LMPerplexityScorer,
    _ppl_to_score,
    clear_lm_cache,
    get_lm_scorer,
)


def test_ppl_to_score():
    assert 0 <= _ppl_to_score(50) <= 1
    assert _ppl_to_score(0) == 1.0
    assert _ppl_to_score(1000) < 0.05
    assert _ppl_to_score(PPL_CAP) == _ppl_to_score(PPL_CAP + 100)


def test_lm_perplexity_scorer_score_line_empty_returns_half():
    scorer = LMPerplexityScorer()
    assert scorer.score_line("") == 0.5
    assert scorer.score_line("   ") == 0.5
    assert scorer.score_line("x") == 0.5


def test_lm_perplexity_scorer_score_lines_batch_empty():
    scorer = LMPerplexityScorer()
    result = scorer.score_lines_batch([])
    assert result == []


def test_lm_perplexity_scorer_score_lines_batch_short_lines():
    scorer = LMPerplexityScorer()
    result = scorer.score_lines_batch(["a", "b", "x"])
    assert len(result) == 3
    assert all(s == 0.5 for s in result)


def test_lm_perplexity_scorer_score_couplet_short():
    scorer = LMPerplexityScorer()
    s = scorer.score_couplet("x", "y")
    assert s == 0.5


def test_get_lm_scorer_returns_scorer_or_none():
    clear_lm_cache()
    scorer = get_lm_scorer()
    if scorer is not None:
        assert isinstance(scorer, LMPerplexityScorer)
    clear_lm_cache()


def test_clear_lm_cache():
    clear_lm_cache()
    get_lm_scorer()
    clear_lm_cache()
    assert True
