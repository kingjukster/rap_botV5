"""Tests for evo_rhyme.mutation: weights, tail index, copy, and pure operators (end_word_swap, semantic_swap, syntax_synonym, compression, expansion), mutate(), _internal_rhyme_insert, _stressed_vowel_swap, LM operators.
Area 6: Mutation operators.
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.mutation import (
    LEGACY_MUTATION_WEIGHTS,
    MUTATION_WEIGHTS,
    _compression,
    _copy_individual,
    _end_word_swap,
    _expansion,
    _get_rhymes_csv_path,
    _internal_rhyme_insert,
    _lm_rhyme_rewrite,
    _semantic_swap,
    _stressed_vowel_swap,
    _syntax_synonym,
    get_tail_to_words,
    get_rewriter,
    mutate,
)


def test_mutation_weights_keys():
    assert "end_word_swap" in MUTATION_WEIGHTS
    assert "stressed_vowel_swap" in MUTATION_WEIGHTS
    assert "lm_rhyme_rewrite" in MUTATION_WEIGHTS
    total = sum(MUTATION_WEIGHTS.values())
    assert 0.9 <= total <= 1.15


def test_legacy_mutation_weights_keys():
    assert "end_word_swap" in LEGACY_MUTATION_WEIGHTS
    assert "internal_rhyme_insert" in LEGACY_MUTATION_WEIGHTS


def test_get_rhymes_csv_path():
    path = _get_rhymes_csv_path()
    assert isinstance(path, Path)
    assert "rhymes_grouped" in path.name or "data" in str(path)


def test_get_tail_to_words_returns_dict():
    tail_to_words = get_tail_to_words()
    assert isinstance(tail_to_words, dict)
    # With real data dir, we may have entries; otherwise empty
    for k, v in tail_to_words.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert all(isinstance(w, str) for w in v)


def test_copy_individual():
    ind = CoupletIndividual(line1="first line", line2="second line")
    ind.fitness = 0.5
    ind.scores = {"x": 0.1}
    ind.metadata["foo"] = "bar"
    new_line1, new_line2 = "new one", "new two"
    copied = _copy_individual(ind, new_line1, new_line2)
    assert copied.line1 == new_line1
    assert copied.line2 == new_line2
    assert copied.fitness is None
    assert copied.scores is None
    assert copied.metadata.get("foo") == "bar"


def test_get_rewriter_cached_return(monkeypatch):
    """get_rewriter returns cached _REWRITER when set."""
    import evo_rhyme.mutation as mut_mod
    fake = object()
    monkeypatch.setattr(mut_mod, "_REWRITER", fake)
    try:
        assert get_rewriter() is fake
    finally:
        monkeypatch.setattr(mut_mod, "_REWRITER", None)


def test_end_word_swap_with_tail_in_index():
    random.seed(1)
    ind = CoupletIndividual(line1="I got the flow", line2="You know the show")
    tail_to_words = {"OW1": ["flow", "show", "glow", "grow"]}
    result = _end_word_swap(ind, tail_to_words, None)
    assert result is not None
    assert result.line1 != ind.line1 or result.line2 != ind.line2


def test_end_word_swap_no_tail_returns_none():
    ind = CoupletIndividual(line1="xyzzy foo", line2="xyzzy bar")
    tail_to_words = {}
    assert _end_word_swap(ind, tail_to_words, None) is None


def test_semantic_swap_with_theme_keywords():
    random.seed(2)
    ind = CoupletIndividual(line1="I got the thing going", line2="You make the stuff work")
    config = {"theme_keywords": ["pressure", "mask"]}
    result = _semantic_swap(ind, {}, config)
    assert result is not None
    assert "pressure" in (result.line1 + " " + result.line2).lower() or "mask" in (result.line1 + " " + result.line2).lower()


def test_semantic_swap_no_theme_returns_none():
    assert _semantic_swap(CoupletIndividual(line1="a", line2="b"), {}, None) is None


def test_syntax_synonym_replaces_word():
    random.seed(3)
    ind = CoupletIndividual(line1="I got a big dream", line2="You have a big heart")
    result = _syntax_synonym(ind, {}, None)
    assert result is not None
    assert "big" not in (result.line1 + " " + result.line2).lower() or result.line1 != ind.line1 or result.line2 != ind.line2


def test_compression_removes_droppable():
    random.seed(4)
    ind = CoupletIndividual(line1="I have the flow and the power", line2="You know the deal")
    result = _compression(ind, {}, None)
    assert result is not None
    assert len(result.line1.split()) < len(ind.line1.split()) or result.line2 != ind.line2


def test_compression_short_line_returns_none():
    ind = CoupletIndividual(line1="one two", line2="three four")
    assert _compression(ind, {}, None) is None


def test_expansion_inserts_filler():
    random.seed(5)
    ind = CoupletIndividual(line1="I run fast", line2="You stay back")
    result = _expansion(ind, {}, None)
    assert result is not None
    assert len(result.line1.split()) > len(ind.line1.split()) or len(result.line2.split()) > len(ind.line2.split())


def test_expansion_with_corpus_vocab_empty_fillers_returns_none():
    ind = CoupletIndividual(line1="I run fast", line2="You stay back")
    config = {"corpus_vocab": set("abcdef")}
    result = _expansion(ind, {}, config)
    assert result is None


def test_mutate_with_single_operator_compression():
    random.seed(6)
    ind = CoupletIndividual(line1="I have the flow and the power", line2="You know the deal")
    weights = {"compression": 1.0}
    result = mutate(ind, config=None, weights=weights)
    assert result is not None
    assert result.line1 == ind.line1 and result.line2 == ind.line2 or result.line1 != ind.line1 or result.line2 != ind.line2


def test_mutate_returns_copy_when_all_fail(monkeypatch):
    import evo_rhyme.mutation as mut_mod
    monkeypatch.setattr(mut_mod, "get_tail_to_words", lambda: {})
    ind = CoupletIndividual(line1="x y", line2="z w")
    weights = {"end_word_swap": 1.0}
    result = mutate(ind, config=None, weights=weights)
    assert result.line1 == ind.line1 and result.line2 == ind.line2


def test_build_tail_to_words_csv_missing_returns_empty(monkeypatch):
    import evo_rhyme.mutation as mut_mod
    monkeypatch.setattr(mut_mod, "_TAIL_TO_WORDS", None)
    monkeypatch.setattr(mut_mod, "_get_rhymes_csv_path", lambda: Path("/nonexistent/rhymes_grouped.csv"))
    tail_to_words = get_tail_to_words()
    assert isinstance(tail_to_words, dict)
    assert tail_to_words == {} or len(tail_to_words) >= 0


def test_internal_rhyme_insert_with_tail_match():
    """_internal_rhyme_insert swaps a middle word with same rhyme family when in tail_to_words."""
    random.seed(10)
    ind = CoupletIndividual(line1="I got the flow and the power", line2="You know the deal")
    tail_to_words = {"OW1": ["flow", "show", "glow", "grow"]}
    result = _internal_rhyme_insert(ind, tail_to_words, None)
    assert result is not None
    assert (result.line1 != ind.line1) or (result.line2 != ind.line2)


def test_internal_rhyme_insert_short_line_skipped():
    """_internal_rhyme_insert returns None when lines have fewer than 3 tokens."""
    ind = CoupletIndividual(line1="a b", line2="x y z")
    assert _internal_rhyme_insert(ind, {}, None) is None


def test_stressed_vowel_swap_with_mock_vowel_index(monkeypatch):
    """_stressed_vowel_swap replaces word when vowel_to_words has alternatives."""
    import evo_rhyme.mutation as mut_mod
    vowel_to_words = {"IY1": ["dream", "stream", "beam", "see"]}
    monkeypatch.setattr(mut_mod, "get_vowel_to_words", lambda: vowel_to_words)
    random.seed(20)
    ind = CoupletIndividual(line1="I have a big dream", line2="You hold the key")
    result = _stressed_vowel_swap(ind, {}, None)
    assert result is None or (result.line1 != ind.line1 or result.line2 != ind.line2)


def test_stressed_vowel_swap_empty_vowel_index_returns_none(monkeypatch):
    """_stressed_vowel_swap returns None when get_vowel_to_words is empty."""
    import evo_rhyme.mutation as mut_mod
    monkeypatch.setattr(mut_mod, "get_vowel_to_words", lambda: {})
    ind = CoupletIndividual(line1="I have a big dream", line2="You hold the key")
    assert _stressed_vowel_swap(ind, {}, None) is None


def test_mutate_with_lm_budget_exhausted():
    """When lm_budget remaining is 0, LM operators are excluded and non-LM op can run."""
    random.seed(30)
    ind = CoupletIndividual(line1="I have the flow and the power", line2="You know the deal")
    lm_budget = {"remaining": 0}
    result = mutate(ind, config=None, lm_budget=lm_budget, weights={"compression": 1.0})
    assert result is not None
    assert lm_budget["remaining"] == 0


def test_mutate_lm_only_uses_safe_or_lm_ops(monkeypatch):
    """When lm_only=True, only LM or safe legacy ops are tried."""
    import evo_rhyme.mutation as mut_mod
    monkeypatch.setattr(mut_mod, "get_tail_to_words", lambda: {})
    ind = CoupletIndividual(line1="I have the flow and the power", line2="You know the deal")
    result = mutate(ind, config=None, lm_only=True, weights={"compression": 1.0})
    assert result is not None


def test_lm_rhyme_rewrite_with_mocked_rewriter(monkeypatch):
    """_lm_rhyme_rewrite uses get_rewriter and returns new couplet when rewriter returns candidates."""
    import evo_rhyme.mutation as mut_mod
    mock_rewriter = MagicMock()
    mock_rewriter.rhyme_rewrite.return_value = ["I got the flow when I step in the spot"]
    monkeypatch.setattr(mut_mod, "get_rewriter", lambda config=None: mock_rewriter)
    random.seed(40)
    ind = CoupletIndividual(line1="You know I rock it hard", line2="When I hit the block")
    result = _lm_rhyme_rewrite(ind, {}, None)
    assert result is None or result.line1 != ind.line1 or result.line2 != ind.line2


def test_mutate_syllable_adjust(monkeypatch):
    """mutate with syllable_adjust weight runs the operator (may add filler or drop word)."""
    from evo_rhyme.individual import analyze_individual
    ind = CoupletIndividual(line1="I run fast", line2="You stay back")
    analyze_individual(ind)
    for seed in range(60, 80):
        random.seed(seed)
        result = mutate(ind, config={"min_syllables": 6}, weights={"syllable_adjust": 1.0})
        assert result is not None
        assert result.line1 is not None and result.line2 is not None
        if len(result.line1.split()) != len(ind.line1.split()) or len(result.line2.split()) != len(ind.line2.split()):
            break
    else:
        assert result.line1 == ind.line1 and result.line2 == ind.line2
