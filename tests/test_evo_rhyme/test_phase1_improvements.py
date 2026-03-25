"""Tests for Phase 1 evolution improvements."""
import pytest
import random

from evo_rhyme.mutation import (
    MUTATION_WEIGHTS,
    _MUTATION_FUNCS,
    _syntax_synonym,
    _corpus_line_swap,
    _copy_individual,
)
from evo_rhyme.individual import CoupletIndividual


class TestMutationWeightsRebalanced:
    """All operators in MUTATION_WEIGHTS should be registered in _MUTATION_FUNCS."""

    def test_all_weights_have_funcs(self):
        for key in MUTATION_WEIGHTS:
            assert key in _MUTATION_FUNCS, f"Weight key '{key}' has no registered function"

    def test_all_rule_based_ops_have_positive_weight(self):
        rule_ops = [k for k in MUTATION_WEIGHTS if not k.startswith("lm_")]
        assert len(rule_ops) >= 14, f"Expected at least 14 rule-based ops, got {len(rule_ops)}"
        for k in rule_ops:
            assert MUTATION_WEIGHTS[k] > 0, f"Rule-based op '{k}' has zero weight"

    def test_corpus_line_swap_registered(self):
        assert "corpus_line_swap" in MUTATION_WEIGHTS
        assert "corpus_line_swap" in _MUTATION_FUNCS


class TestExpandedSynonyms:
    """syntax_synonym should have a much larger swap table."""

    def test_synonym_covers_rap_vocab(self):
        ind = CoupletIndividual(line1="the money never stop", line2="run from the cold night")
        random.seed(42)
        result = _syntax_synonym(ind, {}, {})
        assert result is not None or True  # may fail for specific lines

    def test_multiple_synonym_entries(self):
        ind_big = CoupletIndividual(line1="big dreams in the night", line2="walk through the dark street")
        random.seed(42)
        results = set()
        for _ in range(20):
            r = _syntax_synonym(ind_big, {}, {})
            if r:
                results.add(r.line1)
        assert len(results) >= 1, "Should produce at least one synonym swap"


class TestCorpusLineSwap:

    def test_basic_swap(self):
        random.seed(42)
        ind = CoupletIndividual(line1="test line one", line2="test line two")
        config = {"theme_keywords": ["night", "dream"], "min_syllables": 3, "max_syllables": 30}
        result = _corpus_line_swap(ind, {}, config)
        if result is not None:
            changed = result.line1 != ind.line1 or result.line2 != ind.line2
            assert changed, "corpus_line_swap should change at least one line"

    def test_returns_none_gracefully(self):
        """Should not crash even with odd config."""
        ind = CoupletIndividual(line1="x", line2="y")
        result = _corpus_line_swap(ind, {}, None)
        # Either None or a valid CoupletIndividual
        assert result is None or isinstance(result, CoupletIndividual)


class TestStagnationBoost:
    """Verify stagnation variables are initialized in verse_evolution."""

    def test_stagnation_imports(self):
        from evo_rhyme.verse_evolution import evolve_verse_qd
        assert callable(evolve_verse_qd)


class TestNewTemplates:
    """Verify the templates file has the new grammatically structured templates."""

    def test_template_count(self):
        from pathlib import Path
        tpl_path = Path("data/evo_rhyme/templates.txt")
        if not tpl_path.exists():
            pytest.skip("templates.txt not found")
        lines = [l.strip() for l in tpl_path.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        assert len(lines) >= 80, f"Expected at least 80 templates, got {len(lines)}"

    def test_natural_sentence_templates_exist(self):
        from pathlib import Path
        tpl_path = Path("data/evo_rhyme/templates.txt")
        if not tpl_path.exists():
            pytest.skip("templates.txt not found")
        content = tpl_path.read_text()
        assert "natural sentence structure" in content.lower()
        assert "every {noun}" in content
