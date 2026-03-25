"""Tests for Phase 2 evolution improvements."""
import random
from typing import List

from evo_rhyme.individual import CoupletIndividual, VerseIndividual


class TestRhymeSchemeAwareCrossover:
    """verse_crossover should have scheme_aware_swap in its repertoire."""

    def test_crossover_produces_valid_verse(self):
        from evo_rhyme.verse_evolution import verse_crossover

        p1 = VerseIndividual(lines=["alpha one", "alpha two", "alpha three", "alpha four"])
        p2 = VerseIndividual(lines=["beta one", "beta two", "beta three", "beta four"])
        p1.fitness = 0.5
        p2.fitness = 0.6

        random.seed(42)
        results = set()
        for _ in range(50):
            child = verse_crossover(p1, p2, {"scheme": "AABB"})
            assert len(child.lines) == 4
            results.add(tuple(child.lines))
        assert len(results) > 1, "Crossover should produce varied output"

    def test_scheme_config_accepted(self):
        from evo_rhyme.verse_evolution import verse_crossover

        p1 = VerseIndividual(lines=["a", "b", "c", "d"])
        p2 = VerseIndividual(lines=["e", "f", "g", "h"])
        p1.fitness = 0.5
        p2.fitness = 0.5
        for scheme in ("AABB", "ABAB", "ABBA", "ABCB"):
            child = verse_crossover(p1, p2, {"scheme": scheme})
            assert len(child.lines) == 4


class TestCullNearDuplicates:
    """_cull_near_duplicates should remove fingerprint-identical verses."""

    def test_removes_duplicates(self):
        from evo_rhyme.verse_evolution import _cull_near_duplicates

        v1 = VerseIndividual(lines=["hello world foo", "bar baz qux", "alpha beta gamma", "delta eps zeta"])
        v1.fitness = 0.8
        v2 = VerseIndividual(lines=["hello world foo", "bar baz qux", "alpha beta gamma", "delta eps zeta"])
        v2.fitness = 0.5
        v3 = VerseIndividual(lines=["different start here", "another line goes", "third one appears", "fourth one ends"])
        v3.fitness = 0.6

        result = _cull_near_duplicates([v1, v2, v3], max_pop=10)
        assert len(result) == 2
        assert result[0].fitness == 0.8

    def test_keeps_diverse(self):
        from evo_rhyme.verse_evolution import _cull_near_duplicates

        verses = []
        for i in range(10):
            v = VerseIndividual(lines=[f"unique_{i} word here", f"second_{i} line text",
                                       f"third_{i} verse part", f"fourth_{i} final one"])
            v.fitness = 0.1 * i
            verses.append(v)

        result = _cull_near_duplicates(verses, max_pop=10)
        assert len(result) == 10


class TestVerbConjugationCSV:
    """Vocab loader should supplement verbs from verbs.csv."""

    def test_load_vocab_includes_csv_verbs(self):
        from evo_rhyme.generator import load_vocab
        vocab = load_vocab()
        assert "awake" in vocab["verbs"]
        assert "awoke" in vocab["verbs_past"]
        assert "break" in vocab["verbs"]
        assert "broke" in vocab["verbs_past"]

    def test_no_duplicate_verbs(self):
        from evo_rhyme.generator import load_vocab
        vocab = load_vocab()
        assert len(vocab["verbs"]) == len(set(v.lower() for v in vocab["verbs"])), "Duplicate verbs found"


class TestMultiPointMutation:
    """Multi-point mutation is configured correctly."""

    def test_mutation_weights_complete(self):
        from evo_rhyme.mutation import MUTATION_WEIGHTS, _MUTATION_FUNCS
        rule_ops = [k for k in MUTATION_WEIGHTS if not k.startswith("lm_")]
        for op in rule_ops:
            assert op in _MUTATION_FUNCS, f"Rule op '{op}' missing from registry"
        assert len(rule_ops) >= 14, f"Expected 14+ rule ops, got {len(rule_ops)}"

    def test_couplet_mutate_no_lm(self):
        """Couplet mutate with lm_budget=0 should skip all lm_ ops."""
        from evo_rhyme.mutation import mutate, MUTATION_WEIGHTS
        ind = CoupletIndividual(
            line1="walking through the night with my vision intact",
            line2="shadows in the dark when the truth hit back",
        )
        random.seed(42)
        result = mutate(ind, {}, MUTATION_WEIGHTS, lm_budget={"remaining": 0})
        assert isinstance(result, CoupletIndividual)
