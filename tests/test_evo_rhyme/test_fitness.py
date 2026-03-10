"""
Tests for evo_rhyme.fitness: score_couplet and compute_fitness return valid scores.
"""

import pytest

from evo_rhyme.fitness import compute_fitness, score_couplet
from evo_rhyme.individual import CoupletIndividual, analyze_individual


class TestScoreCouplet:
    """Test score_couplet returns valid component scores."""

    def test_score_couplet_returns_all_keys(self):
        ind = CoupletIndividual(
            line1="I got the flow when I step in the spot",
            line2="You know I rock it hard when I hit the block",
        )
        analyze_individual(ind)
        scores = score_couplet(ind)
        expected_keys = {
            "end_rhyme",
            "internal_rhyme",
            "rhyme_graph",
            "multisyllabic",
            "syllable_balance",
            "stress_alignment",
            "semantic",
            "fluency",
            "lexical_validity",
            "ngram_fluency",
            "novelty",
            "weak_tail_penalty",
            "repetition_penalty",
            "rhyme_family_repetition_penalty",
            "identical_line_penalty",
            "near_duplicate_penalty",
            "template_penalty",
            "corpus_overlap_penalty",
            "theme_penalty",
        }
        assert set(scores.keys()) == expected_keys

    def test_scores_in_valid_range(self):
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        scores = score_couplet(ind)
        for key, val in scores.items():
            assert isinstance(val, (int, float))
            assert 0 <= val <= 1 or (
                key in ("weak_tail_penalty", "repetition_penalty", "rhyme_family_repetition_penalty",
                        "identical_line_penalty", "near_duplicate_penalty", "template_penalty",
                        "corpus_overlap_penalty", "theme_penalty") and val >= 0
            )


class TestComputeFitness:
    """Test compute_fitness returns valid aggregate score."""

    def test_compute_fitness_returns_float(self):
        scores = {
            "end_rhyme": 0.8,
            "internal_rhyme": 0.3,
            "multisyllabic": 0.5,
            "syllable_balance": 0.9,
            "stress_alignment": 0.7,
            "semantic": 0.5,
            "fluency": 0.8,
            "novelty": 0.6,
            "weak_tail_penalty": 0.0,
            "repetition_penalty": 0.0,
            "identical_line_penalty": 0.0,
            "near_duplicate_penalty": 0.0,
            "template_penalty": 0.0,
            "corpus_overlap_penalty": 0.0,
            "theme_penalty": 0.0,
        }
        fitness = compute_fitness(scores)
        assert isinstance(fitness, float)

    def test_full_pipeline(self):
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        scores = score_couplet(ind)
        fitness = compute_fitness(scores)
        assert isinstance(fitness, float)
        assert -1.0 <= fitness <= 2.0

    def test_multisyllabic_rewards_tail_overlap(self):
        """Multisyllabic score should favor longer phoneme tail overlap."""
        # mean/scene have tail IY1 N -> overlap 2 -> score 0.4
        ind1 = CoupletIndividual(line1="Know my flow is too mean", line2="Don't want a bitch making a scene")
        analyze_individual(ind1)
        scores1 = score_couplet(ind1)
        assert "multisyllabic" in scores1
        assert scores1["multisyllabic"] >= 0.2
