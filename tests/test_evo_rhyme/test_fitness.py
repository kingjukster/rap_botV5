"""
Tests for evo_rhyme.fitness: score_couplet, compute_fitness, score_verse, compute_verse_fitness, score_vector, score_verses_batch.
Area 2: Fitness and aggregate scoring.
"""

import pytest

from evo_rhyme.fitness import (
    OBJECTIVE_KEYS,
    compute_fitness,
    compute_verse_fitness,
    score_couplet,
    score_vector,
    score_verse,
    score_verses_batch,
)
from evo_rhyme.individual import CoupletIndividual, VerseIndividual, analyze_individual, analyze_verse_individual


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
            "theme_word_repetition_penalty",
            "rhyme_family_repetition_penalty",
            "identical_line_penalty",
            "near_duplicate_penalty",
            "template_penalty",
            "corpus_overlap_penalty",
            "theme_penalty",
            "coherence",
            "punchline",
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

    def test_score_couplet_with_lm_fluency(self):
        """score_couplet with use_lm_fluency=True returns ngram_fluency (may blend with LM)."""
        ind = CoupletIndividual(
            line1="I got the flow when I step in the spot",
            line2="You know I rock it hard when I hit the block",
        )
        analyze_individual(ind)
        scores = score_couplet(ind, use_lm_fluency=False)
        assert "ngram_fluency" in scores
        # With LM (may fail to load if no torch/transformers)
        scores_lm = score_couplet(ind, use_lm_fluency=True)
        assert "ngram_fluency" in scores_lm
        assert 0 <= scores_lm["ngram_fluency"] <= 1

    def test_ngram_floor_rejects_nonsense(self):
        """Candidates with ngram_fluency < 0.2 get fitness 0 (blocks nonsense phrase structure)."""
        scores_high_ngram = {
            "end_rhyme": 0.9, "internal_rhyme": 0.8, "rhyme_graph": 0.5,
            "multisyllabic": 0.5, "syllable_balance": 0.9, "stress_alignment": 0.8,
            "semantic": 1.0, "fluency": 0.9, "lexical_validity": 1.0,
            "ngram_fluency": 0.5, "novelty": 0.8,
            "weak_tail_penalty": 0.0, "repetition_penalty": 0.0,
            "rhyme_family_repetition_penalty": 0.0, "identical_line_penalty": 0.0,
            "near_duplicate_penalty": 0.0, "template_penalty": 0.0,
            "corpus_overlap_penalty": 0.0, "theme_penalty": 0.0,
        }
        scores_low_ngram = {**scores_high_ngram, "ngram_fluency": 0.05}
        assert compute_fitness(scores_high_ngram, ngram_floor=0.2) > 0
        assert compute_fitness(scores_low_ngram, ngram_floor=0.2) == 0.0
        assert compute_fitness(scores_low_ngram, ngram_floor=None) > 0  # floor disabled

    def test_compute_fitness_with_individual_and_population(self):
        """compute_fitness with individual and population adds diversity/shell penalties."""
        scores = {
            "end_rhyme": 0.8, "internal_rhyme": 0.3, "rhyme_graph": 0.5,
            "multisyllabic": 0.5, "syllable_balance": 0.9, "stress_alignment": 0.7,
            "semantic": 0.5, "fluency": 0.8, "lexical_validity": 1.0,
            "ngram_fluency": 0.5, "novelty": 0.6,
            "weak_tail_penalty": 0.0, "repetition_penalty": 0.0,
            "theme_word_repetition_penalty": 0.0, "rhyme_family_repetition_penalty": 0.0,
            "identical_line_penalty": 0.0, "near_duplicate_penalty": 0.0,
            "template_penalty": 0.0, "corpus_overlap_penalty": 0.0, "theme_penalty": 0.0,
            "coherence": 0.5, "punchline": 0.4,
        }
        ind = CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns")
        pop = [ind, CoupletIndividual(line1="I got the flow", line2="You know the show")]
        f = compute_fitness(scores, individual=ind, population=pop)
        assert isinstance(f, float)
        assert f >= 0

    def test_compute_fitness_with_style_profile(self):
        """compute_fitness with style_profile and style_weight incorporates style similarity."""
        from evo_rhyme.style_profile import build_style_profile
        scores = {
            "end_rhyme": 0.8, "internal_rhyme": 0.3, "rhyme_graph": 0.5,
            "multisyllabic": 0.5, "syllable_balance": 0.9, "stress_alignment": 0.7,
            "semantic": 0.5, "fluency": 0.8, "lexical_validity": 1.0,
            "ngram_fluency": 0.5, "novelty": 0.6,
            "weak_tail_penalty": 0.0, "repetition_penalty": 0.0,
            "theme_word_repetition_penalty": 0.0, "rhyme_family_repetition_penalty": 0.0,
            "identical_line_penalty": 0.0, "near_duplicate_penalty": 0.0,
            "template_penalty": 0.0, "corpus_overlap_penalty": 0.0, "theme_penalty": 0.0,
            "coherence": 0.5, "punchline": 0.4,
        }
        ind = CoupletIndividual(line1="The burns of life", line2="We learn from pain")
        profile = build_style_profile(["The burns of life bring many concerns", "We learn from pain as the whole world turns"])
        f = compute_fitness(scores, individual=ind, style_profile=profile, style_weight=0.1)
        assert isinstance(f, float)
        assert f >= 0

    def test_score_couplet_identical_lines_penalty(self):
        """Identical line1 and line2 get identical_line_penalty 1.0."""
        ind = CoupletIndividual(line1="The same line here", line2="The same line here")
        analyze_individual(ind)
        scores = score_couplet(ind)
        assert scores["identical_line_penalty"] == 1.0

    def test_score_couplet_corpus_overlap_penalty(self):
        """corpus_lines similar to couplet lines increase corpus_overlap_penalty."""
        ind = CoupletIndividual(
            line1="I got the flow when I step in the spot",
            line2="You know I rock it hard when I hit the block",
        )
        analyze_individual(ind)
        corpus = [
            "I got the flow when I step in the spot",
            "Another line from the corpus here",
        ]
        scores = score_couplet(ind, corpus_lines=corpus)
        assert "corpus_overlap_penalty" in scores
        assert scores["corpus_overlap_penalty"] >= 0


class TestScoreVerse:
    """Test score_verse (4-line) and beat_fit integration."""

    def test_score_verse_includes_beat_fit(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        scores = score_verse(verse)
        assert "beat_fit" in scores
        assert isinstance(scores["beat_fit"], (int, float))
        assert 0 <= scores["beat_fit"] <= 1

    def test_score_verse_returns_all_component_keys(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        scores = score_verse(verse)
        expected = {
            "rhyme_scheme_score", "internal_rhyme", "syllable_balance", "fluency",
            "semantic", "lexical_validity", "coherence", "punchline", "beat_fit",
            "flow_alignment", "style_adherence", "prompt_adherence",
        }
        for k in expected:
            assert k in scores, f"missing key {k}"
            assert isinstance(scores[k], (int, float))


class TestComputeVerseFitness:
    """compute_verse_fitness and verse weight dict."""

    def test_compute_verse_fitness_default_weights(self):
        scores = {"rhyme_scheme_score": 0.8, "internal_rhyme": 0.5, "syllable_balance": 0.9}
        f = compute_verse_fitness(scores)
        assert isinstance(f, float)
        assert f >= 0

    def test_compute_verse_fitness_custom_weights(self):
        scores = {"rhyme_scheme_score": 1.0, "internal_rhyme": 0.0}
        f = compute_verse_fitness(scores, weights={"rhyme_scheme_score": 1.0, "internal_rhyme": 0.5})
        assert f >= 0

    def test_compute_verse_fitness_cliche_penalty_scaled(self):
        scores = {"cliche_penalty": 0.02, "rhyme_scheme_score": 0.5}
        f = compute_verse_fitness(scores, weights={"cliche_penalty": 0.2, "rhyme_scheme_score": 0.8})
        assert isinstance(f, float)


class TestScoreVector:
    """score_vector (extract_objectives) rhythm, originality, style_match."""

    def test_score_vector_default_keys(self):
        scores = {"end_rhyme": 0.8, "stress_alignment": 0.7, "syllable_balance": 0.9}
        vec = score_vector(scores)
        assert len(vec) == len(OBJECTIVE_KEYS)
        assert all(isinstance(v, (int, float)) and v >= 0 for v in vec)

    def test_score_vector_rhythm_synthesized(self):
        scores = {"stress_alignment": 0.6, "syllable_balance": 0.8}
        vec = score_vector(scores, keys=["rhythm"])
        assert len(vec) == 1
        assert vec[0] == 0.7

    def test_score_vector_originality_synthesized(self):
        scores = {"corpus_overlap_penalty": 0.2}
        vec = score_vector(scores, keys=["originality"])
        assert len(vec) == 1
        assert vec[0] == 0.8

    def test_score_vector_style_match_fallback(self):
        scores = {"style_adherence": 0.6}
        vec = score_vector(scores, keys=["style_match"])
        assert len(vec) == 1
        assert vec[0] == 0.6


class TestScoreVersesBatch:
    """score_verses_batch empty list and cache-miss path."""

    def test_score_verses_batch_empty_returns_empty(self):
        result = score_verses_batch([])
        assert result == []

    def test_score_verses_batch_one_verse_returns_one_dict(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        results = score_verses_batch([verse], fast_mode=True)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "rhyme_scheme_score" in results[0]

    def test_score_verses_batch_multiple_verses(self):
        """score_verses_batch with multiple verses returns one dict per verse."""
        verses = [
            VerseIndividual(
                lines=[
                    "I fight through pressure every night inside the dark",
                    "I carve a lane through the pain and leave a mark",
                    "My words are blades every phrase got a sharpened spark",
                    "I move with rage but the cadence stays precise and stark",
                ]
            ),
            VerseIndividual(
                lines=[
                    "The burns of life bring many concerns",
                    "We learn from pain as the whole world turns",
                    "I got the flow when I step in the spot",
                    "You know I rock it hard when I hit the block",
                ]
            ),
        ]
        for v in verses:
            analyze_verse_individual(v)
        results = score_verses_batch(verses, fast_mode=True)
        assert len(results) == 2
        assert all(isinstance(r, dict) and "rhyme_scheme_score" in r for r in results)
