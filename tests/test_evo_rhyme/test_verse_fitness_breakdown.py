"""Tests for verse fitness breakdown parity."""

from evo_rhyme.fitness import VERSE_DEFAULT_WEIGHTS, compute_verse_fitness
from evo_rhyme.verse_fitness_breakdown import compute_verse_fitness_breakdown


def test_breakdown_aggregate_matches_compute_verse_fitness():
    scores = {
        "rhyme_scheme_score": 0.8,
        "internal_rhyme": 0.5,
        "rhyme_chain_density": 0.4,
        "global_rhyme_chain_score": 0.3,
        "internal_chain_score": 0.3,
        "rhyme_graph_density": 0.2,
        "rhyme_graph_cluster_coeff": 0.2,
        "rhyme_graph_chain_length": 0.2,
        "syllable_balance": 0.7,
        "fluency": 0.6,
        "lm_fluency": 0.5,
        "semantic": 0.5,
        "lexical_validity": 0.8,
        "coherence": 0.4,
        "punchline": 0.3,
        "identical_line_penalty": 0.0,
        "template_penalty": 0.0,
        "repetition_penalty": 0.0,
        "near_duplicate_penalty": 0.0,
        "filler_line_penalty": 0.0,
        "line_phrase_penalty": 0.0,
        "corpus_overlap_penalty": 0.0,
        "garbled_line_penalty": 0.0,
        "cliche_penalty": 0.01,
        "structural_repetition_penalty": 0.0,
        "cross_verse_repetition_penalty": 0.0,
        "novelty": 0.5,
        "flow_alignment": 0.5,
        "beat_fit": 0.5,
        "flow_continuity_score": 0.5,
        "style_adherence": 0.5,
        "prompt_adherence": 0.5,
    }
    w = VERSE_DEFAULT_WEIGHTS
    a = compute_verse_fitness(scores, w)
    bd = compute_verse_fitness_breakdown(scores, w)
    assert abs(bd.aggregate - a) < 1e-9
