from evo_rhyme.metric_benchmark.pairwise_eval import (
    directional_aux_correct,
    full_matrix_score,
    predict_winner,
    summarize_pairwise,
)


def test_predict_winner_epsilon():
    assert predict_winner(0.6, 0.5, epsilon=0.05) == "a"
    assert predict_winner(0.5, 0.6, epsilon=0.05) == "b"
    assert predict_winner(0.51, 0.5, epsilon=0.05) == "tie"


def test_full_matrix():
    assert full_matrix_score("tie", predict_winner(0.5, 0.5, 0.05)) == 1.0
    assert full_matrix_score("a", predict_winner(0.9, 0.1, 0.05)) == 1.0
    assert full_matrix_score("a", predict_winner(0.1, 0.9, 0.05)) == 0.0


def test_directional_aux():
    assert directional_aux_correct("a", "a") is True
    assert directional_aux_correct("a", "tie") is False
    assert directional_aux_correct("tie", "a") is None


def test_summarize_pairwise():
    j = [("a", 0.8, 0.2), ("b", 0.2, 0.9), ("tie", 0.5, 0.5)]
    s = summarize_pairwise(j, epsilon=0.05)
    assert s.full_matrix_n == 3
    assert s.directional_n == 2
