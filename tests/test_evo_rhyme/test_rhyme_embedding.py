from evo_rhyme.rhyme_embedding import get_rhyme_embedding_space
from evo_rhyme.scoring.rhyme_graph_network import score_rhyme_graph_metrics


def test_rhyme_embedding_space_builds():
    space = get_rhyme_embedding_space()
    assert space.word_vectors
    assert space.config.dim >= 32


def test_rhyme_embedding_neighbors_for_existing_word():
    space = get_rhyme_embedding_space()
    word = next(iter(space.word_vectors.keys()))
    neighbors = space.rhyme_neighbors(word, k=8, min_cosine=0.3)
    assert isinstance(neighbors, list)
    assert all(isinstance(w, str) for w in neighbors)


def test_rhyme_graph_embedding_mode_shape():
    lines = [
        "pressure in my chest when the stress multiplies",
        "lecture from the block where the message amplifies",
        "gesture in the dark while the metal never lies",
        "venture through the storm where the measure quantifies",
    ]
    metrics = score_rhyme_graph_metrics(lines, edge_mode="embedding")
    assert set(metrics.keys()) == {
        "rhyme_graph_density",
        "rhyme_graph_cluster_coeff",
        "rhyme_graph_chain_length",
    }
    for v in metrics.values():
        assert 0.0 <= v <= 1.0

