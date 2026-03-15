from evo_rhyme.style_genome import (
    random_style_genome,
    mutate_style_genome,
    crossover_style_genome,
)
from evo_rhyme.prompt_genome import (
    random_prompt_genome,
    mutate_prompt_genome,
    crossover_prompt_genome,
)
from evo_rhyme.scoring.rhyme_graph_network import score_rhyme_graph_metrics
from evo_rhyme.archive import compact_style_dimensions, ultra_compact_dimensions
from evo_rhyme.lm_proposer import BarRequest, _build_system_prompt


def test_style_genome_mutation_and_crossover():
    a = random_style_genome()
    b = random_style_genome()
    c = crossover_style_genome(a, b)
    m = mutate_style_genome(c, mutation_rate=1.0)
    labels = m.to_labels()
    assert "tone" in labels
    assert "rhyme_scheme" in labels


def test_prompt_genome_mutation_bounds():
    g = random_prompt_genome()
    m = mutate_prompt_genome(g, mutation_scale=0.5)
    d = m.to_dict()
    for v in d.values():
        assert 0.0 <= v <= 1.0


def test_rhyme_graph_metrics_shape():
    lines = [
        "mask tight when the pressure press the breath in my chest",
        "every step through the stress is a test i address",
        "under stress i compress every mess i suppress",
        "when they press for the rest i finesse and progress",
    ]
    metrics = score_rhyme_graph_metrics(lines)
    assert set(metrics.keys()) == {
        "rhyme_graph_density",
        "rhyme_graph_cluster_coeff",
        "rhyme_graph_chain_length",
    }
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_compact_archive_dimensions_shape():
    dims = compact_style_dimensions()
    assert len(dims) == 6
    names = [d.name for d in dims]
    assert names == [
        "rhyme_density",
        "chain_length",
        "style_tone",
        "style_narrativity",
        "metaphor_density",
        "syllable_tightness",
    ]
    total = 1
    for d in dims:
        total *= d.bins
    assert total == 2700


def test_ultra_compact_archive_dimensions_shape():
    dims = ultra_compact_dimensions()
    assert len(dims) == 6
    total = 1
    for d in dims:
        total *= d.bins
    assert total == 729


def test_lm_proposer_style_directives_in_prompt():
    req = BarRequest(
        theme_keywords=["pressure", "mask"],
        role="introspection",
        style_directives={
            "tone": "aggressive",
            "metaphor_density": "rich",
            "narrativity": "high",
        },
    )
    prompt = _build_system_prompt(8, req)
    assert "Follow these style directives" in prompt
    assert "tone: aggressive" in prompt
    assert "metaphor_density: rich" in prompt
    assert "narrativity: high" in prompt

