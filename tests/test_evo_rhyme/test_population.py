"""Tests for evo_rhyme.population and generator: RandomGenerator, TemplateGenerator, create_mixed_population, generator entry points, SeedGenerator.
Area 7: Generator and population.
"""

import tempfile
from pathlib import Path

import pytest

from evo_rhyme.generator import generate_random_couplets, template_fill_couplets
from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.population import (
    RandomGenerator,
    TemplateGenerator,
    VersePopulation,
    VerseSeedGenerator,
    create_initial_population,
    create_initial_verse_population,
    create_mixed_population,
)
from evo_rhyme.seed_generator import SeedGenerator


def test_random_generator_generate_seed_couplets():
    gen = RandomGenerator()
    pop = gen.generate_seed_couplets(theme_keywords=None, size=3)
    assert len(pop) == 3
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)


def test_template_generator_generate_seed_couplets():
    gen = TemplateGenerator()
    pop = gen.generate_seed_couplets(theme_keywords=None, size=2)
    assert len(pop) == 2
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)


def test_create_mixed_population_no_corpus():
    pop = create_mixed_population(corpus_path=None, size=5, analyze=False)
    assert len(pop) == 5
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)


def test_create_mixed_population_with_theme():
    pop = create_mixed_population(
        corpus_path=None,
        theme_keywords=["flow", "money"],
        size=4,
        analyze=False,
    )
    assert len(pop) == 4


def test_create_initial_population():
    gen = RandomGenerator()
    pop = create_initial_population(gen, theme_keywords=None, size=3)
    assert len(pop) == 3
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)


def test_create_initial_population_bad_generator_raises():
    with pytest.raises(TypeError, match="generate_seed_couplets"):
        create_initial_population(object(), size=5)


def test_generate_random_couplets_count_and_analyze():
    pop = generate_random_couplets(theme_keywords=None, count=2, analyze=False)
    assert len(pop) == 2
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)
    assert all(ind.line1 and ind.line2 for ind in pop)


def test_template_fill_couplets_with_theme():
    pop = template_fill_couplets(theme_keywords=["flow", "money"], count=2, analyze=False)
    assert len(pop) == 2
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)


def test_create_mixed_population_with_corpus_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("first line of the verse here\n")
        f.write("second line with more words\n")
        f.write("third bar for the rhyme\n")
        f.write("fourth line to complete\n")
        path = Path(f.name)
    try:
        pop = create_mixed_population(corpus_path=path, size=6, analyze=False)
        assert len(pop) == 6
        assert all(isinstance(ind, CoupletIndividual) for ind in pop)
    finally:
        path.unlink(missing_ok=True)


def test_seed_generator_with_fixture_corpus():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("line one here\n")
        f.write("line two there\n")
        f.write("line three more\n")
        path = Path(f.name)
    try:
        gen = SeedGenerator(corpus_path=path)
        couplets = gen.generate_seed_couplets(theme_keywords=None, size=2)
        assert len(couplets) >= 1
        assert all(isinstance(c, CoupletIndividual) for c in couplets)
    finally:
        path.unlink(missing_ok=True)


def test_create_mixed_population_analyze_true():
    """With analyze=True, the analyze branch is exercised (coverage)."""
    pop = create_mixed_population(corpus_path=None, size=3, analyze=True)
    assert len(pop) == 3
    assert all(ind.line1 and ind.line2 for ind in pop)


def test_create_mixed_population_fallback_when_short():
    """When a source returns fewer than requested, fallback template fill is used."""
    from unittest.mock import patch
    call_count = [0]

    def mock_template_fill(*, count, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return []  # First call returns empty to trigger fallback
        return template_fill_couplets(count=count, **kwargs)

    with patch("evo_rhyme.generator.template_fill_couplets", side_effect=mock_template_fill):
        pop = create_mixed_population(corpus_path=None, size=5, analyze=False)
    assert len(pop) == 5
    assert all(isinstance(ind, CoupletIndividual) for ind in pop)
    assert call_count[0] >= 2  # Fallback path ran


def test_verse_seed_generator_template_mode():
    gen = VerseSeedGenerator(init_mode="template")
    verses = gen.generate_seed_verses(theme_keywords=None, size=3)
    assert len(verses) == 3
    from evo_rhyme.individual import VerseIndividual
    assert all(isinstance(v, VerseIndividual) for v in verses)
    assert all(len(v.lines) == 4 for v in verses)


def test_verse_seed_generator_random_mode():
    gen = VerseSeedGenerator(init_mode="random")
    verses = gen.generate_seed_verses(theme_keywords=None, size=2)
    assert len(verses) == 2
    from evo_rhyme.individual import VerseIndividual
    assert all(isinstance(v, VerseIndividual) for v in verses)


def test_verse_seed_generator_mixed_mode():
    gen = VerseSeedGenerator(init_mode="mixed")
    verses = gen.generate_seed_verses(theme_keywords=None, size=3)
    assert len(verses) == 3
    from evo_rhyme.individual import VerseIndividual
    assert all(isinstance(v, VerseIndividual) for v in verses)


def test_verse_population_create():
    pop = VersePopulation.create(theme_keywords=None, size=4, init_mode="template")
    assert len(pop.individuals) == 4
    assert pop.size == 4


def test_create_initial_verse_population():
    verses = create_initial_verse_population(
        theme_keywords=None, size=3, init_mode="template"
    )
    assert len(verses) == 3
    from evo_rhyme.individual import VerseIndividual
    assert all(isinstance(v, VerseIndividual) for v in verses)
    assert all(len(v.lines) == 4 for v in verses)
