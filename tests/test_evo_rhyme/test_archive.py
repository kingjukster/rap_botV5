"""Tests for evo_rhyme.archive: dimension extractors, MAPElitesArchive, compact/ultra_compact.
Area 5: default_verse_dimensions, sample_empty_niches, top_k, best_per_niche, extractors.
"""

from typing import Any, Dict, List

import pytest

from evo_rhyme.archive import (
    MAPElitesArchive,
    ArchiveDimension,
    compact_style_dimensions,
    create_verse_archive,
    default_verse_dimensions,
    style_chain_dimensions,
    ultra_compact_dimensions,
)
from evo_rhyme.individual import VerseIndividual


def _make_verse(
    lines: List[str],
    scores: Dict[str, float] | None = None,
    fitness: float | None = None,
    metadata: Dict[str, Any] | None = None,
) -> VerseIndividual:
    v = VerseIndividual(lines=lines)
    v.scores = scores or {}
    v.fitness = fitness
    v.metadata = metadata or {}
    return v


def test_archive_dimension():
    def extractor(ind: VerseIndividual) -> int:
        return 1
    dim = ArchiveDimension(name="test", bins=3, bin_labels=["a", "b", "c"], extractor=extractor)
    assert dim.name == "test"
    assert dim.bins == 3
    assert dim.extractor(_make_verse(["x"])) == 1


def test_extract_rhyme_density_via_compact():
    dims = compact_style_dimensions()
    rhyme_dim = next(d for d in dims if d.name == "rhyme_density")
    low = _make_verse(["a", "b", "c", "d"], scores={"internal_rhyme": 0.05})
    high = _make_verse(["a", "b", "c", "d"], scores={"internal_rhyme": 0.5})
    assert rhyme_dim.extractor(low) == 0
    assert rhyme_dim.extractor(high) == 4


def test_extract_style_tone_via_metadata():
    dims = compact_style_dimensions()
    tone_dim = next(d for d in dims if d.name == "style_tone")
    ind = _make_verse(["line"], metadata={"style_genome_labels": {"tone": "aggressive"}})
    assert tone_dim.extractor(ind) == 3
    ind2 = _make_verse(["line"], metadata={"style_genome_labels": {"tone": "chaotic"}})
    assert tone_dim.extractor(ind2) == 4


def test_extract_style_narrativity():
    dims = compact_style_dimensions()
    narr_dim = next(d for d in dims if d.name == "style_narrativity")
    ind = _make_verse(["line"], metadata={"style_genome_labels": {"narrativity": "high"}})
    assert narr_dim.extractor(ind) == 2


def test_map_elites_archive_add_and_coverage():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    assert archive.total_niches() > 0
    assert archive.coverage() == 0.0
    ind = _make_verse(["first bar here", "second bar there", "third line", "fourth line"], fitness=0.8)
    added = archive.add(ind)
    assert added is True
    assert archive.occupied_niches() == 1
    assert archive.coverage() == 1.0 / archive.total_niches()


def test_map_elites_archive_add_batch():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    individuals = [
        _make_verse(["a", "b", "c", "d"], fitness=0.5),
        _make_verse(["e", "f", "g", "h"], fitness=0.6),
    ]
    n = archive.add_batch(individuals)
    assert n >= 1
    assert archive.occupied_niches() >= 1


def test_map_elites_archive_fitness_replacement():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    ind1 = _make_verse(["x", "y", "z", "w"], scores={"internal_rhyme": 0.1}, fitness=0.5)
    ind2 = _make_verse(["x", "y", "z", "w"], scores={"internal_rhyme": 0.1}, fitness=0.9)
    archive.add(ind1)
    added = archive.add(ind2)
    assert added is True
    best = archive.top_k(1)
    assert len(best) == 1
    assert best[0].fitness == 0.9


def test_map_elites_archive_sample_parents():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    assert archive.sample_parents(3) == []
    archive.add(_make_verse(["a", "b", "c", "d"], fitness=0.7))
    parents = archive.sample_parents(2)
    assert len(parents) == 2


def test_map_elites_archive_empty_niches():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    empty = archive.empty_niches()
    assert len(empty) == archive.total_niches()
    archive.add(_make_verse(["a", "b", "c", "d"], fitness=0.5))
    empty_after = archive.empty_niches()
    assert len(empty_after) == archive.total_niches() - 1


def test_compact_style_dimensions_count():
    dims = compact_style_dimensions()
    assert len(dims) == 6


def test_style_chain_dimensions_includes_chain_and_style():
    dims = style_chain_dimensions()
    names = {d.name for d in dims}
    assert "chain_length" in names
    assert "style_tone" in names
    assert len(dims) > len(default_verse_dimensions())


def test_create_verse_archive_default():
    archive = create_verse_archive()
    assert isinstance(archive, MAPElitesArchive)
    assert archive.total_niches() > 0


def test_create_verse_archive_custom_dims():
    dims = ultra_compact_dimensions()
    archive = create_verse_archive(dimensions=dims)
    assert isinstance(archive, MAPElitesArchive)
    assert len(archive.dimensions) == len(dims)


def test_ultra_compact_dimensions_count():
    dims = ultra_compact_dimensions()
    assert len(dims) == 6


# ---------------------------------------------------------------------------
# Area 5: default_verse_dimensions, sample_empty_niches, top_k, best_per_niche
# ---------------------------------------------------------------------------


def test_default_verse_dimensions_count():
    dims = default_verse_dimensions()
    assert len(dims) == 7
    names = {d.name for d in dims}
    assert "rhyme_density" in names
    assert "intensity" in names
    assert "syllable_tightness" in names
    assert "theme_balance" in names
    assert "sentiment_polarity" in names


def test_default_verse_dimensions_intensity_extractor():
    dims = default_verse_dimensions()
    intensity_dim = next(d for d in dims if d.name == "intensity")
    calm = _make_verse(["the cat sat on the mat", "we had a little chat", "so soft and flat", "no fight no spat"])
    aggressive = _make_verse(["kill the enemy", "blood and war", "strike the beast", "rage and fire"])
    assert 0 <= intensity_dim.extractor(calm) <= 4
    assert intensity_dim.extractor(aggressive) >= 3


def test_default_verse_dimensions_sentiment_extractor():
    dims = default_verse_dimensions()
    sent_dim = next(d for d in dims if d.name == "sentiment_polarity")
    dark = _make_verse(["pain death blood", "suffer shadow grave", "lost fear cry", "hell burn alone"])
    hopeful = _make_verse(["light hope rise", "dream shine fly", "free love win", "peace faith heal"])
    assert sent_dim.extractor(dark) == 0
    assert sent_dim.extractor(hopeful) == 2


def test_default_verse_dimensions_theme_balance_extractor():
    dims = default_verse_dimensions()
    theme_dim = next(d for d in dims if d.name == "theme_balance")
    low_semantic = _make_verse(["a", "b", "c", "d"], scores={"semantic": 0.1})
    high_semantic = _make_verse(["a", "b", "c", "d"], scores={"semantic": 0.8})
    assert theme_dim.extractor(low_semantic) == 0
    assert theme_dim.extractor(high_semantic) == 1


def test_map_elites_archive_sample_empty_niches():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    empty = archive.sample_empty_niches(3)
    assert len(empty) <= 3
    assert all(isinstance(c, tuple) and len(c) == len(dims) for c in empty)
    for c in empty:
        assert c not in archive.best_per_niche()


def test_map_elites_archive_sample_empty_niches_zero_returns_empty():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    assert archive.sample_empty_niches(0) == []


def test_map_elites_archive_top_k():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    archive.add(_make_verse(["a", "b", "c", "d"], fitness=0.5))
    archive.add(_make_verse(["e", "f", "g", "h"], fitness=0.9))
    archive.add(_make_verse(["i", "j", "k", "l"], fitness=0.7))
    top = archive.top_k(5)
    assert len(top) >= 1
    assert top[0].fitness == max(ind.fitness for ind in top)


def test_map_elites_archive_best_per_niche():
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    ind = _make_verse(["a", "b", "c", "d"], fitness=0.8)
    archive.add(ind)
    best = archive.best_per_niche()
    assert isinstance(best, dict)
    assert len(best) == 1
    coord = next(iter(best))
    assert best[coord].fitness == 0.8


def test_map_elites_archive_to_json_and_from_json():
    """Serialization: to_json produces list of entries; from_json restores archive."""
    dims = ultra_compact_dimensions()
    archive = MAPElitesArchive(dimensions=dims)
    ind = _make_verse(
        ["first bar here", "second bar there", "third line", "fourth line"],
        fitness=0.75,
        scores={"internal_rhyme": 0.3},
    )
    archive.add(ind)
    data = archive.to_json()
    assert isinstance(data, list)
    assert len(data) >= 1
    entry = data[0]
    assert "niche" in entry
    assert "lines" in entry
    assert entry.get("fitness") == 0.75

    restored = MAPElitesArchive.from_json(data, dims)
    assert restored.occupied_niches() == archive.occupied_niches()
    top_orig = archive.top_k(1)
    top_restored = restored.top_k(1)
    assert len(top_orig) == len(top_restored)
    if top_orig and top_restored:
        assert top_restored[0].fitness == top_orig[0].fitness
        assert top_restored[0].lines == top_orig[0].lines
