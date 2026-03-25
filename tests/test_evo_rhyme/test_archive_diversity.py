"""MAP-Elites occupancy diversity and novelty tie-break."""

from evo_rhyme.archive import ArchiveDimension, MAPElitesArchive, create_verse_archive
from evo_rhyme.individual import VerseIndividual


def _dummy_dim() -> ArchiveDimension:
    def ex(ind: VerseIndividual) -> int:
        return 0

    return ArchiveDimension(name="x", bins=3, bin_labels=["a", "b", "c"], extractor=ex)


def test_occupancy_diversity_stats_empty():
    arch = MAPElitesArchive([_dummy_dim()])
    s = arch.occupancy_diversity_stats()
    assert s["occupied"] == 0
    assert s["mean_normalized_entropy"] == 0.0


def test_novelty_tiebreak_prefers_higher_novelty():
    arch = create_verse_archive([_dummy_dim()], novelty_tiebreak=True)
    a = VerseIndividual(
        lines=["one two three four", "five six seven eight", "nine ten eleven twelve", "a b c d"],
        features=None,
        scores={"novelty": 0.3},
        fitness=0.5,
        metadata={},
    )
    b = VerseIndividual(
        lines=["w x y z", "w x y z", "w x y z", "w x y z"],
        features=None,
        scores={"novelty": 0.8},
        fitness=0.5,
        metadata={},
    )
    assert arch.add(a)
    assert arch.add(b)
    occ = list(arch.best_per_niche().values())[0]
    assert occ.scores["novelty"] == 0.8
