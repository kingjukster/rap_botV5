from evo_rhyme.metric_benchmark.schema import PairwiseRecord, pairwise_crosses_split, verse_ids_by_split
from evo_rhyme.metric_benchmark.schema import VerseRecord


def test_pairwise_same_split_no_leakage():
    verses = [
        VerseRecord("a", [], None, "corpus", split="train"),
        VerseRecord("b", [], None, "corpus", split="train"),
        VerseRecord("c", [], None, "corpus", split="test"),
    ]
    by_split = verse_ids_by_split(verses)
    p_ok = PairwiseRecord("p1", "a", "b", "a", "overall", "random", "train")
    p_bad = PairwiseRecord("p2", "a", "c", "a", "overall", "random", "train")
    assert not pairwise_crosses_split(p_ok, by_split)
    assert pairwise_crosses_split(p_bad, by_split)
