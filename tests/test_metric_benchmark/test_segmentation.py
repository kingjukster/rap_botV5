from evo_rhyme.metric_benchmark.segmentation import merge_short_stanzas, segment_song_to_verses
from evo_rhyme.metric_benchmark.segmentation import SegmentationStats


def test_merge_short_stanzas():
    stats = SegmentationStats()
    stanzas = [["a"], ["b", "c"]]
    out = merge_short_stanzas(stanzas, min_lines=2, stats=stats)
    assert len(out) >= 1
    assert sum(len(s) for s in out) >= 2


def test_segment_song_blank_lines():
    text = "first bar here\nsecond bar\n\nthird bar\nfourth bar\n"
    verses, stats = segment_song_to_verses(text, min_lines=2, max_bars=8)
    assert len(verses) >= 1
    lines, h = verses[0]
    assert len(lines) >= 2
    assert "verse_index" in h
