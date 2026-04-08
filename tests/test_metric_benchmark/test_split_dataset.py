"""Deterministic stratified split."""

from evo_rhyme.metric_benchmark.schema import LABEL_KEYS


def _labeled_row(i: int, overall: float, source: str = "corpus"):
    lab = {k: 0.3 + (i % 7) * 0.05 for k in LABEL_KEYS}
    lab["overall"] = overall
    return {
        "id": f"id_{i}",
        "lyrics": [f"line a {i}", f"line b {i}", f"line c {i}"],
        "labels": lab,
        "source": source,
        "unlabeled": False,
    }


def test_split_three_way_counts():
    from evo_rhyme.metric_benchmark.split import split_three_way

    rows = []
    for i in range(30):
        overall = 0.1 + (i % 10) * 0.09
        rows.append(_labeled_row(i, overall, source="corpus" if i % 2 == 0 else "synthetic"))

    out, extra = split_three_way(rows, seed=123)
    assert extra["splits"]["train"] + extra["splits"]["val"] + extra["splits"]["test"] == 30
    splits = [r["split"] for r in out if r.get("split")]
    assert len(splits) == 30
    assert set(splits) <= {"train", "val", "test"}


def test_split_reproducible():
    from evo_rhyme.metric_benchmark.split import split_three_way

    rows = [_labeled_row(i, 0.2 + i * 0.02) for i in range(24)]
    a, _ = split_three_way(rows, seed=999)
    b, _ = split_three_way(rows, seed=999)
    sa = sorted((r["id"], r["split"]) for r in a if r.get("split"))
    sb = sorted((r["id"], r["split"]) for r in b if r.get("split"))
    assert sa == sb
