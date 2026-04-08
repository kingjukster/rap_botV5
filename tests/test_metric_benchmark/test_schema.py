"""Schema validation for metric benchmark JSONL."""

import json

import pytest

from evo_rhyme.metric_benchmark.schema import (
    LABEL_KEYS,
    parse_pairwise_line,
    parse_verse_line,
    validate_verse_record,
)


def _full_labels():
    return {k: 0.5 for k in LABEL_KEYS}


def test_validate_verse_ok():
    obj = {
        "id": "v1",
        "lyrics": ["line one here", "line two here"],
        "labels": _full_labels(),
        "source": "corpus",
    }
    ok, errs = validate_verse_record(obj)
    assert ok, errs


def test_validate_verse_too_few_bars():
    obj = {
        "id": "v1",
        "lyrics": ["only one"],
        "labels": _full_labels(),
        "source": "corpus",
    }
    ok, errs = validate_verse_record(obj, min_bars=2)
    assert not ok


def test_parse_verse_line():
    obj = {
        "id": "v1",
        "lyrics": ["a", "b"],
        "labels": _full_labels(),
        "source": "synthetic",
    }
    rec = parse_verse_line(json.dumps(obj))
    assert rec.id == "v1"
    assert rec.labels is not None
    assert rec.labels["flow"] == 0.5


def test_parse_pairwise_line():
    obj = {
        "id": "p1",
        "verse_a_id": "a",
        "verse_b_id": "b",
        "better": "a",
        "axis": "overall",
        "pair_type": "random",
        "split": "train",
    }
    rec = parse_pairwise_line(json.dumps(obj))
    assert rec.better == "a"


def test_parse_pairwise_tie():
    obj = {
        "id": "p1",
        "verse_a_id": "a",
        "verse_b_id": "b",
        "better": "tie",
        "axis": "flow",
        "pair_type": "adversarial_flow_rhyme",
        "split": "test",
    }
    rec = parse_pairwise_line(json.dumps(obj))
    assert rec.pair_type == "adversarial_flow_rhyme"
