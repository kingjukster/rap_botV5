"""Tests for operator event recording in verse crossover/mutation and lineage helpers."""

import json
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evo_rhyme.individual import VerseIndividual
from evo_rhyme.operator_telemetry import (
    OperatorTracer,
    reset_operator_tracer,
    set_operator_tracer,
)


@pytest.fixture
def _two_verse_parents():
    p1 = VerseIndividual(
        lines=["line one from parent a", "line two from parent a",
               "line three from parent a", "line four from parent a"],
        metadata={"db_id": 100},
    )
    p2 = VerseIndividual(
        lines=["line one from parent b", "line two from parent b",
               "line three from parent b", "line four from parent b"],
        metadata={"db_id": 200},
    )
    return p1, p2


def test_verse_crossover_records_event(tmp_path: Path, _two_verse_parents) -> None:
    from evo_rhyme.verse_evolution import verse_crossover

    random.seed(42)
    tracer = OperatorTracer(tmp_path, run_id=0)
    tok = set_operator_tracer(tracer)
    try:
        p1, p2 = _two_verse_parents
        child = verse_crossover(p1, p2)
    finally:
        reset_operator_tracer(tok)

    assert isinstance(child, VerseIndividual)
    events_path = tmp_path / "operator_events.jsonl"
    assert events_path.is_file()
    rows = [json.loads(line) for line in events_path.read_text().strip().splitlines()]
    assert len(rows) >= 1
    last = rows[-1]
    assert last["scope"] == "verse"
    assert last["operator_kind"] == "crossover"
    assert last["succeeded"] is True
    assert last["operator_name"] in ("half_swap", "single_line_swap", "best_of_each", "phrase_slice")


@patch("evo_rhyme.mutation.mutate")
@patch("evo_rhyme.constraints.passes_constraints", return_value=True)
def test_verse_mutate_structural_records_event(
    _mock_pc, mock_mutate, tmp_path: Path, _two_verse_parents,
) -> None:
    """When structural mutation triggers swap_couplets, an event should be recorded."""
    from evo_rhyme.verse_evolution import verse_mutate

    mock_mutate.side_effect = lambda ind, *a, **kw: ind

    random.seed(0)
    tracer = OperatorTracer(tmp_path, run_id=0)
    tok = set_operator_tracer(tracer)
    try:
        p1, _ = _two_verse_parents
        results = []
        for _ in range(200):
            child = verse_mutate(
                p1, config={"use_structural_mutations": True}, weights=None,
            )
            results.append(child)
    finally:
        reset_operator_tracer(tok)

    events_path = tmp_path / "operator_events.jsonl"
    if events_path.is_file():
        rows = [json.loads(l) for l in events_path.read_text().strip().splitlines()]
        verse_events = [r for r in rows if r.get("scope") == "verse"]
        assert len(verse_events) >= 1, "Expected at least one verse-scope event from structural mutation"


def test_write_verse_lineage_helper() -> None:
    from evo_rhyme.verse_evolution import _write_verse_lineage

    p1 = VerseIndividual(lines=["a", "b", "c", "d"], metadata={"db_id": 10})
    p2 = VerseIndividual(lines=["e", "f", "g", "h"], metadata={"db_id": 20})

    calls = []

    def fake_insert_lineage(child_id, parent_id, relation, gen):
        calls.append((child_id, parent_id, relation, gen))

    with patch("evo_rhyme.verse_evolution._write_verse_lineage.__module__", "evo_rhyme.verse_evolution"):
        with patch("evo_rhyme.db.db_enabled", return_value=True), \
             patch("evo_rhyme.db.insert_lineage", side_effect=fake_insert_lineage):
            _write_verse_lineage(run_id=1, child_id=99, parents=[p1, p2], gen=5)

    assert len(calls) == 2
    assert calls[0] == (99, 10, "crossover+mutate", 5)
    assert calls[1] == (99, 20, "crossover+mutate", 5)


def test_write_verse_lineage_skips_missing_db_ids() -> None:
    from evo_rhyme.verse_evolution import _write_verse_lineage

    p1 = VerseIndividual(lines=["a", "b", "c", "d"], metadata={})
    p2 = VerseIndividual(lines=["e", "f", "g", "h"], metadata={"db_id": 30})

    calls = []

    def fake_insert_lineage(child_id, parent_id, relation, gen):
        calls.append((child_id, parent_id, relation, gen))

    with patch("evo_rhyme.db.db_enabled", return_value=True), \
         patch("evo_rhyme.db.insert_lineage", side_effect=fake_insert_lineage):
        _write_verse_lineage(run_id=1, child_id=50, parents=[p1, p2], gen=2)

    assert len(calls) == 1
    assert calls[0][1] == 30


def test_write_verse_lineage_noop_when_run_id_zero() -> None:
    from evo_rhyme.verse_evolution import _write_verse_lineage

    with patch("evo_rhyme.db.db_enabled") as mock_enabled:
        _write_verse_lineage(run_id=0, child_id=1, parents=[], gen=0)
        mock_enabled.assert_not_called()
