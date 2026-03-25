"""Operator telemetry JSONL + DB mirroring tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from evo_rhyme.operator_telemetry import (
    OperatorTracer,
    get_operator_tracer,
    record_operator_event,
    reset_operator_tracer,
    set_operator_tracer,
)


def test_operator_tracer_writes_jsonl(tmp_path: Path) -> None:
    tok = set_operator_tracer(OperatorTracer(tmp_path))
    try:
        t = get_operator_tracer()
        assert t is not None
        t.record(scope="couplet", operator_name="end_word_swap", succeeded=True)
    finally:
        reset_operator_tracer(tok)

    p = tmp_path / "operator_events.jsonl"
    assert p.is_file()
    line = p.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["operator_name"] == "end_word_swap"
    assert row["succeeded"] is True


def test_tracer_gen_propagated_to_jsonl(tmp_path: Path) -> None:
    """The ``gen`` attribute should appear in each JSONL row."""
    tracer = OperatorTracer(tmp_path, run_id=0, gen=7)
    tok = set_operator_tracer(tracer)
    try:
        record_operator_event(scope="couplet", operator_name="x", succeeded=True)
    finally:
        reset_operator_tracer(tok)

    row = json.loads((tmp_path / "operator_events.jsonl").read_text().strip())
    assert row["gen"] == 7


@patch("evo_rhyme.operator_telemetry.OperatorTracer._mirror_to_db")
def test_tracer_mirrors_to_db_when_run_id_set(mock_mirror: MagicMock, tmp_path: Path) -> None:
    tracer = OperatorTracer(tmp_path, run_id=42, gen=3)
    tok = set_operator_tracer(tracer)
    try:
        record_operator_event(scope="verse", operator_name="half_swap", succeeded=True)
    finally:
        reset_operator_tracer(tok)
    mock_mirror.assert_called_once()
    row = mock_mirror.call_args[0][0]
    assert row["operator_name"] == "half_swap"
    assert row["gen"] == 3


@patch("evo_rhyme.operator_telemetry.OperatorTracer._mirror_to_db")
def test_tracer_skips_db_when_no_run_id(mock_mirror: MagicMock, tmp_path: Path) -> None:
    tracer = OperatorTracer(tmp_path, run_id=0, gen=0)
    tok = set_operator_tracer(tracer)
    try:
        record_operator_event(scope="couplet", operator_name="y", succeeded=True)
    finally:
        reset_operator_tracer(tok)
    mock_mirror.assert_not_called()


def test_record_noop_without_tracer() -> None:
    """record_operator_event is a no-op if no tracer is active."""
    tok = set_operator_tracer(None)
    try:
        record_operator_event(scope="couplet", operator_name="z", succeeded=True)
    finally:
        reset_operator_tracer(tok)
