"""Operator telemetry JSONL."""

import json
from pathlib import Path

from evo_rhyme.operator_telemetry import (
    OperatorTracer,
    get_operator_tracer,
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
