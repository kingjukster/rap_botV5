"""Tests for label_with_ai quota / resume helpers (import script without evo_rhyme __init__)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _stub_evo_rhyme_metric_benchmark() -> None:
    """Avoid importing evo_rhyme package (pulls torch/numpy)."""
    fake = types.ModuleType("evo_rhyme")
    fake_mb = types.ModuleType("evo_rhyme.metric_benchmark")
    fake_schema = types.ModuleType("evo_rhyme.metric_benchmark.schema")
    fake_schema.LABEL_KEYS = (
        "flow",
        "rhyme",
        "semantic",
        "punchline",
        "fluency",
        "originality",
        "overall",
    )
    fake_protocol = types.ModuleType("evo_rhyme.metric_benchmark.protocol")

    def default_protocol_manifest() -> dict:
        return {}

    def load_protocol_manifest(_p) -> dict:
        return {}

    fake_protocol.default_protocol_manifest = default_protocol_manifest
    fake_protocol.load_protocol_manifest = load_protocol_manifest
    sys.modules["evo_rhyme"] = fake
    sys.modules["evo_rhyme.metric_benchmark"] = fake_mb
    sys.modules["evo_rhyme.metric_benchmark.schema"] = fake_schema
    sys.modules["evo_rhyme.metric_benchmark.protocol"] = fake_protocol


def _load_label_script():
    _stub_evo_rhyme_metric_benchmark()
    path = ROOT / "scripts/metric_benchmark/label_with_ai.py"
    name = "_label_with_ai_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lab():
    return _load_label_script()


def test_is_insufficient_quota_error_string(lab) -> None:
    assert lab.is_insufficient_quota_error(RuntimeError("insufficient_quota in body"))


def test_is_insufficient_quota_error_dict_body(lab) -> None:
    e = type("E", (), {})()
    e.body = {"error": {"code": "insufficient_quota", "message": "x"}}
    assert lab.is_insufficient_quota_error(e)


def test_row_stratum_rank(lab) -> None:
    assert lab.row_stratum_rank({"harvest": {"metric_benchmark_stratum": "technical"}}) == 0
    assert lab.row_stratum_rank({"harvest": {"metric_benchmark_stratum": "control"}}) == 3
    assert lab.row_stratum_rank({}) == len(lab.STRATUM_LABEL_ORDER)


def test_build_pending_stratum_order(lab) -> None:
    merged = [
        {"id": "m", "harvest": {"metric_benchmark_stratum": "modern"}},
        {"id": "t", "harvest": {"metric_benchmark_stratum": "technical"}},
    ]
    pairs = lab.build_pending_label_pairs(merged, stop_after=None, force=True, label_order="stratum")
    assert [p[1] for p in pairs] == [1, 0]


def test_build_pending_stop_after_semantic(lab) -> None:
    merged = [
        {"id": "a", "harvest": {"metric_benchmark_stratum": "technical"}},
        {"id": "b", "harvest": {"metric_benchmark_stratum": "modern"}},
    ]
    pairs = lab.build_pending_label_pairs(merged, stop_after="semantic", force=True, label_order="stratum")
    assert [p[1] for p in pairs] == [0]


def test_build_pending_input_order(lab) -> None:
    merged = [
        {"id": "b", "harvest": {"metric_benchmark_stratum": "technical"}},
        {"id": "a", "harvest": {"metric_benchmark_stratum": "semantic"}},
    ]
    pairs = lab.build_pending_label_pairs(merged, stop_after=None, force=True, label_order="input")
    assert [p[1] for p in pairs] == [0, 1]


def test_load_jsonl_by_id_roundtrip(tmp_path: Path, lab) -> None:
    p = tmp_path / "x.jsonl"
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = lab._load_jsonl_by_id(p)
    assert d["a"]["x"] == 1
    assert d["b"]["x"] == 2
