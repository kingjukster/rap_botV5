"""Tests for scripts/metric_benchmark/filter_kaggle_baseline.py (dynamic import)."""

from __future__ import annotations

import csv
import json
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_filter_module():
    path = ROOT / "scripts/metric_benchmark/filter_kaggle_baseline.py"
    name = "_filter_kaggle_baseline_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def filt():
    return _load_filter_module()


def _write_songs_csv(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "artists", "lyrics"])
        w.writeheader()
        for i in range(n):
            w.writerow(
                {
                    "id": str(i + 1),
                    "artists": json.dumps(["Eminem"]),
                    "lyrics": "first bar\n\nsecond bar",
                }
            )


def test_backfill_reaches_total_rows(tmp_path: Path, filt) -> None:
    csv_path = tmp_path / "songs.csv"
    out_path = tmp_path / "out.csv"
    _write_songs_csv(csv_path, n=20)
    quotas = filt.default_quota_fractions()
    st = filt.filter_csv(
        csv_path,
        out_path,
        total_rows=10,
        quotas=quotas,
        seed=1,
        max_passes=3,
        backfill_to_total=True,
    )
    assert st["output_rows"] == 10
    assert st["backfill_rows"] == 7
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    c = Counter(r["metric_benchmark_stratum"] for r in rows)
    assert c["technical"] == 3
    assert c["backfill"] == 7


def test_no_backfill_partial_stratified_only(tmp_path: Path, filt) -> None:
    csv_path = tmp_path / "songs.csv"
    out_path = tmp_path / "out.csv"
    _write_songs_csv(csv_path, n=20)
    quotas = filt.default_quota_fractions()
    st = filt.filter_csv(
        csv_path,
        out_path,
        total_rows=10,
        quotas=quotas,
        seed=1,
        max_passes=3,
        backfill_to_total=False,
    )
    assert st["output_rows"] == 3
    assert st["backfill_rows"] == 0
