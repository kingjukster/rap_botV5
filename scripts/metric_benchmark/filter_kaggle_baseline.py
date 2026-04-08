#!/usr/bin/env python3
"""
Stream the Kaggle Spotify songs.csv and emit a stratified subset by baseline artist groups.

Default quotas (from plan): technical 30%, semantic 30%, modern 20%, control 20%.

Uses only the Python standard library. Intended to run in Docker with /data mounted.

The CSV ``artists`` column must be a JSON array of names (Kaggle/Spotify style), e.g. ``["Eminem"]`` — not a plain string.

Example:
  docker compose -f docker-compose.metric-benchmark.yml run --rm filter \\
    --input /data/songs.csv --output /data/metric_benchmark/baseline_filtered.csv --total-rows 2000
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_baseline():
    path = ROOT / "evo_rhyme/metric_benchmark/baseline_artists.py"
    name = "_mb_baseline_artists"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_baseline = _load_baseline()
classify_track_group = _baseline.classify_track_group
parse_artists_field = _baseline.parse_artists_field
default_quota_fractions = _baseline.default_quota_fractions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _allocate_targets(total: int, quotas: Dict[str, float], groups: List[str]) -> Dict[str, int]:
    vals = {g: total * quotas[g] for g in groups}
    base = {g: int(vals[g]) for g in groups}
    rem = total - sum(base.values())
    fracs = sorted(((vals[g] - base[g], g) for g in groups), reverse=True)
    for i in range(rem):
        base[fracs[i][1]] += 1
    return base


def _one_pass(
    input_path: Path,
    fieldnames: List[str],
    groups: List[str],
    targets: Dict[str, int],
    pools: Dict[str, List[Dict[str, str]]],
    seen_ids: set,
    stats: Dict[str, Any],
    skip_empty_lyrics: bool,
) -> None:
    with open(input_path, newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            stats["rows_read"] += 1
            if all(len(pools[g]) >= targets[g] for g in groups):
                break
            rid = row.get("id") or row.get("ID")
            if rid and rid in seen_ids:
                continue

            lyrics = (row.get("lyrics") or "").strip()
            if skip_empty_lyrics and not lyrics:
                stats["rows_empty_lyrics"] += 1
                continue

            artists = parse_artists_field(row.get("artists") or "")
            g = classify_track_group(artists)
            if g is None:
                stats["rows_no_baseline_artist"] += 1
                continue
            if len(pools[g]) < targets[g]:
                copy_row = {k: row.get(k, "") for k in fieldnames}
                copy_row["metric_benchmark_stratum"] = g
                pools[g].append(copy_row)
                stats["picked"][g] += 1
                if rid:
                    seen_ids.add(rid)


def filter_csv(
    input_path: Path,
    output_path: Path,
    *,
    total_rows: int,
    quotas: Dict[str, float],
    seed: int,
    skip_empty_lyrics: bool = True,
    max_passes: int = 3,
    backfill_to_total: bool = False,
) -> Dict[str, Any]:
    _ = seed  # reserved for stochastic sampling variants
    groups = list(_baseline.GROUP_PRIORITY)
    targets = _allocate_targets(total_rows, quotas, groups)
    pools: Dict[str, List[Dict[str, str]]] = {g: [] for g in groups}
    seen_ids: set = set()
    stats: Dict[str, Any] = {
        "rows_read": 0,
        "rows_empty_lyrics": 0,
        "rows_no_baseline_artist": 0,
        "targets": dict(targets),
        "picked": {g: 0 for g in groups},
        "passes": 0,
        "backfill_rows": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header")
        fieldnames = list(reader.fieldnames)

    for pass_num in range(max_passes):
        if all(len(pools[g]) >= targets[g] for g in groups):
            break
        stats["passes"] = pass_num + 1
        before = sum(len(pools[g]) for g in groups)
        _one_pass(input_path, fieldnames, groups, targets, pools, seen_ids, stats, skip_empty_lyrics)
        after = sum(len(pools[g]) for g in groups)
        if after == before:
            logger.warning(
                "No progress on pass %d (dataset likely exhausted for empty buckets).",
                pass_num + 1,
            )
            break

    stratified_total = sum(len(pools[g]) for g in groups)
    under = {g: targets[g] - len(pools[g]) for g in groups}
    if any(u > 0 for u in under.values()):
        logger.warning(
            "Stratified underfill vs targets (often sparse Tier-5 / control in Kaggle): %s",
            under,
        )

    backfill_rows: List[Dict[str, str]] = []
    if backfill_to_total and stratified_total < total_rows:
        need = total_rows - stratified_total
        logger.info(
            "Backfilling %d rows with any baseline-tier track (stratum=backfill) to reach --total-rows=%d",
            need,
            total_rows,
        )
        with open(input_path, newline="", encoding="utf-8", errors="replace") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if need <= 0:
                    break
                stats["rows_read"] += 1
                rid = row.get("id") or row.get("ID")
                if rid and rid in seen_ids:
                    continue
                if skip_empty_lyrics and not (row.get("lyrics") or "").strip():
                    stats["rows_empty_lyrics"] += 1
                    continue
                artists = parse_artists_field(row.get("artists") or "")
                if classify_track_group(artists) is None:
                    stats["rows_no_baseline_artist"] += 1
                    continue
                copy_row = {k: row.get(k, "") for k in fieldnames}
                copy_row["metric_benchmark_stratum"] = "backfill"
                backfill_rows.append(copy_row)
                if rid:
                    seen_ids.add(rid)
                need -= 1
        stats["backfill_rows"] = len(backfill_rows)

    # Write combined CSV: stratified groups then backfill
    all_rows: List[Dict[str, str]] = []
    for g in groups:
        all_rows.extend(pools[g])
    all_rows.extend(backfill_rows)

    out_fields = list(fieldnames)
    if "metric_benchmark_stratum" not in out_fields:
        out_fields.append("metric_benchmark_stratum")

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    stats["output_rows"] = len(all_rows)
    stats["underfill"] = under
    stats_path = output_path.with_suffix(".filter_stats.json")
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    logger.info(
        "Wrote %d rows to %s (targets %s, underfill %s)",
        len(all_rows),
        output_path,
        targets,
        stats["underfill"],
    )
    logger.info("Stats JSON: %s", stats_path)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Filter Kaggle CSV by baseline artist stratification.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-rows", type=int, default=2000, help="Total songs to keep after stratification")
    p.add_argument("--quota-technical", type=float, default=None)
    p.add_argument("--quota-semantic", type=float, default=None)
    p.add_argument("--quota-modern", type=float, default=None)
    p.add_argument("--quota-control", type=float, default=None)
    p.add_argument("--seed", type=int, default=42, help="Reserved for future randomized reservoir sampling")
    p.add_argument("--max-passes", type=int, default=5, dest="max_passes", help="Rescan CSV passes to fill quotas")
    p.add_argument(
        "--backfill-to-total",
        action="store_true",
        help="After stratified passes, add more baseline-tier rows (stratum=backfill) until output has --total-rows",
    )
    args = p.parse_args()

    quotas = default_quota_fractions()
    if args.quota_technical is not None:
        quotas["technical"] = args.quota_technical
    if args.quota_semantic is not None:
        quotas["semantic"] = args.quota_semantic
    if args.quota_modern is not None:
        quotas["modern"] = args.quota_modern
    if args.quota_control is not None:
        quotas["control"] = args.quota_control
    s = sum(quotas.values())
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"Quotas must sum to 1.0, got {s}")

    filter_csv(
        args.input,
        args.output,
        total_rows=args.total_rows,
        quotas=quotas,
        seed=args.seed,
        max_passes=args.max_passes,
        backfill_to_total=args.backfill_to_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
