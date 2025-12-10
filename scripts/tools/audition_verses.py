#!/usr/bin/env python
"""
audition_verses.py

Quick CLI for browsing top verses from scored_dataset.jsonl. Supports filtering
by seed/tag/persona and sorting by critic metrics. Useful for fast manual
auditions before promoting new data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import textwrap
from typing import Dict, Iterable, List

from config.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audition top-ranked verses from scored_dataset.jsonl.")
    parser.add_argument("--scored", type=str, default=None, help="Path to scored_dataset.jsonl (default from config).")
    parser.add_argument("--metric", type=str, default="overall_score", choices=["overall_score", "depth_score", "coherence_score", "originality_score", "verse_score"], help="Metric used for sorting.")
    parser.add_argument("--top", type=int, default=5, help="Number of verses to display.")
    parser.add_argument("--min_overall", type=float, default=None, help="Minimum critic overall_score.")
    parser.add_argument("--seed_id", type=str, default=None, help="Filter by structured seed id.")
    parser.add_argument("--seed", type=str, default=None, help="Filter by seed substring.")
    parser.add_argument("--tag", action="append", default=[], help="Require at least one of these tags (can be repeated).")
    parser.add_argument("--persona", type=str, default=None, help="Filter by persona substring.")
    parser.add_argument("--scheme", type=str, default=None, help="Filter by rhyme scheme (e.g., AABB).")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write selected records (JSONL).")
    parser.add_argument("--config", type=str, default=None, help="Optional config file for defaults.")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def verse_matches(record: Dict, args: argparse.Namespace) -> bool:
    critic = record.get("critic") or {}
    overall = critic.get("overall_score")
    if args.min_overall is not None and (overall is None or overall < args.min_overall):
        return False
    if args.seed_id and record.get("seed_id") != args.seed_id:
        return False
    if args.seed and args.seed.lower() not in (record.get("seed") or "").lower():
        return False
    if args.persona and args.persona.lower() not in (record.get("persona") or "").lower():
        return False
    if args.scheme and (record.get("scheme") or "").upper() != args.scheme.upper():
        return False
    if args.tag:
        tags = [t.lower() for t in (record.get("seed_tags") or [])]
        if not any(tag.lower() in tags for tag in args.tag):
            return False
    return True


def format_entry(record: Dict, metric: str, idx: int):
    critic = record.get("critic") or {}
    verse_text = record.get("verse_text") or ""
    wrapped = textwrap.indent(verse_text.strip(), "    ")
    tags = ", ".join(record.get("seed_tags") or [])
    overall = float(critic.get("overall_score", 0.0))
    depth = float(critic.get("depth_score", 0.0))
    coherence = float(critic.get("coherence_score", 0.0))
    originality = float(critic.get("originality_score", 0.0))
    verse_score = float(record.get("verse_score") or 0.0)
    metric_value = verse_score if metric == "verse_score" else float(critic.get(metric, 0.0))
    print("=" * 80)
    print(
        f"[{idx}] {record.get('seed_id') or record.get('seed')}"
        f" | scheme={record.get('scheme')} | tag(s)={tags or '—'}"
    )
    print(
        f"  critic overall={overall:.2f} depth={depth:.2f} coherence={coherence:.2f} "
        f"originality={originality:.2f} | verse_score={verse_score:.3f}"
    )
    print(f"  metric={metric} → {metric_value:.3f}")
    print(wrapped)


def main():
    args = parse_args()
    settings = load_settings(args.config)
    scored_path = Path(args.scored or settings.scored_dataset_path)
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored dataset not found: {scored_path}")

    records = [record for record in read_jsonl(scored_path) if verse_matches(record, args)]
    if not records:
        print("[AUDITION] No records matched the filters.")
        return

    def metric_value(record: Dict) -> float:
        if args.metric == "verse_score":
            return float(record.get("verse_score") or 0.0)
        critic = record.get("critic") or {}
        return float(critic.get(args.metric, 0.0))

    records.sort(key=metric_value, reverse=True)
    selected = records[: args.top]
    for idx, record in enumerate(selected, start=1):
        format_entry(record, args.metric, idx)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in selected:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[AUDITION] Wrote {len(selected)} records to {output_path}")


if __name__ == "__main__":
    main()
