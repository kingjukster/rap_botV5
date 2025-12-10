#!/usr/bin/env python
"""
score_with_openai.py

Reads generation logs (JSONL) and queries an OpenAI model to produce
strict JSON critic scores. Appends/creates an output JSONL compatible
with consolidate_stage3.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import time
from typing import Dict, Iterable, Set

from openai import OpenAI, OpenAIError


SYSTEM_PROMPT = """You are a ruthless underground rap critic.
Return ONLY valid JSON with numeric fields overall_score, depth_score,
coherence_score, originality_score, line_scores (array of floats), tags
(array of strings), theme (short text), and notes (<=3 sentences). Scores
must be between 0 and 5 (inclusive). Be harsh and concise."""

USER_TEMPLATE = """Verse ID: {verse_id}
Artist: {artist}
Seed: {seed}
Scheme: {scheme}
Verse text:
{verse_text}

Output JSON only."""


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


def parse_args():
    parser = argparse.ArgumentParser(description="Score verses with the OpenAI critic.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/generated_raw.jsonl",
        help="JSONL file produced by generate_rhymed_verse.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/critic_scores.jsonl",
        help="Destination JSONL for critic scores.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat/completions model to use.",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    parser.add_argument("--max", type=int, default=None, help="Optional cap on number of verses to score.")
    return parser.parse_args()


def load_completed(path: Path) -> Set[str]:
    completed: Set[str] = set()
    if not path.exists():
        return completed
    for record in read_jsonl(path):
        verse_id = record.get("verse_id")
        if verse_id:
            completed.add(verse_id)
    return completed


def normalize_text(entry: Dict) -> str:
    text = entry.get("verse_text")
    if text:
        return text
    bars = entry.get("bars") or []
    lines = [bar.get("text", "") for bar in bars if bar.get("text")]
    lines.append("<END_SONG>")
    return "\n".join(lines)


def main():
    args = parse_args()
    client = OpenAI()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    completed = load_completed(output_path)
    print(f"[INFO] Existing scores found for {len(completed)} verse_ids.")

    output_file = open(output_path, "a", encoding="utf-8")
    scored = 0

    try:
        for entry in read_jsonl(input_path):
            verse_id = entry.get("verse_id")
            if not verse_id or verse_id in completed:
                continue
            verse_text = normalize_text(entry)
            user_prompt = USER_TEMPLATE.format(
                verse_id=verse_id,
                artist=entry.get("artist"),
                seed=entry.get("seed"),
                scheme=entry.get("scheme"),
                verse_text=verse_text,
            )
            while True:
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    content = response.choices[0].message.content.strip()
                    critic_json = json.loads(content)
                    critic_json["verse_id"] = verse_id
                    output_file.write(json.dumps(critic_json) + "\n")
                    output_file.flush()
                    completed.add(verse_id)
                    scored += 1
                    print(f"[CRITIC] Scored verse_id={verse_id} overall={critic_json.get('overall_score')}")
                    break
                except (OpenAIError, json.JSONDecodeError) as exc:
                    print(f"[WARN] Error scoring verse {verse_id}: {exc}; retrying in 5s")
                    time.sleep(5.0)
            if args.max and scored >= args.max:
                print(f"[INFO] Reached max={args.max} scores; stopping.")
                break
            time.sleep(args.sleep)
    finally:
        output_file.close()

    print(f"[DONE] Added {scored} critic scores to {output_path}")


if __name__ == "__main__":
    main()
