#!/usr/bin/env python
"""
mine_seed_density.py

Scan the cleaned Kaggle corpus and surface high-density rhyme segments
that make good Stage-3 seed prompts.

Usage:
    python scripts/tools/mine_seed_density.py \
        --corpus data/elite_kaggle_corpus_clean.txt \
        --output data/seeds/kaggle_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable

META_RE = re.compile(r"<([^>=]+)=([^>]+)>")
BAR_RE = re.compile(r"^\[BAR\]\s*(.*)")
TAG_RHY = re.compile(r"\[RHY=([A-Z])\]")
TAG_SYL = re.compile(r"\[SYL_([^]]+)\]")
TAG_INT = re.compile(r"\[INT_([^]]+)\]")


def parse_meta(line: str) -> Dict[str, str]:
    meta = {}
    for match in META_RE.finditer(line):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        meta[key] = value
    return meta


@dataclass
class Bar:
    text: str
    rhyme: str | None
    intensity: str | None
    syllables: float | None


def parse_bar_line(raw: str) -> Bar | None:
    match = BAR_RE.match(raw)
    if not match:
        return None
    content = match.group(1)
    tag_start = content.find("[")
    if tag_start != -1:
        text = content[:tag_start].strip()
        tags = content[tag_start:]
    else:
        text = content.strip()
        tags = ""
    rhyme_match = TAG_RHY.search(tags)
    syllable_match = TAG_SYL.search(tags)
    intensity_match = TAG_INT.search(tags)
    syllable_value = None
    if syllable_match:
        token = syllable_match.group(1)
        if token.endswith("_PLUS"):
            try:
                syllable_value = float(token.split("_")[0])
            except ValueError:
                syllable_value = None
        elif "_" in token:
            parts = token.split("_")
            try:
                syllable_value = sum(float(p) for p in parts) / len(parts)
            except ValueError:
                syllable_value = None
        else:
            try:
                syllable_value = float(token)
            except ValueError:
                syllable_value = None
    return Bar(
        text=text,
        rhyme=rhyme_match.group(1) if rhyme_match else None,
        intensity=intensity_match.group(1) if intensity_match else None,
        syllables=syllable_value,
    )


def iter_tracks(path: Path) -> Iterable[Dict]:
    current_meta: Dict[str, str] = {}
    current_bars: List[Bar] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<") and line.endswith(">") and "[BAR]" not in line:
                if current_bars:
                    yield {"meta": current_meta, "bars": current_bars}
                    current_bars = []
                current_meta = parse_meta(line)
                continue
            if line.startswith("[BAR]"):
                bar = parse_bar_line(line)
                if bar:
                    current_bars.append(bar)
    if current_bars:
        yield {"meta": current_meta, "bars": current_bars}


def score_window(window: List[Bar]) -> Dict:
    rhymes = [bar.rhyme for bar in window if bar.rhyme]
    unique_rhymes = len(set(rhymes))
    dense_bars = sum(1 for bar in window if bar.intensity and bar.intensity.upper() == "DENSE")
    med_bars = sum(1 for bar in window if bar.intensity and bar.intensity.upper() == "MED")
    avg_syllables = sum(bar.syllables or 0.0 for bar in window) / len(window)
    rhyme_sequence = "".join(letter or "_" for letter in rhymes)
    score = (dense_bars * 2) + med_bars + unique_rhymes + (avg_syllables / 10.0)
    return {
        "score": score,
        "unique_rhymes": unique_rhymes,
        "dense_bars": dense_bars,
        "med_bars": med_bars,
        "avg_syllables": avg_syllables,
        "rhyme_sequence": rhyme_sequence,
        "text": " ".join(bar.text for bar in window),
    }


def mine_segments(
    corpus: Path,
    window_size: int,
    min_dense: int,
    min_unique_rhymes: int,
    top_n: int,
) -> List[Dict]:
    candidates: List[Dict] = []
    for track in iter_tracks(corpus):
        meta = track["meta"]
        bars: List[Bar] = track["bars"]
        if len(bars) < window_size:
            continue
        window: deque[Bar] = deque(maxlen=window_size)
        for idx, bar in enumerate(bars):
            window.append(bar)
            if len(window) < window_size:
                continue
            window_bars = list(window)
            metrics = score_window(window_bars)
            if metrics["dense_bars"] < min_dense:
                continue
            if metrics["unique_rhymes"] < min_unique_rhymes:
                continue
            candidate = {
                "artist": meta.get("artist"),
                "title": meta.get("title"),
                "start_bar_index": idx - window_size + 1,
                "window_size": window_size,
                "seed_prompt": metrics["text"],
                "score": round(metrics["score"], 4),
                "unique_rhymes": metrics["unique_rhymes"],
                "dense_bars": metrics["dense_bars"],
                "med_bars": metrics["med_bars"],
                "avg_syllables": metrics["avg_syllables"],
                "rhyme_sequence": metrics["rhyme_sequence"],
            }
            candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if top_n and len(candidates) > top_n:
        candidates = candidates[:top_n]
    return candidates


def parse_cli():
    parser = argparse.ArgumentParser(description="Mine high-density rhyme segments for seed prompts.")
    parser.add_argument("--corpus", type=str, required=True, help="Path to the cleaned Kaggle corpus.")
    parser.add_argument("--output", type=str, required=True, help="Where to write candidate JSONL entries.")
    parser.add_argument("--window_size", type=int, default=4, help="Number of bars per candidate segment.")
    parser.add_argument("--min_dense", type=int, default=1, help="Minimum number of INT_DENSE bars in a window.")
    parser.add_argument(
        "--min_unique_rhymes",
        type=int,
        default=2,
        help="Require at least this many distinct rhyme letters in the window.",
    )
    parser.add_argument("--top_n", type=int, default=500, help="Limit number of output rows.")
    return parser.parse_args()


def main():
    args = parse_cli()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    output_path = Path(args.output)
    candidates = mine_segments(
        corpus=corpus_path,
        window_size=args.window_size,
        min_dense=args.min_dense,
        min_unique_rhymes=args.min_unique_rhymes,
        top_n=args.top_n,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[DONE] Wrote {len(candidates):,} seed candidates to {output_path}")


if __name__ == "__main__":
    main()
