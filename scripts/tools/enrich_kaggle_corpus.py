#!/usr/bin/env python
"""enrich_kaggle_corpus.py

Parse the cleaned Kaggle corpus and emit a JSONL file with extra metadata
for each bar (language heuristic, section hints, rhyme density stats, etc.).
Also writes a small summary JSON for quick sanity checks.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

META_RE = re.compile(r"<([^>=]+)=([^>]+)>")
BAR_RE = re.compile(r"^\[BAR\]\s*(.*)")
TAG_RHY = re.compile(r"\[RHY=([A-Z])\]")
TAG_SYL = re.compile(r"\[SYL_([^\]]+)\]")
TAG_INT = re.compile(r"\[INT_([^\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z']+")

PROFANITY = {
    "fuck",
    "shit",
    "bitch",
    "ass",
    "nigga",
    "nigger",
    "motherfucker",
    "dick",
    "pussy",
}


def parse_meta(raw: str) -> Dict[str, str]:
    meta = {}
    for match in META_RE.finditer(raw):
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


def iter_tracks(path: Path) -> Iterable[Tuple[Dict[str, str], List[Bar]]]:
    current_meta: Dict[str, str] = {}
    current_bars: List[Bar] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<") and line.endswith(">") and "[BAR]" not in line:
                if current_bars:
                    yield current_meta, current_bars
                    current_bars = []
                current_meta = parse_meta(line)
                continue
            if line.startswith("[BAR]"):
                bar = parse_bar_line(line)
                if bar:
                    current_bars.append(bar)
    if current_bars:
        yield current_meta, current_bars


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


def normalized_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9 ]+", "", lowered)
    return lowered.strip()


def has_profanity(text: str) -> bool:
    tokens = WORD_RE.findall(text.lower())
    return any(token in PROFANITY for token in tokens)


def sliding_metrics(bars: List[Bar], window: int) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    window_deque: deque[Bar] = deque(maxlen=window)
    for bar in bars:
        window_deque.append(bar)
        rhymes = [b.rhyme for b in window_deque if b.rhyme]
        unique_rhymes = len(set(rhymes))
        dense_count = sum(1 for b in window_deque if b.intensity and b.intensity.upper() == "DENSE")
        avg_syllables = sum(b.syllables or 0.0 for b in window_deque) / len(window_deque)
        metrics.append(
            {
                "unique_rhymes_window": unique_rhymes,
                "dense_bars_window": dense_count,
                "avg_syllables_window": avg_syllables,
            }
        )
    return metrics


def infer_section_hint(text: str, seen_counts: Counter) -> str:
    normalized = normalized_text(text)
    if not normalized:
        return "other"
    count = seen_counts[normalized]
    seen_counts[normalized] += 1
    if count >= 2:
        return "hook"
    if len(normalized.split()) <= 4:
        return "filler"
    return "verse"


def process_corpus(
    corpus: Path,
    output_path: Path,
    summary_path: Path,
    window_size: int = 4,
    english_threshold: float = 0.85,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    rhyme_diversity: List[int] = []
    track_counter = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for track_idx, (meta, bars) in enumerate(iter_tracks(corpus)):
            track_counter += 1
            if not bars:
                continue
            sliding = sliding_metrics(bars, window_size)
            seen_lines: Counter = Counter()
            for idx, bar in enumerate(bars):
                ascii_score = ascii_ratio(bar.text)
                tokens = WORD_RE.findall(bar.text.lower())
                alpha_ratio = (len(tokens) / max(1, len(bar.text.split()))) if bar.text.strip() else 0.0
                english_like = ascii_score >= english_threshold and alpha_ratio >= 0.5
                entry = {
                    "track_index": track_idx,
                    "artist": meta.get("artist"),
                    "title": meta.get("title"),
                    "source": meta.get("source"),
                    "coherence": float(meta.get("coherence", 0.0) or 0.0),
                    "bar_index": idx,
                    "text": bar.text,
                    "rhyme_letter": bar.rhyme,
                    "intensity": bar.intensity,
                    "syllables": bar.syllables,
                    "ascii_ratio": round(ascii_score, 4),
                    "alpha_ratio": round(alpha_ratio, 4),
                    "english_like": english_like,
                    "has_profanity": has_profanity(bar.text),
                    "section_hint": infer_section_hint(bar.text, seen_lines),
                }
                entry.update(sliding[idx])
                rhyme_diversity.append(sliding[idx]["unique_rhymes_window"])
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                stats["total_bars"] += 1
                if english_like:
                    stats["english_like_bars"] += 1
                if entry["section_hint"] == "hook":
                    stats["hook_bars"] += 1
                if bar.intensity and bar.intensity.upper() == "DENSE":
                    stats["dense_bars"] += 1

    summary = {
        "total_tracks": track_counter,
        "total_bars": stats.get("total_bars", 0),
        "english_like_ratio": (stats.get("english_like_bars", 0) / max(1, stats.get("total_bars", 0))),
        "hook_ratio": (stats.get("hook_bars", 0) / max(1, stats.get("total_bars", 0))),
        "dense_ratio": (stats.get("dense_bars", 0) / max(1, stats.get("total_bars", 0))),
        "avg_unique_rhymes_window": (sum(rhyme_diversity) / max(1, len(rhyme_diversity))),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich Kaggle corpus with metadata tags.")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned Kaggle corpus (TXT).")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path for enriched bars.")
    parser.add_argument("--summary", type=str, required=True, help="Summary JSON path.")
    parser.add_argument("--window_size", type=int, default=4, help="Sliding window size for density metrics.")
    parser.add_argument(
        "--english_threshold",
        type=float,
        default=0.85,
        help="Minimum ASCII ratio to mark a bar as English-like.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_path = Path(args.input)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    process_corpus(
        corpus=corpus_path,
        output_path=output_path,
        summary_path=summary_path,
        window_size=args.window_size,
        english_threshold=args.english_threshold,
    )
    print(f"[DONE] Enriched corpus written to {output_path}")
    print(f"[DONE] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
