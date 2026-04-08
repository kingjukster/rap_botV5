"""
Deterministic verse segmentation: stanza_blank_line_v2 (plan).

- Split on blank lines into stanzas.
- Merge stanzas with < min_lines forward until >= min_lines or EOF.
- Stanzas with > max_bars lines -> sliding windows of size max_bars with stride.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SegmentationStats:
    merged_short_stanzas: int = 0
    dropped_incomplete: int = 0
    sliding_windows: int = 0
    raw_stanzas: int = 0


def normalize_lyrics_lines(text: str) -> List[str]:
    """Split lyrics into non-empty stripped lines (preserve order)."""
    if not text or not str(text).strip():
        return []
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [ln.strip() for ln in lines if ln.strip()]


def _split_stanzas_by_blank(lines: List[str]) -> List[List[str]]:
    stanzas: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if not ln.strip():
            if cur:
                stanzas.append(cur)
                cur = []
            continue
        cur.append(ln.strip())
    if cur:
        stanzas.append(cur)
    return stanzas


def merge_short_stanzas(
    stanzas: List[List[str]],
    min_lines: int,
    stats: SegmentationStats,
) -> List[List[str]]:
    """Merge consecutive stanzas until each chunk has at least min_lines (or EOF)."""
    out: List[List[str]] = []
    i = 0
    while i < len(stanzas):
        chunk = list(stanzas[i])
        i += 1
        while len(chunk) < min_lines and i < len(stanzas):
            stats.merged_short_stanzas += 1
            chunk.extend(stanzas[i])
            i += 1
        if len(chunk) >= min_lines:
            out.append(chunk)
        else:
            stats.dropped_incomplete += 1
            logger.debug("dropping trailing %d lines (< min_lines)", len(chunk))
    return out


def sliding_window_verses(
    lines: List[str],
    max_bars: int,
    stride: int,
    stats: SegmentationStats,
) -> List[Tuple[List[str], Dict[str, Any]]]:
    """Return list of (verse_lines, harvest_extra)."""
    n = len(lines)
    if n <= max_bars:
        return [
            (
                lines,
                {"window_index": 0, "segmentation_rule": "stanza_blank_line_v2"},
            )
        ]
    result: List[Tuple[List[str], Dict[str, Any]]] = []
    wi = 0
    start = 0
    while start + max_bars <= n:
        chunk = lines[start : start + max_bars]
        result.append(
            (
                chunk,
                {
                    "window_index": wi,
                    "segmentation_rule": "sliding_window_v1",
                    "window_start_line": start,
                },
            )
        )
        stats.sliding_windows += 1
        wi += 1
        start += stride
    return result


def segment_song_to_verses(
    lyrics_text: str,
    *,
    min_lines: int = 2,
    max_bars: int = 16,
    stride: Optional[int] = None,
) -> Tuple[List[Tuple[List[str], Dict[str, Any]]], SegmentationStats]:
    """
    Segment full song lyrics into verses (list of bar lists + harvest metadata).

    stride defaults to max(1, max_bars // 2).
    """
    stats = SegmentationStats()
    if stride is None:
        stride = max(1, max_bars // 2)
    lines = normalize_lyrics_lines(lyrics_text)
    if len(lines) < min_lines:
        return [], stats

    raw_stanzas = _split_stanzas_by_blank(lines)
    stats.raw_stanzas = len(raw_stanzas)
    merged = merge_short_stanzas(raw_stanzas, min_lines, stats)

    verses: List[Tuple[List[str], Dict[str, Any]]] = []
    for vi, stanza in enumerate(merged):
        base_harvest = {
            "verse_index": vi,
            "segmentation_rule": "stanza_blank_line_v2",
        }
        if len(stanza) <= max_bars:
            verses.append((stanza, dict(base_harvest)))
        else:
            for chunk, extra in sliding_window_verses(stanza, max_bars, stride, stats):
                h = {**base_harvest, **extra}
                verses.append((chunk, h))
    return verses, stats
