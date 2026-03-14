"""
evo_rhyme/style_profile.py

StyleProfile: statistical profile of a reference corpus for style-matching.
Built from lines or couplets; used to score style similarity of generated couplets.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evo_rhyme.phonetics import extract_rhyme_tail, syllable_count_line, tokenize_line


@dataclass
class StyleProfile:
    """Statistical profile of reference corpus for style matching."""

    avg_syllables_per_line: float
    avg_word_length: float
    word_freq_dist: List[Tuple[str, int]]  # top N (word, count)
    rhyme_tail_dist: List[Tuple[str, int]]  # top N (tail, count)
    line_length_std: float  # std dev of syllable counts per line


def _lines_from_corpus(
    lines_or_couplets: List[str],
) -> List[str]:
    """
    Flatten corpus into list of lines.
    Handles both line-per-item and couplet-per-item (two lines separated by | or newline).
    """
    result: List[str] = []
    for item in lines_or_couplets:
        item = item.strip()
        if not item:
            continue
        if "|" in item:
            for part in item.split("|"):
                part = part.strip()
                if part:
                    result.append(part)
        elif "\n" in item:
            for part in item.split("\n"):
                part = part.strip()
                if part:
                    result.append(part)
        else:
            result.append(item)
    return result


def build_style_profile(
    lines_or_couplets: List[str],
    top_n: int = 50,
) -> StyleProfile:
    """
    Build StyleProfile from reference corpus.

    Args:
        lines_or_couplets: List of lines or couplets (one per item; couplets may use | or newline).
        top_n: Number of top words and rhyme tails to keep in distributions.

    Returns:
        StyleProfile with avg_syllables_per_line, avg_word_length, word_freq_dist,
        rhyme_tail_dist, line_length_std.
    """
    lines = _lines_from_corpus(lines_or_couplets)
    if not lines:
        return StyleProfile(
            avg_syllables_per_line=10.0,
            avg_word_length=4.0,
            word_freq_dist=[],
            rhyme_tail_dist=[],
            line_length_std=0.0,
        )

    # Syllable counts per line
    syl_counts: List[int] = []
    word_lengths: List[float] = []
    word_counts: Counter = Counter()
    tail_counts: Counter = Counter()

    for line in lines:
        syl = syllable_count_line(line)
        syl_counts.append(syl)
        tokens = tokenize_line(line)
        for t in tokens:
            if len(t) > 1:  # skip single-char noise
                word_lengths.append(len(t))
                word_counts[t.lower()] += 1
        if tokens:
            tail = extract_rhyme_tail(tokens[-1])
            if tail:
                tail_counts[tail] += 1

    avg_syl = sum(syl_counts) / len(syl_counts) if syl_counts else 10.0
    avg_wlen = sum(word_lengths) / len(word_lengths) if word_lengths else 4.0
    variance = (
        sum((s - avg_syl) ** 2 for s in syl_counts) / len(syl_counts)
        if syl_counts
        else 0.0
    )
    line_std = math.sqrt(variance) if variance > 0 else 0.0

    return StyleProfile(
        avg_syllables_per_line=avg_syl,
        avg_word_length=avg_wlen,
        word_freq_dist=word_counts.most_common(top_n),
        rhyme_tail_dist=tail_counts.most_common(top_n),
        line_length_std=line_std,
    )


def load_lines_from_file(path: Path) -> List[str]:
    """Load lines from a text file (one line or couplet per file line)."""
    path = Path(path)
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def get_artist_presets() -> Dict[str, StyleProfile]:
    """Return preset style profiles for well-known rap artists.

    These are approximate statistical profiles based on common
    characteristics of each artist's style. Used by emitters to
    generate style-diverse candidates.
    """
    return {
        "kendrick": StyleProfile(
            avg_syllables_per_line=11.5,
            avg_word_length=4.2,
            word_freq_dist=[
                ("humble", 8), ("dna", 6), ("loyalty", 5), ("pray", 4),
                ("blood", 4), ("real", 4), ("wicked", 3), ("mortal", 3),
                ("spirit", 3), ("level", 3), ("story", 3), ("power", 3),
                ("heart", 3), ("streets", 2), ("faith", 2), ("mirror", 2),
                ("truth", 2), ("weakness", 2), ("vanity", 2), ("pride", 2),
            ],
            rhyme_tail_dist=[
                ("AY1 N", 5), ("AY1 T", 4), ("IY1", 4), ("AH1 N", 3),
                ("EH1 L", 3), ("AY1 V", 3), ("IH1 NG", 3), ("OW1", 2),
            ],
            line_length_std=3.2,
        ),
        "doom": StyleProfile(
            avg_syllables_per_line=13.0,
            avg_word_length=4.8,
            word_freq_dist=[
                ("villain", 6), ("metal", 5), ("doom", 5), ("mask", 4),
                ("rhyme", 4), ("scheme", 3), ("bizarre", 3), ("herb", 3),
                ("flow", 3), ("cipher", 3), ("plot", 2), ("chrome", 2),
                ("sinister", 2), ("formula", 2), ("abstract", 2),
                ("grotesque", 2), ("technique", 2), ("riddle", 2),
                ("enigma", 2), ("paradox", 2),
            ],
            rhyme_tail_dist=[
                ("IY1 M", 5), ("AE1 K", 4), ("AH1 N", 4), ("OW1", 3),
                ("AY1 M", 3), ("EH1 S", 3), ("IH1 NG", 2), ("AO1 R", 2),
            ],
            line_length_std=2.8,
        ),
        "nas": StyleProfile(
            avg_syllables_per_line=11.0,
            avg_word_length=4.0,
            word_freq_dist=[
                ("street", 6), ("project", 5), ("life", 5), ("world", 4),
                ("dream", 4), ("bridge", 3), ("queen", 3), ("king", 3),
                ("ghetto", 3), ("struggle", 3), ("hustle", 3), ("mind", 3),
                ("soul", 2), ("child", 2), ("survive", 2), ("vision", 2),
                ("concrete", 2), ("jungle", 2), ("escape", 2), ("legacy", 2),
            ],
            rhyme_tail_dist=[
                ("AY1 F", 4), ("IY1 T", 4), ("AY1 N", 3), ("EH1 R", 3),
                ("IY1 M", 3), ("AH1 N", 3), ("OW1", 2), ("AY1 Z", 2),
            ],
            line_length_std=2.5,
        ),
        "eminem": StyleProfile(
            avg_syllables_per_line=14.0,
            avg_word_length=4.1,
            word_freq_dist=[
                ("slim", 5), ("shady", 5), ("marshall", 4), ("rage", 4),
                ("anger", 3), ("mother", 3), ("daughter", 3), ("crazy", 3),
                ("brain", 3), ("pain", 3), ("insane", 3), ("game", 3),
                ("fame", 2), ("blame", 2), ("name", 2), ("shame", 2),
                ("flame", 2), ("frame", 2), ("claim", 2), ("aim", 2),
            ],
            rhyme_tail_dist=[
                ("EY1 N", 6), ("EY1 M", 5), ("AE1 K", 4), ("IH1 NG", 3),
                ("AY1 T", 3), ("EH1 R", 3), ("AY1 N", 2), ("EY1 Z", 2),
            ],
            line_length_std=3.5,
        ),
        "jcole": StyleProfile(
            avg_syllables_per_line=12.0,
            avg_word_length=4.3,
            word_freq_dist=[
                ("dream", 5), ("cole", 4), ("forest", 3), ("hills", 3),
                ("dollar", 3), ("sign", 3), ("middle", 3), ("child", 3),
                ("born", 3), ("sinner", 3), ("crooked", 2), ("smile", 2),
                ("heaven", 2), ("wisdom", 2), ("patience", 2), ("growth", 2),
                ("lesson", 2), ("struggle", 2), ("humble", 2), ("journey", 2),
            ],
            rhyme_tail_dist=[
                ("AY1 N", 4), ("AY1 M", 4), ("IY1", 3), ("OW1", 3),
                ("AH1 N", 3), ("EH1 L", 2), ("AY1 F", 2), ("EH1 R", 2),
            ],
            line_length_std=2.6,
        ),
    }
