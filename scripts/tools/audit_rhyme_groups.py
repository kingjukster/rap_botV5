#!/usr/bin/env python
"""
Quick heuristic audit for rhymes_grouped.csv.

Flags the words in each group whose suffixes do not match the group's
majority rhyme key. Helpful for spotting noisy Siamese assignments.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def clean_tail(word: str, max_len: int) -> str:
    cleaned = re.sub(r"[^a-z]", "", str(word).lower())
    return cleaned[-max_len:] if cleaned else ""


def suffix_overlap(a: str, b: str) -> int:
    max_len = min(len(a), len(b))
    count = 0
    for i in range(1, max_len + 1):
        if a[-i:] == b[-i:]:
            count = i
        else:
            break
    return count


def audit_groups(
    csv_path: Path,
    rhyme_key_len: int,
    min_suffix_overlap: int,
    top_k: int,
) -> List[Tuple[int, str, str, int, str]]:
    df = pd.read_csv(csv_path)
    if "word" not in df.columns or "group" not in df.columns:
        raise ValueError("CSV must contain 'word' and 'group' columns.")

    issues: List[Tuple[int, str, str, int, str]] = []

    for group_id, group_df in df.groupby("group"):
        words = group_df["word"].tolist()
        suffixes = [clean_tail(w, rhyme_key_len) for w in words if w]
        if not suffixes:
            continue
        majority_suffix, freq = Counter(suffixes).most_common(1)[0]
        for _, row in group_df.iterrows():
            word = str(row["word"])
            method = str(row.get("method", "unknown"))
            suffix = clean_tail(word, rhyme_key_len)
            cleaned = re.sub(r"[^a-z]", "", word.lower())
            if not cleaned:
                continue
            overlap = suffix_overlap(suffix, majority_suffix)
            if len(cleaned) > min_suffix_overlap and overlap < min_suffix_overlap:
                issues.append((group_id, word, majority_suffix, overlap, method))

    issues.sort(key=lambda item: (item[3], item[0]))
    return issues[:top_k] if top_k and top_k > 0 else issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rhyme groups for suffix mismatches.")
    parser.add_argument("--input", type=str, default="data/rhymes_grouped.csv", help="Path to rhymes_grouped.csv")
    parser.add_argument("--rhyme_key_len", type=int, default=4, help="Length of suffix used for majority voting.")
    parser.add_argument(
        "--min_suffix_overlap",
        type=int,
        default=2,
        help="Minimum overlap (in chars) required to consider a word aligned with the group.",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Limit output to the worst-K offenders (0 = all).")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    issues = audit_groups(
        csv_path=csv_path,
        rhyme_key_len=args.rhyme_key_len,
        min_suffix_overlap=args.min_suffix_overlap,
        top_k=args.top_k,
    )
    if not issues:
        print("[AUDIT] No obvious suffix mismatches found.")
        return
    print(f"[AUDIT] Showing {len(issues)} potential mismatches (group, word, majority_suffix, overlap, method):")
    for group_id, word, majority_suffix, overlap, method in issues:
        print(f"  G{group_id:04d} :: '{word}' vs '{majority_suffix}' overlap={overlap} method={method}")


if __name__ == "__main__":
    main()
