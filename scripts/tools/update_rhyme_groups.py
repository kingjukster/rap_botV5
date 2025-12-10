#!/usr/bin/env python
"""
update_rhyme_groups.py

Automatically expand rhymes_grouped.csv by mining the cleaned elite corpus.

Steps:
  1. Scan the Stage-2/3 corpus and collect high-frequency bar-ending words.
  2. Skip entries that already exist in rhymes_grouped.csv.
  3. Try to assign each remaining word to an existing group via:
       a) Pronouncing-based rhyme signature matches.
       b) Siamese encoder similarity between end-word embeddings.
  4. Cluster any leftovers by rhyme key / phones tail and mint new group ids.
  5. Write an updated CSV with confidence + method metadata.

Usage example:

    python scripts/tools/update_rhyme_groups.py \
        --corpus_path data/elite_kaggle_corpus_clean.txt \
        --existing_csv rhymes_grouped.csv \
        --output_csv rhymes_grouped.csv \
        --min_count 4 \
        --siamese_model_dir rhyme_siamese
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pronouncing
import torch

from rapbot.rhyme_scorer import SiameseRhymeScorer


WORD_RE = re.compile(r"[A-Za-z']+")
BAR_RE = re.compile(r"\[BAR\](.*)")
BRACKET_RE = re.compile(r"\[[^\]]+\]")


def extract_end_word(text: str) -> str:
    """
    Extract the final lexical word from a bar line.
    """
    cleaned = BRACKET_RE.sub(" ", text)
    tokens = WORD_RE.findall(cleaned.lower())
    if not tokens:
        return ""
    return tokens[-1]


def collect_bar_endings(corpus_path: str) -> Counter:
    """
    Iterate through the cleaned corpus and count end words.
    """
    counts: Counter = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if "[BAR]" not in line:
                continue
            match = BAR_RE.search(line)
            if not match:
                continue
            bar_text = match.group(1)
            end_word = extract_end_word(bar_text)
            if end_word:
                counts[end_word] += 1
    return counts


def rhyme_signature(word: str) -> str | None:
    """
    Pronouncing-based rhyming signature (rhyming_part) if available.
    """
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return None
    part = pronouncing.rhyming_part(phones[0])
    return part or None


def build_pronouncing_lookup(word_to_group: Dict[str, int]) -> Dict[str, Counter]:
    """
    Map pronouncing rhyming_part -> Counter of group ids.
    """
    lookup: Dict[str, Counter] = defaultdict(Counter)
    for word, group in word_to_group.items():
        sig = rhyme_signature(word)
        if not sig:
            continue
        lookup[sig][group] += 1
    return lookup


def rhyme_key(word: str, max_len: int = 4) -> str:
    w = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not w:
        return ""
    return w[-max_len:]


def invert_groups(word_to_group: Dict[str, int]) -> Dict[int, List[str]]:
    by_group: Dict[int, List[str]] = defaultdict(list)
    for word, group in word_to_group.items():
        by_group[group].append(word)
    return by_group


def build_group_embeddings(
    group_to_words: Dict[int, List[str]],
    scorer: SiameseRhymeScorer,
    max_words_per_group: int = 6,
) -> Dict[int, torch.Tensor]:
    """
    Average Siamese embeddings for each group using a few representative words.
    """
    embeddings: Dict[int, torch.Tensor] = {}
    for group, words in group_to_words.items():
        embs = []
        for word in words[:max_words_per_group]:
            emb = scorer.embed(word)
            if emb is None or emb.numel() == 0:
                continue
            embs.append(emb.detach().cpu())
        if embs:
            stacked = torch.stack(embs, dim=0)
            embeddings[group] = stacked.mean(dim=0)
    return embeddings


def assign_with_siamese(
    word: str,
    scorer: SiameseRhymeScorer,
    group_embeddings: Dict[int, torch.Tensor],
    threshold: float,
) -> Tuple[int, float] | None:
    if not group_embeddings:
        return None
    emb = scorer.embed(word)
    if emb is None or emb.numel() == 0:
        return None
    emb = emb.detach().cpu()
    best_group = None
    best_sim = -1.0
    for group_id, ref in group_embeddings.items():
        sim = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), ref.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_group = group_id
    if best_group is None or best_sim < threshold:
        return None
    return best_group, float(best_sim)


def expand_rhyme_groups(
    corpus_path: str,
    existing_csv: str,
    output_csv: str,
    min_count: int = 4,
    max_new_words: int | None = None,
    siamese_model_dir: str | None = None,
    siamese_threshold: float = 0.72,
    rhyme_key_len: int = 4,
) -> Dict[str, int]:
    """
    Main expansion routine. Returns summary stats.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not os.path.exists(existing_csv):
        raise FileNotFoundError(f"Existing rhyme CSV not found: {existing_csv}")

    print(f"[RHYME-UPDATE] Loading existing rhyme CSV: {existing_csv}")
    existing_df = pd.read_csv(existing_csv)
    if "word" not in existing_df.columns or "group" not in existing_df.columns:
        raise ValueError("existing_csv must include 'word' and 'group' columns.")

    if "confidence" not in existing_df.columns:
        existing_df["confidence"] = 1.0
    if "method" not in existing_df.columns:
        existing_df["method"] = "manual"
    if "source_count" not in existing_df.columns:
        existing_df["source_count"] = -1

    word_to_group = {str(row["word"]).strip().lower(): int(row["group"]) for _, row in existing_df.iterrows()}
    if not word_to_group:
        raise ValueError("No entries found in existing rhyme CSV.")

    next_group_id = max(word_to_group.values()) + 1

    print(f"[RHYME-UPDATE] Scanning corpus for end words: {corpus_path}")
    counts = collect_bar_endings(corpus_path)
    print(f"[RHYME-UPDATE] Found {len(counts):,} unique bar endings.")

    new_candidates = [
        (word, freq) for word, freq in counts.items()
        if freq >= min_count and word not in word_to_group
    ]
    new_candidates.sort(key=lambda x: x[1], reverse=True)
    if max_new_words:
        new_candidates = new_candidates[:max_new_words]
    print(f"[RHYME-UPDATE] Considering {len(new_candidates):,} unseen endings with freq >= {min_count}.")

    pron_lookup = build_pronouncing_lookup(word_to_group)
    scorer = None
    group_embeddings: Dict[int, torch.Tensor] = {}
    if siamese_model_dir and os.path.isdir(siamese_model_dir):
        print(f"[RHYME-UPDATE] Loading Siamese encoder from {siamese_model_dir}")
        scorer = SiameseRhymeScorer(siamese_model_dir)
        group_embeddings = build_group_embeddings(invert_groups(word_to_group), scorer)
        print(f"[RHYME-UPDATE] Built embeddings for {len(group_embeddings):,} groups.")
    elif siamese_model_dir:
        print(f"[RHYME-UPDATE][WARN] Siamese dir '{siamese_model_dir}' not found. Skipping similarity stage.")

    rows_to_append = []
    unassigned = {}

    for word, freq in new_candidates:
        assigned = False
        sig = rhyme_signature(word)
        if sig and sig in pron_lookup:
            target_group = pron_lookup[sig].most_common(1)[0][0]
            rows_to_append.append({
                "word": word,
                "group": target_group,
                "confidence": 0.95,
                "method": "pronouncing",
                "source_count": freq,
            })
            assigned = True
        elif scorer is not None:
            siamese_hit = assign_with_siamese(word, scorer, group_embeddings, siamese_threshold)
            if siamese_hit:
                group_id, sim = siamese_hit
                rows_to_append.append({
                    "word": word,
                    "group": group_id,
                    "confidence": float(round(sim, 4)),
                    "method": "siamese",
                    "source_count": freq,
                })
                assigned = True

        if not assigned:
            unassigned[word] = {"freq": freq, "sig": sig}

    print(f"[RHYME-UPDATE] Assigned {len(rows_to_append):,} words via pronouncing/siamese.")
    print(f"[RHYME-UPDATE] {len(unassigned):,} words remain for clustering.")

    buckets: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for word, meta in unassigned.items():
        key = meta["sig"] or rhyme_key(word, rhyme_key_len)
        if not key:
            key = f"UNK_{word[:3]}"
        buckets[key].append((word, meta["freq"]))

    for key, items in buckets.items():
        group_id = next_group_id
        next_group_id += 1
        conf = 0.5 if len(items) > 1 else 0.35
        method = "cluster" if len(items) > 1 else "singleton"
        for word, freq in items:
            rows_to_append.append({
                "word": word,
                "group": group_id,
                "confidence": conf,
                "method": method,
                "source_count": freq,
            })

    print(f"[RHYME-UPDATE] Total new rows to append: {len(rows_to_append):,}")

    if not rows_to_append:
        print("[RHYME-UPDATE] Nothing to append. Existing CSV remains unchanged.")
        return {"appended": 0, "output": existing_csv}

    new_df = pd.DataFrame(rows_to_append)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.sort_values(by=["group", "word"], inplace=True)

    output_path = output_csv or existing_csv
    combined.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[RHYME-UPDATE] Wrote updated rhyme CSV to: {output_path}")

    return {"appended": len(rows_to_append), "output": output_path}


def parse_args():
    p = argparse.ArgumentParser(description="Automatically expand rhymes_grouped.csv from corpus statistics.")
    p.add_argument("--corpus_path", type=str, required=True, help="Path to cleaned elite corpus (TXT).")
    p.add_argument("--existing_csv", type=str, required=True, help="Current rhymes_grouped CSV.")
    p.add_argument("--output_csv", type=str, required=True, help="Where to write the updated CSV.")
    p.add_argument("--min_count", type=int, default=4, help="Minimum frequency for considering a new ending.")
    p.add_argument("--max_new_words", type=int, default=None, help="Optional cap on number of new words to process.")
    p.add_argument(
        "--siamese_model_dir",
        type=str,
        default="/workspace/rap-botV4/rhyme_siamese",
        help="Path to Siamese encoder for similarity clustering.",
    )
    p.add_argument("--siamese_threshold", type=float, default=0.72, help="Cosine similarity threshold for Siamese assignments.")
    p.add_argument("--rhyme_key_len", type=int, default=4, help="Fallback rhyme key length for clustering.")
    return p.parse_args()


def main():
    args = parse_args()
    expand_rhyme_groups(
        corpus_path=args.corpus_path,
        existing_csv=args.existing_csv,
        output_csv=args.output_csv,
        min_count=args.min_count,
        max_new_words=args.max_new_words,
        siamese_model_dir=args.siamese_model_dir,
        siamese_threshold=args.siamese_threshold,
        rhyme_key_len=args.rhyme_key_len,
    )


if __name__ == "__main__":
    main()
