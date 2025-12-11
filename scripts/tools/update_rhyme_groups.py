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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Any

import pandas as pd
import pronouncing
import torch
import numpy as np
from sklearn.cluster import DBSCAN

from rapbot.rhyme_scorer import SiameseRhymeScorer


WORD_RE = re.compile(r"[A-Za-z']+")
BAR_RE = re.compile(r"\[BAR\](.*)")
BRACKET_RE = re.compile(r"\[[^\]]+\]")

ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

VOWEL_GROUPS = [
    {"AA", "AO", "AH"},
    {"AE", "EH", "EY"},
    {"IH", "IY"},
    {"OW", "UW", "UH"},
    {"ER"},
    {"AW", "AY", "OY"},
]

@dataclass
class GroupProfile:
    group_id: int
    words: List[str]
    features: List[PhoneticFeature]

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.words = []
        self.features = []

    def add(self, word: str, feature: Optional[PhoneticFeature]):
        self.words.append(word)
        if feature:
            self.features.append(feature)

VOWEL_NEIGHBORS: Dict[str, set] = {}
for group in VOWEL_GROUPS:
    for vowel in group:
        VOWEL_NEIGHBORS[vowel] = set(group)
CONSONANT_GROUPS = [
    {"B", "P"},
    {"D", "T"},
    {"G", "K"},
    {"S", "Z", "SH", "ZH"},
    {"F", "V"},
    {"CH", "JH"},
    {"M", "N", "NG"},
    {"L", "R"},
]

PHONETIC_CACHE: Dict[str, Optional["PhoneticFeature"]] = {}


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone.upper())


def vowel_group_key(vowel: str) -> str:
    upper = str(vowel).upper()
    for idx, group in enumerate(VOWEL_GROUPS):
        if upper in group:
            return f"VG{idx}"
    return upper


def consonant_group_key(phone: str) -> str:
    upper = strip_stress(phone)
    for idx, group in enumerate(CONSONANT_GROUPS):
        if upper in group:
            return f"CG{idx}"
    return upper


@dataclass
class PhoneticFeature:
    vowel: str
    stress: int
    coda: Tuple[str, ...]
    vowel_group: str
    raw: Tuple[str, ...]


def extract_last_syllable(word: str) -> Optional[PhoneticFeature]:
    word = str(word).lower()
    if word in PHONETIC_CACHE:
        return PHONETIC_CACHE[word]

    phones = pronouncing.phones_for_word(word)
    feature: Optional[PhoneticFeature] = None
    for ph in phones:
        tokens = ph.split()
        vowel = None
        stress = 0
        coda: List[str] = []
        for idx in range(len(tokens) - 1, -1, -1):
            token = tokens[idx]
            base = strip_stress(token)
            if base in ARPA_VOWELS and vowel is None:
                vowel = base
                if token[-1].isdigit():
                    stress = int(token[-1])
                coda = [strip_stress(t) for t in tokens[idx + 1 :] if strip_stress(t) not in ARPA_VOWELS]
                break
        if vowel:
            feature = PhoneticFeature(
                vowel=vowel,
                stress=stress,
                coda=tuple(coda) if coda else tuple(),
                vowel_group=vowel_group_key(vowel),
                raw=tuple(tokens),
            )
            break
    PHONETIC_CACHE[word] = feature
    return feature


def vowel_similarity(v1: Optional[str], v2: Optional[str]) -> float:
    if not v1 or not v2:
        return 0.0
    if v1 == v2:
        return 1.0
    key1 = vowel_group_key(v1)
    key2 = vowel_group_key(v2)
    if key1 == key2:
        return 0.75
    return 0.25


def stress_similarity(s1: int, s2: int) -> float:
    if s1 == s2:
        if s1 > 0:
            return 1.0
        return 0.7
    if s1 > 0 and s2 > 0:
        return 0.8
    if s1 == 0 and s2 == 0:
        return 0.6
    return 0.35


def consonant_similarity(c1: Tuple[str, ...], c2: Tuple[str, ...]) -> float:
    if not c1 and not c2:
        return 0.5
    if not c1 or not c2:
        return 0.3
    if c1 == c2:
        return 1.0
    head1 = c1[0]
    head2 = c2[0]
    if head1 == head2:
        return 0.8
    if consonant_group_key(head1) == consonant_group_key(head2):
        return 0.6
    return 0.3


def phonetic_similarity(f1: Optional[PhoneticFeature], f2: Optional[PhoneticFeature]) -> float:
    if not f1 or not f2:
        return 0.0
    sigma_v = vowel_similarity(f1.vowel, f2.vowel)
    if sigma_v <= 0:
        return 0.0
    sigma_s = stress_similarity(f1.stress, f2.stress)
    sigma_c = consonant_similarity(f1.coda, f2.coda)
    base = 0.5 * sigma_v + 0.2 * sigma_s + 0.3 * sigma_c
    return float(base * sigma_v)


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


def suffix_overlap(a: str, b: str) -> int:
    a_clean = re.sub(r"[^a-z]", "", str(a).lower())
    b_clean = re.sub(r"[^a-z]", "", str(b).lower())
    max_len = min(len(a_clean), len(b_clean))
    count = 0
    for i in range(1, max_len + 1):
        if a_clean[-i:] == b_clean[-i:]:
            count = i
        else:
            break
    return count


def assign_with_siamese(
    word: str,
    scorer: SiameseRhymeScorer,
    group_embeddings: Dict[int, torch.Tensor],
    threshold: float,
    group_to_words: Dict[int, List[str]],
    min_suffix_overlap: int,
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
    if min_suffix_overlap > 0:
        candidates = group_to_words.get(best_group, [])
        clean_word = re.sub(r"[^a-z]", "", word.lower())
        if clean_word and len(clean_word) > min_suffix_overlap and candidates:
            overlaps = [
                suffix_overlap(word, existing)
                for existing in candidates
                if existing and existing != word
            ]
            max_overlap = max(overlaps) if overlaps else 0
            if max_overlap < min_suffix_overlap:
                return None
    return best_group, float(best_sim)


def assign_with_phonetics(
    word: str,
    feature_cache: Dict[str, Optional[PhoneticFeature]],
    group_profiles: Dict[int, GroupProfile],
    vowel_index: Dict[str, set],
    threshold: float,
    max_refs: int = 6,
) -> Tuple[int, float] | None:
    word_lower = word.lower()
    feat = feature_cache.get(word_lower)
    if feat is None:
        feat = extract_last_syllable(word_lower)
        feature_cache[word_lower] = feat
    if not feat or not feat.vowel:
        return None
    candidate_groups = set(vowel_index.get(feat.vowel, []))
    for neighbor in VOWEL_NEIGHBORS.get(feat.vowel, []):
        candidate_groups.update(vowel_index.get(neighbor, []))
    if not candidate_groups:
        candidate_groups = set(group_profiles.keys())
    best_group = None
    best_score = 0.0
    for group_id in candidate_groups:
        profile = group_profiles.get(group_id)
        if not profile or not profile.features:
            continue
        for other_feat in profile.features[:max_refs]:
            score = phonetic_similarity(feat, other_feat)
            if score > best_score:
                best_score = score
                best_group = group_id
    if best_group is None or best_score < threshold:
        return None
    return best_group, best_score


def expand_rhyme_groups(
    corpus_path: str,
    existing_csv: str,
    output_csv: str,
    min_count: int = 4,
    max_new_words: int | None = None,
    siamese_model_dir: str | None = None,
    siamese_threshold: float = 0.72,
    min_suffix_overlap: int = 2,
    phonetic_threshold: float = 0.55,
    cluster_eps: float = 0.45,
    cluster_min_samples: int = 2,
    rhyme_key_len: int = 4,
    reassign_manual: bool = False,
    full_recluster: bool = False,
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

    manual_candidates: List[Tuple[str, int]] = []
    if full_recluster:
        print(f"[RHYME-UPDATE] Full recluster requested. Seeding with {len(existing_df):,} existing entries.")
        for _, row in existing_df.iterrows():
            word = str(row["word"]).strip().lower()
            if not word:
                continue
            freq = int(row.get("source_count", -1))
            if freq <= 0:
                freq = max(min_count, 1)
            manual_candidates.append((word, freq))
        existing_df = existing_df.iloc[0:0].copy()
    elif reassign_manual:
        mask = existing_df["method"].astype(str).str.lower() == "manual"
        manual_df = existing_df[mask]
        if not manual_df.empty:
            print(f"[RHYME-UPDATE] Reassigning {len(manual_df):,} manual entries.")
            for _, row in manual_df.iterrows():
                word = str(row["word"]).strip().lower()
                if not word:
                    continue
                freq = int(row.get("source_count", -1))
                if freq <= 0:
                    freq = max(min_count, 1)
                manual_candidates.append((word, freq))
            existing_df = existing_df[~mask].copy()
        else:
            print("[RHYME-UPDATE] No manual entries found to reassign.")

    word_to_group = {str(row["word"]).strip().lower(): int(row["group"]) for _, row in existing_df.iterrows()}
    if not word_to_group and not manual_candidates and not full_recluster:
        raise ValueError("No entries found in existing rhyme CSV after removing manual entries.")

    next_group_id = max(word_to_group.values(), default=-1) + 1

    print(f"[RHYME-UPDATE] Scanning corpus for end words: {corpus_path}")
    counts = collect_bar_endings(corpus_path)
    print(f"[RHYME-UPDATE] Found {len(counts):,} unique bar endings.")

    new_candidates = [
        (word, freq) for word, freq in counts.items()
        if freq >= min_count and word not in word_to_group
    ]
    if manual_candidates:
        manual_candidates = [(w, freq) for w, freq in manual_candidates if w not in word_to_group]
        print(f"[RHYME-UPDATE] Added {len(manual_candidates):,} manual words back into candidate list.")
        new_candidates = manual_candidates + new_candidates
    new_candidates.sort(key=lambda x: x[1], reverse=True)
    if max_new_words:
        new_candidates = new_candidates[:max_new_words]
    print(f"[RHYME-UPDATE] Considering {len(new_candidates):,} unseen endings with freq >= {min_count}.")

    feature_cache, group_profiles, vowel_index = build_group_profiles(word_to_group)

    def cluster_all_words(words: List[Tuple[str, int]]) -> Dict[str, Tuple[int, float]]:
        nonlocal next_group_id
        assignments: Dict[str, Tuple[int, float]] = {}
        signature_map: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
        for word, _ in words:
            lower = word.lower()
            feat = feature_cache.get(lower)
            if feat is None:
                feat = extract_last_syllable(lower)
                feature_cache[lower] = feat
            clean_tail = re.sub(r"[^a-z]", "", lower)
            tail = clean_tail[-rhyme_key_len:] if clean_tail else lower[-rhyme_key_len:]
            if feat:
                key = (
                    feat.vowel_group,
                    feat.stress,
                    tuple(feat.coda[:2]),
                    tail,
                )
            else:
                key = ("suffix", tail)
            signature_map[key].append(word)
        for key, group_words in signature_map.items():
            group_id = next_group_id
            next_group_id += 1
            for word in group_words:
                assignments[word] = (group_id, 0.75)
        return assignments

    if not word_to_group:
        print("[RHYME-UPDATE] No existing groups; performing pure phonetic clustering.")
        assignments = cluster_all_words(new_candidates)
        rows_to_append = []
        for word, freq in new_candidates:
            group_id, conf = assignments[word]
            rows_to_append.append({
                "word": word,
                "group": group_id,
                "confidence": conf,
                "method": "cluster_phonetic",
                "source_count": freq,
            })
        combined = pd.DataFrame(rows_to_append)
        combined.sort_values(by=["group", "word"], inplace=True)
        combined.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[RHYME-UPDATE] Wrote rebuilt rhyme CSV to: {output_csv}")
        return {"appended": len(rows_to_append), "output": output_csv}

    pron_lookup = build_pronouncing_lookup(word_to_group)
    group_to_words = invert_groups(word_to_group)
    scorer = None
    group_embeddings: Dict[int, torch.Tensor] = {}
    if siamese_model_dir and os.path.isdir(siamese_model_dir):
        print(f"[RHYME-UPDATE] Loading Siamese encoder from {siamese_model_dir}")
        scorer = SiameseRhymeScorer(siamese_model_dir)
        group_embeddings = build_group_embeddings(group_to_words, scorer)
        print(f"[RHYME-UPDATE] Built embeddings for {len(group_embeddings):,} groups.")
    elif siamese_model_dir:
        print(f"[RHYME-UPDATE][WARN] Siamese dir '{siamese_model_dir}' not found. Skipping similarity stage.")

    rows_to_append = []
    unassigned = {}

    for word, freq in new_candidates:
        assigned = False

        if word_to_group:
            phonetic_hit = assign_with_phonetics(
                word,
                feature_cache,
                group_profiles,
                vowel_index,
                phonetic_threshold,
            )
            if phonetic_hit:
                group_id, score = phonetic_hit
                rows_to_append.append({
                    "word": word,
                    "group": group_id,
                    "confidence": float(round(score, 4)),
                    "method": "phonetic",
                    "source_count": freq,
                })
                assigned = True
                word_to_group[word] = group_id
                group_to_words.setdefault(group_id, []).append(word)
                update_group_profile(group_id, word, group_profiles, vowel_index, feature_cache)
                continue

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
            word_to_group[word] = target_group
            group_to_words.setdefault(target_group, []).append(word)
            update_group_profile(target_group, word, group_profiles, vowel_index, feature_cache)
        elif scorer is not None and group_embeddings:
            siamese_hit = assign_with_siamese(
                word,
                scorer,
                group_embeddings,
                siamese_threshold,
                group_to_words,
                min_suffix_overlap,
            )
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
                word_to_group[word] = group_id
                group_to_words.setdefault(group_id, []).append(word)
                update_group_profile(group_id, word, group_profiles, vowel_index, feature_cache)

        if not assigned:
            unassigned[word] = {"freq": freq, "sig": sig}

    print(f"[RHYME-UPDATE] Assigned {len(rows_to_append):,} words via pronouncing/siamese.")
    print(f"[RHYME-UPDATE] {len(unassigned):,} words remain for clustering.")

    # Attempt phonetic clustering using DBSCAN
    def run_phonetic_clustering(words: List[str]) -> Dict[str, Tuple[int, float]]:
        nonlocal next_group_id
        assignments: Dict[str, Tuple[int, float]] = {}
        if not words:
            return assignments
        buckets_by_vowel: Dict[str, List[str]] = defaultdict(list)
        for w in words:
            feat = feature_cache.get(w) or extract_last_syllable(w)
            feature_cache[w] = feat
            if feat and feat.vowel:
                buckets_by_vowel[feat.vowel].append(w)
        nonlocal next_group_id
        for bucket, bucket_words in buckets_by_vowel.items():
            if len(bucket_words) < max(cluster_min_samples, 2):
                continue
            feats = [feature_cache[w] for w in bucket_words]
            valid_indices = [i for i, f in enumerate(feats) if f]
            if len(valid_indices) < max(cluster_min_samples, 2):
                continue
            # Build distance matrix only for valid subset
            sub_words = [bucket_words[i] for i in valid_indices]
            sub_feats = [feats[i] for i in valid_indices]
            n = len(sub_words)
            dist = np.ones((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = phonetic_similarity(sub_feats[i], sub_feats[j])
                    dist[i, j] = dist[j, i] = 1.0 - sim
            model = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric="precomputed")
            labels = model.fit_predict(dist)
            label_to_group: Dict[int, int] = {}
            for label in set(labels):
                if label == -1:
                    continue
                indices = [idx for idx, l in enumerate(labels) if l == label]
                if len(indices) < cluster_min_samples:
                    continue
                if label not in label_to_group:
                    label_to_group[label] = next_group_id
                    next_group_id += 1
                group_id = label_to_group[label]
                sims = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        ii = indices[i]
                        jj = indices[j]
                        sims.append(1.0 - dist[ii, jj])
                conf = float(np.mean(sims)) if sims else 0.6
                for idx in indices:
                    assignments[sub_words[idx]] = (group_id, conf)
        return assignments

    phonetic_cluster_assignments = run_phonetic_clustering(list(unassigned.keys()))
    for word, (group_id, conf) in phonetic_cluster_assignments.items():
        rows_to_append.append({
            "word": word,
            "group": group_id,
            "confidence": max(0.5, round(conf, 4)),
            "method": "cluster_phonetic",
            "source_count": unassigned[word]["freq"],
        })
        word_to_group[word] = group_id
        group_to_words.setdefault(group_id, []).append(word)
        update_group_profile(group_id, word, group_profiles, vowel_index, feature_cache)
        unassigned.pop(word, None)

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
        group_to_words.setdefault(group_id, [])
        for word, freq in items:
            rows_to_append.append({
                "word": word,
                "group": group_id,
                "confidence": conf,
                "method": method,
                "source_count": freq,
            })
            word_to_group[word] = group_id
            group_to_words[group_id].append(word)

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
        default=str(ROOT / "rhyme_siamese"),
        help="Path to Siamese encoder for similarity clustering.",
    )
    p.add_argument("--siamese_threshold", type=float, default=0.72, help="Cosine similarity threshold for Siamese assignments.")
    p.add_argument(
        "--min_suffix_overlap",
        type=int,
        default=2,
        help="Minimum shared trailing characters with an existing group member when using Siamese assignments.",
    )
    p.add_argument(
        "--phonetic_threshold",
        type=float,
        default=0.55,
        help="Minimum phonetic similarity score to join an existing group.",
    )
    p.add_argument(
        "--cluster_eps",
        type=float,
        default=0.45,
        help="DBSCAN epsilon for phonetic clustering of new rhyme endings.",
    )
    p.add_argument(
        "--cluster_min_samples",
        type=int,
        default=2,
        help="Minimum samples for DBSCAN phonetic clustering.",
    )
    p.add_argument("--rhyme_key_len", type=int, default=4, help="Fallback rhyme key length for clustering.")
    p.add_argument(
        "--reassign_manual",
        action="store_true",
        help="Drop existing manual entries and reassign them via pronouncing/siamese/cluster rules.",
    )
    p.add_argument(
        "--full_recluster",
        action="store_true",
        help="Ignore existing group assignments and rebuild all rhyme groups from scratch.",
    )
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
        min_suffix_overlap=args.min_suffix_overlap,
        phonetic_threshold=args.phonetic_threshold,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
        rhyme_key_len=args.rhyme_key_len,
        reassign_manual=args.reassign_manual,
        full_recluster=args.full_recluster,
    )
def build_group_profiles(word_to_group: Dict[str, int]) -> Tuple[Dict[str, Optional[PhoneticFeature]], Dict[int, GroupProfile], Dict[str, set]]:
    feature_cache: Dict[str, Optional[PhoneticFeature]] = {}
    profiles: Dict[int, GroupProfile] = {}
    vowel_index: Dict[str, set] = defaultdict(set)
    for word, group in word_to_group.items():
        profile = profiles.setdefault(group, GroupProfile(group))
        feat = extract_last_syllable(word)
        feature_cache[word] = feat
        profile.add(word, feat)
        if feat and feat.vowel:
            vowel_index[feat.vowel].add(group)
    return feature_cache, profiles, vowel_index


def update_group_profile(
    group_id: int,
    word: str,
    group_profiles: Dict[int, GroupProfile],
    vowel_index: Dict[str, set],
    feature_cache: Dict[str, Optional[PhoneticFeature]],
):
    word_lower = word.lower()
    feat = feature_cache.get(word_lower)
    if feat is None:
        feat = extract_last_syllable(word_lower)
        feature_cache[word_lower] = feat
    profile = group_profiles.setdefault(group_id, GroupProfile(group_id))
    profile.add(word, feat)
    if feat and feat.vowel:
        vowel_index[feat.vowel].add(group_id)


if __name__ == "__main__":
    main()
