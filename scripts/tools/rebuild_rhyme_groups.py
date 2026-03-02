#!/usr/bin/env python
"""
rebuild_rhyme_groups.py

Rebuild rhyme groups from scratch using phonetic analysis.
This tool creates a clean rhymes_grouped.csv without relying on existing data.

Methodology:
- Use pronouncing library for phonetic analysis
- Cluster words by phonetic similarity
- Output clean CSV with confidence scores
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import pronouncing
import numpy as np
from sklearn.cluster import DBSCAN

# Import phonetic functions from update_rhyme_groups if available
try:
    from scripts.tools.update_rhyme_groups import (
        extract_last_syllable,
        phonetic_similarity,
        ARPA_VOWELS,
        VOWEL_GROUPS,
        CONSONANT_GROUPS,
    )
except ImportError:
    # Define minimal versions if import fails
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
    
    def strip_stress(phone: str) -> str:
        return re.sub(r"\d", "", phone.upper())
    
    @dataclass
    class PhoneticFeature:
        vowel: str
        stress: int
        coda: Tuple[str, ...]
        vowel_group: str
        raw: Tuple[str, ...]
    
    PHONETIC_CACHE: Dict[str, Optional[PhoneticFeature]] = {}
    
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
                vowel_group = None
                for idx, group in enumerate(VOWEL_GROUPS):
                    if vowel in group:
                        vowel_group = f"VG{idx}"
                        break
                if vowel_group is None:
                    vowel_group = vowel
                
                feature = PhoneticFeature(
                    vowel=vowel,
                    stress=stress,
                    coda=tuple(coda) if coda else tuple(),
                    vowel_group=vowel_group,
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
        for group in VOWEL_GROUPS:
            if v1 in group and v2 in group:
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
    
    def consonant_group_key(phone: str) -> str:
        upper = strip_stress(phone)
        for idx, group in enumerate(CONSONANT_GROUPS):
            if upper in group:
                return f"CG{idx}"
        return upper
    
    def consonant_similarity(c1: Tuple[str, ...], c2: Tuple[str, ...]) -> float:
        if not c1 and not c2:
            return 0.5
        if not c1 or not c2:
            return 0.3
        if c1 == c2:
            return 1.0
        head1 = c1[0] if c1 else ""
        head2 = c2[0] if c2 else ""
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


WORD_RE = re.compile(r"[A-Za-z']+")


def collect_words_from_corpus(corpus_path: Path, min_freq: int = 1) -> Counter:
    """
    Collect words from corpus (if provided).
    Returns word frequency counter.
    """
    if not corpus_path.exists():
        return Counter()
    
    counts = Counter()
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                words = WORD_RE.findall(line.lower())
                for word in words:
                    if len(word) >= 2:  # Skip very short words
                        counts[word] += 1
    except Exception as e:
        print(f"[WARN] Failed to read corpus: {e}")
    
    # Filter by minimum frequency
    return Counter({w: c for w, c in counts.items() if c >= min_freq})


def build_rhyme_groups_phonetic(
    words: List[str],
    similarity_threshold: float = 0.6,
    min_group_size: int = 2,
) -> Dict[str, int]:
    """
    Build rhyme groups using phonetic similarity clustering.
    Returns: word -> group_id mapping
    """
    print(f"Building rhyme groups for {len(words)} words...")
    
    # Extract phonetic features
    features: Dict[str, Optional[PhoneticFeature]] = {}
    valid_words = []
    
    for word in words:
        feat = extract_last_syllable(word)
        features[word] = feat
        if feat is not None:
            valid_words.append(word)
    
    print(f"  Extracted phonetic features for {len(valid_words)} words")
    
    if len(valid_words) < 2:
        print("[WARN] Not enough words with phonetic features")
        return {}
    
    # Build distance matrix
    n = len(valid_words)
    print(f"  Building distance matrix ({n}x{n})...")
    dist_matrix = np.ones((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i + 1, n):
            word1, word2 = valid_words[i], valid_words[j]
            feat1, feat2 = features[word1], features[word2]
            sim = phonetic_similarity(feat1, feat2)
            dist = 1.0 - sim
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    # Cluster using DBSCAN
    print(f"  Clustering with threshold {similarity_threshold}...")
    eps = 1.0 - similarity_threshold
    model = DBSCAN(eps=eps, min_samples=min_group_size, metric="precomputed")
    labels = model.fit_predict(dist_matrix)
    
    # Build word -> group mapping
    word_to_group: Dict[str, int] = {}
    label_to_group: Dict[int, int] = {}
    next_group_id = 0
    
    for label in set(labels):
        if label == -1:  # Noise points
            continue
        if label not in label_to_group:
            label_to_group[label] = next_group_id
            next_group_id += 1
    
    for idx, label in enumerate(labels):
        if label != -1:
            word = valid_words[idx]
            group_id = label_to_group[label]
            word_to_group[word] = group_id
    
    # Assign noise points to singleton groups
    for idx, label in enumerate(labels):
        if label == -1:
            word = valid_words[idx]
            word_to_group[word] = next_group_id
            next_group_id += 1
    
    print(f"  Created {len(set(word_to_group.values()))} rhyme groups")
    print(f"  Assigned {len(word_to_group)} words to groups")
    
    return word_to_group


def build_rhyme_groups_pronouncing(words: List[str]) -> Dict[str, int]:
    """
    Build rhyme groups using pronouncing library's rhyming_part.
    Simpler but less flexible than phonetic clustering.
    """
    print(f"Building rhyme groups using pronouncing for {len(words)} words...")
    
    rhyme_signature_to_words: Dict[str, List[str]] = defaultdict(list)
    
    for word in words:
        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            rhyme_part = pronouncing.rhyming_part(phones[0])
            if rhyme_part:
                rhyme_signature_to_words[rhyme_part].append(word)
    
    # Assign group IDs
    word_to_group: Dict[str, int] = {}
    next_group_id = 0
    
    for rhyme_sig, group_words in rhyme_signature_to_words.items():
        if len(group_words) >= 1:  # Include singletons
            for word in group_words:
                word_to_group[word] = next_group_id
            next_group_id += 1
    
    # Assign words without rhyme signatures to singleton groups
    for word in words:
        if word not in word_to_group:
            word_to_group[word] = next_group_id
            next_group_id += 1
    
    print(f"  Created {len(set(word_to_group.values()))} rhyme groups")
    return word_to_group


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild rhyme groups from scratch using phonetic analysis"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output rhymes_grouped.csv",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Optional corpus file to extract words from",
    )
    parser.add_argument(
        "--word_list",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of words to include",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["phonetic", "pronouncing", "both"],
        default="phonetic",
        help="Method to use for grouping",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.6,
        help="Minimum phonetic similarity for grouping (0.0-1.0)",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help="Minimum words per group (singletons allowed if 1)",
    )
    parser.add_argument(
        "--min_word_freq",
        type=int,
        default=1,
        help="Minimum frequency for words from corpus",
    )
    
    args = parser.parse_args()
    
    # Collect words
    words_set = set()
    
    if args.word_list:
        words_set.update(w.lower() for w in args.word_list)
        print(f"Added {len(args.word_list)} words from --word_list")
    
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        if corpus_path.exists():
            word_counts = collect_words_from_corpus(corpus_path, args.min_word_freq)
            words_set.update(word_counts.keys())
            print(f"Added {len(word_counts)} unique words from corpus")
        else:
            print(f"[WARN] Corpus file not found: {corpus_path}")
    
    if not words_set:
        print("[ERROR] No words to process. Provide --corpus_path or --word_list")
        return 1
    
    words = sorted(list(words_set))
    print(f"Processing {len(words)} unique words")
    print()
    
    # Build rhyme groups
    if args.method in ("phonetic", "both"):
        word_to_group = build_rhyme_groups_phonetic(
            words,
            similarity_threshold=args.similarity_threshold,
            min_group_size=args.min_group_size,
        )
    elif args.method == "pronouncing":
        word_to_group = build_rhyme_groups_pronouncing(words)
    else:
        print(f"[ERROR] Unknown method: {args.method}")
        return 1
    
    if not word_to_group:
        print("[ERROR] Failed to build rhyme groups")
        return 1
    
    # Write CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for word, group_id in sorted(word_to_group.items(), key=lambda x: (x[1], x[0])):
        rows.append({
            "word": word,
            "group": group_id,
            "confidence": 0.75,  # Default confidence for rebuilt groups
            "method": args.method,
            "source_count": 1,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total words: {len(word_to_group)}")
    print(f"Total groups: {len(set(word_to_group.values()))}")
    
    # Group size statistics
    group_sizes = Counter(word_to_group.values())
    sizes_list = list(group_sizes.values())
    if sizes_list:
        print(f"Average group size: {sum(sizes_list) / len(sizes_list):.2f}")
        print(f"Largest group: {max(sizes_list)} words")
        print(f"Smallest group: {min(sizes_list)} words")
    
    print()
    print("[OK] Rhyme groups rebuilt successfully")
    print()
    print("Next steps:")
    print("  1. Validate the output: python scripts/tools/validate_rhyme_data.py --rhyme_csv", output_path)
    print("  2. Use the new CSV in rhyme detection")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
