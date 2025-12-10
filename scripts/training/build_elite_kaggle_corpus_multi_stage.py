#!/usr/bin/env python
"""
build_elite_kaggle_corpus_multi_stage.py

Multi-stage pipeline to:

  1) Download the Genius lyrics dataset from Kaggle (10GB-scale).
  2) Pass A (filter): Stream it in chunks, filter to rap / hip-hop tracks
     by a curated list of elite artists, clean lyrics to bars, enforce
     ≥ MIN_BARS, and write reusable top-tier bar-level files, now with
     structural annotations (rhyme group/letter, syllables, internal density).
  3) Pass B (score): Run Siamese rhyme/story coherence scoring on the
     top-tier, cleaned songs ONLY, and write per-song scores.
  4) Pass C (export): Select final elite songs by threshold or top-K,
     and write:
        - CSV: one bar per row + metadata + song_score + annotations.
        - TXT: corpus with <ARTIST=...> tags, annotated [BAR] lines,
          and <END_SONG> markers.
  5) Optional: auto-expand rhymes_grouped.csv with update_rhyme_groups.py
     so rhyme planning inherits new slang discovered in the corpus.

Usage example (all passes):

    python build_elite_kaggle_corpus_multi_stage.py \
      --mode all \
      --download_dir /workspace/data_kaggle \
      --output_csv   /workspace/rap-botV4/data/elite_kaggle_lines_clean.csv \
      --output_txt   /workspace/rap-botV4/data/elite_kaggle_corpus_clean.txt \
      --siamese_model_dir /workspace/rap-botV4/rhyme_siamese \
      --rhyme_groups_csv /workspace/rap-botV4/rhymes_grouped.csv \
      --threshold 0.40 \
      --selection_mode threshold \
      --chunksize 10000 \
      --min_bars 16 \
      --expected_total_rows 5134856 \
      --progress_every 50
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import re
import csv
import glob
import time
import argparse
import subprocess
import zipfile
from typing import List, Dict

import pandas as pd
import torch
import unidecode
import string
import pronouncing
from functools import lru_cache

from rapbot.rhyme_scorer import SiameseRhymeScorer


# ---------------------------------------------------------------------------
# ELITE ARTIST LIST + NORMALISATION / MATCHING
# ---------------------------------------------------------------------------

ELITE_ARTISTS = [
    "Elzhi",
    "Beanie Sigel",
    "Canibus",
    "Prodigy",
    "Masta Ace",
    "Kurupt",
    "Busta Rhymes",
    "Bun B",
    "Pusha T",
    "Chuck D",
    "Talib Kweli",
    "Ab-Soul",
    "Slick Rick",
    "Roc Marciano",
    "AZ",
    "Common",
    "Earl Sweatshirt",
    "Big L",
    "Royce da 5'9",
    "Royce da 5’9″",
    "Lauryn Hill",
    "Styles P",
    "Guru",
    "Jadakiss",
    "Inspectah Deck",
    "Ras Kass",
    "Aesop Rock",
    "Redman",
    "Big Pun",
    "Big Daddy Kane",
    "GZA",
    "Kool Keith",
    "Method Man",
    "2Pac",
    "Tupac",
    "KRS-One",
    "Ghostface Killah",
    "Black Thought",
    "Scarface",
    "Kool G Rap",
    "Lil Wayne",
    "Mos Def",
    "Yasiin Bey",
    "Jay-Z",
    "Jay Z",
    "Pharoahe Monch",
    "The Notorious B.I.G.",
    "Notorious B.I.G.",
    "Notorious BIG",
    "Eminem",
    "Kendrick Lamar",
    "MF DOOM",
    "Nas",
    "Lupe Fiasco",
    "Rakim",
    "Andre 3000",
]


def normalize_name(name: str) -> str:
    """Lowercase, strip, remove weird quotes/punct for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    x = unidecode.unidecode(name).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


TARGET_ARTIST_TOKENS = [normalize_name(a) for a in ELITE_ARTISTS]


def is_elite_artist(artist: str) -> bool:
    """Return True if artist string looks like one of the elite artists."""
    norm = normalize_name(artist)
    if not norm:
        return False
    for tgt in TARGET_ARTIST_TOKENS:
        if not tgt:
            continue
        if tgt in norm or norm in tgt:
            return True
    return False


# ---------------------------------------------------------------------------
# GENRE / LANGUAGE FILTERS
# ---------------------------------------------------------------------------

RAP_TAG_KEYWORDS = [
    "rap",
    "hip hop",
    "hip-hop",
    "hiphop",
    "boom bap",
    "boom-bap",
    "trap",
    "gangsta",
    "grime",
    "drill",
]


def is_rap_track(row: pd.Series) -> bool:
    """
    Heuristic: use tag/genre columns to decide if it's a rap/hip-hop song.
    Also drop obvious non-English entries.
    """
    tag = ""
    for col in ["tag", "genre", "primary_tag", "primary_artist_tags"]:
        if col in row and isinstance(row[col], str):
            tag = row[col]
            break

    tag_l = tag.lower() if isinstance(tag, str) else ""

    lang = ""
    for col in ["language", "language_detected", "lyrics_language"]:
        if col in row and isinstance(row[col], str):
            lang = row[col]
            break
    lang_l = lang.lower() if isinstance(lang, str) else ""

    # Must be English-ish
    if lang_l and not any(k in lang_l for k in ["en", "eng", "english"]):
        return False

    if not tag_l:
        return False

    return any(k in tag_l for k in RAP_TAG_KEYWORDS)


# ---------------------------------------------------------------------------
# LYRICS CLEANING / SPLITTING
# ---------------------------------------------------------------------------

META_LINE_RE = re.compile(
    r"""^
        (verse|chorus|hook|bridge|intro|outro|
         pre[-\s]?chorus|post[-\s]?chorus|refrain)
        [\s\d:.\-]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Lines that are almost certainly credits / publishing notes, not bars.
CREDIT_LINE_RE = re.compile(
    r"""^(
          published\s+by|
          produced\s+by|
          written\s+by|
          recorded\s+at|
          engineered\s+by|
          mixed\s+by|
          mastered\s+by|
          courtesy\s+of|
          copyright\s+\d{4}|
          ℗\s*\d{4}
        )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Lines that look like numbered lists: "1. Song", "2) Track" etc.
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\).:-]\s+")

# Title patterns that are almost never actual songs.
NON_SONG_TITLE_KEYWORDS = [
    "credits",
    "tracklist",
    "discography",
    "liner notes",
    "results",
    "report",
    "review",
    "interview",
    "press release",
    "mixtape credits",
    "album credits",
    "record report",
    "top 50 albums",
    "all-star race",
]


def ascii_ratio(s: str) -> float:
    """Rough heuristic: ratio of ASCII letters/digits/punct to total chars."""
    if not s:
        return 0.0
    allowed = set(string.ascii_letters + string.digits + " '\".,?!-:/&")
    num = sum(1 for ch in s if ch in allowed)
    return num / max(1, len(s))


def clean_lyrics_to_lines(raw_lyrics: str) -> List[str]:
    """
    Clean Genius-style lyrics to a list of bars.
    """
    if not isinstance(raw_lyrics, str):
        return []

    text = unidecode.unidecode(raw_lyrics)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove bracketed section headers e.g., [Chorus], [Verse 1], etc.
    text = re.sub(r"\[.*?\]", " ", text)

    raw_lines = text.split("\n")
    cleaned: List[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Drop obvious structural/meta labels
        if META_LINE_RE.match(line):
            continue

        # Drop {Scratch} / {Annotation} lines
        if line.startswith("{") and line.endswith("}"):
            continue

        # Drop credits / publishing lines
        if CREDIT_LINE_RE.match(line):
            continue

        # Drop numbered list lines (tracklists, album lists, etc.)
        if NUMBERED_LIST_RE.match(line):
            continue

        # Strip inline parentheticals: "(prod by ...)", "(Remix)" etc.
        prev = None
        while prev != line:
            prev = line
            line = re.sub(r"\([^)]*\)", " ", line)
            line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Drop lines that look non-English (very low ASCII ratio)
        if ascii_ratio(line) < 0.5:
            continue

        cleaned.append(line)

    # Compress repeated lines: keep at most 2 identical in a row.
    deduped: List[str] = []
    last = None
    repeat_count = 0
    for ln in cleaned:
        if last is not None and ln == last:
            repeat_count += 1
            if repeat_count >= 2:
                # This is 3rd or later repetition → drop
                continue
        else:
            repeat_count = 0
        deduped.append(ln)
        last = ln

    return deduped


def looks_like_real_song(title: str, lines: List[str]) -> bool:
    """
    Filter out non-song entries like "DROGAS Light Credits",
    "My Top 50 Albums of All Time", race results, etc.
    """
    if not lines:
        return False

    # Safely coerce title to a lowercase string
    if isinstance(title, str):
        t = title.lower()
    else:
        t = ""

    # Title heuristics: drop if title strongly suggests non-song content.
    for kw in NON_SONG_TITLE_KEYWORDS:
        if kw in t:
            return False

    # If first few lines are heavily list/credit-like, treat as non-song.
    list_like = 0
    credit_like = 0
    sample_lines = lines[:40]
    for ln in sample_lines:
        s = ln.strip()
        if not s:
            continue
        if NUMBERED_LIST_RE.match(s):
            list_like += 1
        if CREDIT_LINE_RE.match(s):
            credit_like += 1

        if list_like + credit_like >= max(3, len(sample_lines) // 5):
            return False

    return True


# ---------------------------------------------------------------------------
# STRUCTURAL ANNOTATION HELPERS (RHYME / SYLLABLES / INTERNALS)
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize_words(line: str) -> List[str]:
    return WORD_RE.findall(line.lower())


@lru_cache(maxsize=50000)
def count_syllables_word(word: str) -> int:
    phones = pronouncing.phones_for_word(word)
    if phones:
        return pronouncing.syllable_count(phones[0])
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def count_syllables_line(line: str) -> int:
    words = tokenize_words(line)
    if not words:
        return 0
    return sum(count_syllables_word(w) for w in words)


def syllable_bucket(n: int) -> str:
    if n <= 6:
        return "SYL_0_6"
    elif n <= 8:
        return "SYL_7_8"
    elif n <= 10:
        return "SYL_9_10"
    elif n <= 12:
        return "SYL_11_12"
    elif n <= 14:
        return "SYL_13_14"
    else:
        return "SYL_15_PLUS"


def last_content_word(line: str):
    words = tokenize_words(line)
    if not words:
        return None
    return words[-1]


def load_rhyme_lookup(path: str) -> Dict[str, int]:
    """
    Load word -> rhyme_group_id mapping from rhymes_grouped.csv.
    Expected columns: 'word', 'group'.
    """
    df = pd.read_csv(path)
    lookup: Dict[str, int] = {}
    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        if not w:
            continue
        g = int(row["group"])
        lookup[w] = g
    return lookup


def internal_rhyme_density(line: str, rhyme_lookup: Dict[str, int]) -> str:
    """
    Rough internal rhyme density:
      - Look at all words except last.
      - Map to rhyme groups.
      - Count groups appearing 2+ times.
    """
    words = tokenize_words(line)
    if len(words) <= 2:
        return "INT_NONE"

    groups = []
    for w in words[:-1]:
        gid = rhyme_lookup.get(w)
        if gid is not None:
            groups.append(gid)

    if not groups:
        return "INT_NONE"

    from collections import Counter
    counts = Counter(groups)
    repeated = sum(1 for _, c in counts.items() if c >= 2)

    if repeated >= 3:
        return "INT_DENSE"
    elif repeated >= 1:
        return "INT_MED"
    else:
        return "INT_NONE"


def assign_rhyme_letters(group_ids: List[int]) -> List[str]:
    """
    Map numeric group ids (per song) to letters A, B, C, ...
    group_ids: list of int or None

    Returns list of strings: "A", "B", ... "Z" used as UNKNOWN.
    """
    mapping: Dict[int, str] = {}
    letters: List[str] = []
    next_ord = ord("A")

    for gid in group_ids:
        if gid is None:
            letters.append("Z")  # unknown bucket
            continue
        if gid not in mapping:
            mapping[gid] = chr(next_ord)
            next_ord += 1
            if next_ord > ord("Y"):  # reserve Z for unknown
                next_ord = ord("Y")
        letters.append(mapping[gid])

    return letters


def annotate_song_bars(lines: List[str], rhyme_lookup: Dict[str, int]):
    """
    Given a list of cleaned bars for a song, return parallel lists with:
      - rhyme_group_id
      - rhyme_letter
      - syllable_count
      - syllable_bucket
      - internal_density
    """
    end_group_ids: List[int] = []
    for ln in lines:
        last_w = last_content_word(ln)
        if last_w is None:
            end_group_ids.append(None)
        else:
            end_group_ids.append(rhyme_lookup.get(last_w.lower()))

    rhyme_letters = assign_rhyme_letters(end_group_ids)

    syllable_counts: List[int] = []
    syllable_buckets: List[str] = []
    internal_tags: List[str] = []

    for ln in lines:
        sc = count_syllables_line(ln)
        sb = syllable_bucket(sc)
        it = internal_rhyme_density(ln, rhyme_lookup)
        syllable_counts.append(sc)
        syllable_buckets.append(sb)
        internal_tags.append(it)

    return end_group_ids, rhyme_letters, syllable_counts, syllable_buckets, internal_tags

def compute_siamese_end_rhyme_groups(
    lines: List[str],
    scorer,
    threshold: float = 0.70,
    batch_size: int = 64,
):
    """
    Use the Siamese model to cluster bar endings into rhyme groups.

    For each line:
      - Take last 2-3 words as an 'end phrase'
      - Embed with scorer.embed_batch(...)
      - Compute cosine similarity matrix
      - Greedy clustering: lines with sim >= threshold share a group

    Returns:
      group_ids  : list[int or None]   (cluster id per line)
      rhyme_letters : list[str]        (A/B/C/.../Z per line)
    """
    # 1) Build end phrases
    end_phrases: List[str] = []
    for ln in lines:
        words = tokenize_words(ln)
        if not words:
            end_phrases.append("")        # empty, will likely get its own cluster
        else:
            k = min(3, len(words))
            end_phrase = " ".join(words[-k:])
            end_phrases.append(end_phrase)

    # 2) Get embeddings from Siamese model
    with torch.no_grad():
        embs = scorer.embed_batch(end_phrases, batch_size=batch_size)

    # 3) Move to device + normalize
    if hasattr(scorer, "device"):
        device = scorer.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(embs, torch.Tensor):
        E = embs.to(device)
    else:
        E = torch.tensor(embs, device=device, dtype=torch.float32)

    E = E / (E.norm(dim=1, keepdim=True) + 1e-8)  # [N, D]
    sim = E @ E.T                                  # [N, N] cosine similarity

    n = sim.size(0)
    if n == 0:
        return [], []

    # 4) Greedy clustering based on threshold
    group_ids = [-1] * n
    current_group = 0

    for i in range(n):
        if group_ids[i] != -1:
            continue
        group_ids[i] = current_group
        # All j that rhyme enough with i join the group
        for j in range(i + 1, n):
            if group_ids[j] == -1 and sim[i, j].item() >= threshold:
                group_ids[j] = current_group
        current_group += 1

    # 5) Map numeric groups -> rhyme letters (A/B/C/.../Z)
    rhyme_letters = assign_rhyme_letters(group_ids)

    return group_ids, rhyme_letters


# ---------------------------------------------------------------------------
# SIAMESE SCORE AGGREGATION (VERSE-LEVEL COHERENCE)
# ---------------------------------------------------------------------------

def verse_coherence_score(lines: List[str], scorer, batch_size: int = 32) -> float:
    """
    Compute an average pairwise cosine similarity between bars
    using your Siamese model.

    This version:
      - Uses scorer.embed_batch for embeddings.
      - Moves embeddings to GPU (if available).
      - Computes the pairwise cosine matrix E @ E.T on GPU.
    """
    # Basic cleanup
    lines = [ln for ln in lines if ln and ln.strip()]
    if len(lines) < 2:
        return 0.0

    # 1) Get embeddings from the Siamese scorer
    with torch.no_grad():
        embs = scorer.embed_batch(lines, batch_size=batch_size)

    # 2) Decide device: use scorer.device if present, else cuda/cpu default
    if hasattr(scorer, "device"):
        device = scorer.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Convert embeddings to a tensor on that device
    if isinstance(embs, torch.Tensor):
        E = embs.to(device)
    else:
        # embs is likely a NumPy array
        E = torch.tensor(embs, device=device, dtype=torch.float32)

    # 4) L2-normalize rows: cosine similarity = normalized dot product
    E = E / (E.norm(dim=1, keepdim=True) + 1e-8)

    # 5) Compute full pairwise cosine similarity matrix ON GPU
    sim = E @ E.T  # [N, N]

    n = sim.size(0)
    if n < 2:
        return 0.0

    # 6) Mean of all off-diagonal entries
    total = sim.sum() - torch.diag(sim).sum()
    denom = n * (n - 1)
    score = total / denom

    return float(score.item())


# ---------------------------------------------------------------------------
# KAGGLE DOWNLOAD / UNZIP HELPERS
# ---------------------------------------------------------------------------

def ensure_kaggle_dataset(dataset: str, download_dir: str) -> str:
    """
    Ensure the Kaggle dataset is downloaded and unzipped.
    Returns path to the main CSV file (largest .csv in download_dir).
    """
    os.makedirs(download_dir, exist_ok=True)

    csv_candidates = glob.glob(os.path.join(download_dir, "*.csv"))
    if csv_candidates:
        csv_candidates.sort(key=os.path.getsize, reverse=True)
        print(f"[INFO] Using existing CSV: {csv_candidates[0]}")
        return csv_candidates[0]

    print(f"[INFO] Downloading Kaggle dataset: {dataset}")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        download_dir,
        "--force",
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    zips = glob.glob(os.path.join(download_dir, "*.zip"))
    if not zips:
        raise FileNotFoundError("No .zip file found after Kaggle download.")
    zips.sort(key=os.path.getsize, reverse=True)
    zip_path = zips[0]
    print(f"[INFO] Unzipping {zip_path} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(download_dir)

    csv_candidates = glob.glob(os.path.join(download_dir, "*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No .csv file found after unzipping Kaggle dataset.")

    csv_candidates.sort(key=os.path.getsize, reverse=True)
    main_csv = csv_candidates[0]
    print(f"[INFO] Main CSV detected: {main_csv}")
    return main_csv


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


# ---------------------------------------------------------------------------
# PASS A: FILTER & CLEAN TO TOP-TIER BARS (NOW WITH ANNOTATIONS)
# ---------------------------------------------------------------------------

def run_pass_filter(
    csv_path: str,
    top_tier_bars_csv: str,
    top_tier_meta_csv: str,
    chunksize: int,
    min_bars: int,
    expected_total_rows: int,
    rhyme_lookup: Dict[str, int],
    siamese_scorer,   # NEW
):
    """
    Pass A:
      - Stream raw Kaggle CSV.
      - Filter to rap + elite artists.
      - Clean lyrics, split to bars.
      - Enforce ≥ min_bars.
      - Add structural annotations for each bar.
      - Write:
          * top_tier_meta_csv: song_id, artist, title, num_bars
          * top_tier_bars_csv: song_id, artist, title, line_index, line_text,
                              rhyme_group_id, rhyme_letter,
                              syllable_count, syllable_bucket,
                              internal_density
    """
    print(f"[PASS A] Filtering & cleaning from: {csv_path}")
    print(f"[PASS A] Writing meta to: {top_tier_meta_csv}")
    print(f"[PASS A] Writing bars to: {top_tier_bars_csv}")

    if rhyme_lookup is None:
        raise ValueError(
            "[PASS A] rhyme_lookup is None. Provide --rhyme_groups_csv so we can "
            "annotate bars with rhyme/structure info."
        )

    os.makedirs(os.path.dirname(top_tier_bars_csv), exist_ok=True)
    os.makedirs(os.path.dirname(top_tier_meta_csv), exist_ok=True)

    meta_f = open(top_tier_meta_csv, "w", newline="", encoding="utf-8")
    bars_f = open(top_tier_bars_csv, "w", newline="", encoding="utf-8")

    meta_writer = csv.writer(meta_f)
    bars_writer = csv.writer(bars_f)

    meta_writer.writerow(["song_id", "artist", "title", "num_bars"])
    bars_writer.writerow(
        [
            "song_id",
            "artist",
            "title",
            "line_index",
            "line_text",
            "rhyme_group_id",
            "rhyme_letter",
            "syllable_count",
            "syllable_bucket",
            "internal_density",
        ]
    )

    total_rows = 0
    total_songs_considered = 0
    total_songs_kept = 0
    total_bars_written = 0

    start_time = time.time()

    chunk_iter = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_idx, df_chunk in enumerate(chunk_iter, start=1):
        rows_in_chunk = len(df_chunk)
        total_rows += rows_in_chunk

        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
        eta_sec = (
            (expected_total_rows - total_rows) / rows_per_sec
            if rows_per_sec > 0 and expected_total_rows > total_rows
            else 0
        )
        pct = (
            100.0 * total_rows / expected_total_rows
            if expected_total_rows > 0
            else 0.0
        )

        print(
            f"[PASS A][CHUNK {chunk_idx}] rows={rows_in_chunk}, "
            f"total_rows={total_rows} ({pct:.2f}% of expected {expected_total_rows}), "
            f"elapsed={format_eta(elapsed)}, ETA={format_eta(eta_sec)}"
        )

        # Required columns check
        for col in ["artist", "lyrics", "title"]:
            if col not in df_chunk.columns:
                print(
                    f"[WARN] Column '{col}' not found in chunk {chunk_idx}; "
                    "skipping this chunk."
                )
                df_chunk = pd.DataFrame(columns=df_chunk.columns)
                break

        for idx, row in df_chunk.iterrows():
            artist = row.get("artist", "")
            title = row.get("title", "")
            lyrics = row.get("lyrics", "")

            if not isinstance(lyrics, str) or not lyrics.strip():
                continue

            # Genre / language filter
            if not is_rap_track(row):
                continue

            # Restrict to elite artists
            if not is_elite_artist(artist):
                continue

            lines = clean_lyrics_to_lines(lyrics)
            if len(lines) < min_bars:
                continue

            if not looks_like_real_song(title, lines):
                continue

            total_songs_considered += 1

            song_id = f"chunk{chunk_idx}_row{idx}"
            num_bars = len(lines)

            # 1) Siamese-based rhyme clusters for bar endings
            rhyme_group_ids, rhyme_letters = compute_siamese_end_rhyme_groups(
                lines,
                scorer=siamese_scorer,
                threshold=0.70,        # tune as needed
                batch_size=64,
            )
            
            # 2) Syllables + internal density (can still use rhyme_lookup here if you like)
            syllable_counts = []
            syllable_buckets = []
            internal_tags = []
            
            for ln in lines:
                sc = count_syllables_line(ln)
                sb = syllable_bucket(sc)
                it = internal_rhyme_density(ln, rhyme_lookup) if rhyme_lookup is not None else "INT_NONE"
                syllable_counts.append(sc)
                syllable_buckets.append(sb)
                internal_tags.append(it)
    
            meta_writer.writerow([song_id, artist, title, num_bars])

            for li, bar in enumerate(lines):
                bar_clean = bar.strip()
                if not bar_clean:
                    continue
                bars_writer.writerow(
                    [
                        song_id,
                        artist,
                        title,
                        li,
                        bar_clean,
                        rhyme_group_ids[li],
                        rhyme_letters[li],
                        syllable_counts[li],
                        syllable_buckets[li],
                        internal_tags[li],
                    ]
                )
                total_bars_written += 1

            total_songs_kept += 1

        print(
            f"[PASS A][CHUNK {chunk_idx} DONE] "
            f"top-tier songs so far={total_songs_kept}, "
            f"total_bars_written={total_bars_written}"
        )

    meta_f.close()
    bars_f.close()

    elapsed = time.time() - start_time
    print("\n[PASS A SUMMARY]")
    print(f"  Total raw CSV rows scanned     : {total_rows}")
    print(f"  Total candidate songs (elite+) : {total_songs_considered}")
    print(f"  Total songs kept (>= {min_bars} bars) : {total_songs_kept}")
    print(f"  Total bars written             : {total_bars_written}")
    print(f"  top_tier_meta_csv path         : {top_tier_meta_csv}")
    print(f"  top_tier_bars_csv path         : {top_tier_bars_csv}")
    print(f"  Elapsed                        : {format_eta(elapsed)}")


# ---------------------------------------------------------------------------
# PASS B: SIAMESE SCORING ON TOP-TIER BARS
# ---------------------------------------------------------------------------

def run_pass_score(
    top_tier_bars_csv: str,
    scores_csv: str,
    siamese_model_dir: str,
    batch_size: int,
    progress_every: int,
):
    """
    Pass B:
      - Stream top_tier_bars_csv grouped by song_id (assumes songs are
        contiguous because we wrote them that way in Pass A).
      - Run Siamese coherence scoring on all bars per song.
      - Write scores_csv with: song_id, artist, title, num_bars, song_score.
    """
    if not os.path.exists(top_tier_bars_csv):
        raise FileNotFoundError(
            f"[PASS B] top_tier_bars_csv not found: {top_tier_bars_csv} "
            "(run --mode filter first)"
        )

    print(f"[PASS B] Scoring from bars file: {top_tier_bars_csv}")
    print(f"[PASS B] Writing scores to: {scores_csv}")
    print(f"[PASS B] Loading Siamese model from: {siamese_model_dir}")

    scorer = SiameseRhymeScorer(siamese_model_dir)
    
    # Ensure model is on GPU if available (depends on your implementation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(scorer, "to"):
        scorer.to(device)
        scorer.device = device
    elif hasattr(scorer, "model"):
        scorer.model.to(device)
        scorer.device = device
    else:
        scorer.device = device  # verse_coherence_score will still use this device
    
    print(f"[PASS B] Siamese model ready on device: {scorer.device}")


    os.makedirs(os.path.dirname(scores_csv), exist_ok=True)
    scores_f = open(scores_csv, "w", newline="", encoding="utf-8")
    scores_writer = csv.writer(scores_f)
    scores_writer.writerow(["song_id", "artist", "title", "num_bars", "song_score"])

    # Stream through bars CSV and group by song_id
    with open(top_tier_bars_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError("[PASS B] Bars CSV appears to be empty.")

        # Determine indices
        try:
            idx_song_id = header.index("song_id")
            idx_artist = header.index("artist")
            idx_title = header.index("title")
            idx_line_text = header.index("line_text")
        except ValueError as e:
            raise RuntimeError(
                "[PASS B] Expected columns 'song_id','artist','title','line_text' "
                f"in bars CSV. Got: {header}"
            ) from e

        current_song_id = None
        current_artist = None
        current_title = None
        current_lines: List[str] = []
        song_count = 0

        def flush_current():
            nonlocal song_count
            if current_song_id is None or not current_lines:
                return
            song_count += 1
            try:
                score = verse_coherence_score(current_lines, scorer, batch_size=batch_size)
            except Exception as e:
                print(
                    f"[WARN] Siamese scoring failed for '{current_artist} - {current_title}': {e}"
                )
                return
            num_bars = len(current_lines)
            scores_writer.writerow(
                [current_song_id, current_artist, current_title, num_bars, f"{score:.6f}"]
            )
            if song_count % progress_every == 0:
                print(
                    f"[PASS B] scored {song_count} songs; "
                    f"last: {current_artist} - {current_title} score={score:.3f}"
                )

        for row in reader:
            song_id = row[idx_song_id]
            artist = row[idx_artist]
            title = row[idx_title]
            line_text = row[idx_line_text]

            if current_song_id is None:
                current_song_id = song_id
                current_artist = artist
                current_title = title
                current_lines = [line_text]
                continue

            if song_id == current_song_id:
                current_lines.append(line_text)
            else:
                # flush previous song
                flush_current()
                # start new song
                current_song_id = song_id
                current_artist = artist
                current_title = title
                current_lines = [line_text]

        # Flush the last song
        flush_current()

    scores_f.close()
    print(f"[PASS B SUMMARY] Scored songs written to: {scores_csv}")


# ---------------------------------------------------------------------------
# PASS C: SELECT ELITE SONGS AND EXPORT FINAL CORPUS
# ---------------------------------------------------------------------------

def run_pass_export(
    top_tier_bars_csv: str,
    scores_csv: str,
    elite_csv: str,
    elite_txt: str,
    threshold: float,
    selection_mode: str,
    target_num_songs: int,
):
    """
    Pass C:
      - Load song scores from scores_csv.
      - Select songs by threshold OR top-K.
      - Stream bars file again, keeping only selected songs.
      - Write:
          * elite_csv: artist, title, song_id, song_score, line_index, line_text,
                       rhyme_group_id, rhyme_letter, syllable_count,
                       syllable_bucket, internal_density
          * elite_txt: tagged training corpus for LM with [BAR]/[RHY]/[SYL]/[INT].
    """
    if not os.path.exists(top_tier_bars_csv):
        raise FileNotFoundError(
            f"[PASS C] top_tier_bars_csv not found: {top_tier_bars_csv} "
            "(run --mode filter first)"
        )
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(
            f"[PASS C] scores_csv not found: {scores_csv} "
            "(run --mode score first)"
        )

    print(f"[PASS C] Loading song scores from: {scores_csv}")
    scores: List[Dict[str, str]] = []
    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["num_bars"] = int(row["num_bars"])
                row["song_score"] = float(row["song_score"])
            except Exception:
                continue
            scores.append(row)

    if not scores:
        raise RuntimeError("[PASS C] No scores found; cannot export elite corpus.")

    # Selection
    if selection_mode == "top_k":
        # sort by score descending, keep top target_num_songs
        scores_sorted = sorted(scores, key=lambda r: r["song_score"], reverse=True)
        keep = scores_sorted[:target_num_songs]
        keep_ids = {r["song_id"] for r in keep}
        print(
            f"[PASS C] selection_mode=top_k, target_num_songs={target_num_songs}, "
            f"actual_kept={len(keep_ids)}, min_score_kept={keep[-1]['song_score']:.3f}"
        )
    else:
        # threshold-based
        keep_ids = {r["song_id"] for r in scores if r["song_score"] >= threshold}
        min_kept_score = min(
            (r["song_score"] for r in scores if r["song_id"] in keep_ids),
            default=None,
        )

        # Safe string for logging
        min_score_str = f"{min_kept_score:.3f}" if min_kept_score is not None else "N/A"

        print(
            f"[PASS C] selection_mode=threshold, threshold={threshold:.3f}, "
            f"kept_songs={len(keep_ids)}, "
            f"min_score_kept={min_score_str}"
        )

    # Build score lookup for export
    score_by_song_id = {r["song_id"]: r["song_score"] for r in scores}

    # Prepare outputs
    os.makedirs(os.path.dirname(elite_csv), exist_ok=True)
    os.makedirs(os.path.dirname(elite_txt), exist_ok=True)

    elite_f_csv = open(elite_csv, "w", newline="", encoding="utf-8")
    elite_writer = csv.writer(elite_f_csv)
    elite_writer.writerow(
        [
            "artist",
            "title",
            "song_id",
            "song_score",
            "line_index",
            "line_text",
            "rhyme_group_id",
            "rhyme_letter",
            "syllable_count",
            "syllable_bucket",
            "internal_density",
        ]
    )

    elite_f_txt = open(elite_txt, "w", encoding="utf-8")

    print(f"[PASS C] Streaming bars from: {top_tier_bars_csv}")
    print(f"[PASS C] Writing elite CSV to: {elite_csv}")
    print(f"[PASS C] Writing elite TXT to: {elite_txt}")

    kept_song_count = 0
    kept_bar_count = 0

    with open(top_tier_bars_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError("[PASS C] Bars CSV appears to be empty.")

        idx_song_id = header.index("song_id")
        idx_artist = header.index("artist")
        idx_title = header.index("title")
        idx_line_index = header.index("line_index")
        idx_line_text = header.index("line_text")
        idx_rhyme_group_id = header.index("rhyme_group_id")
        idx_rhyme_letter = header.index("rhyme_letter")
        idx_syllable_count = header.index("syllable_count")
        idx_syllable_bucket = header.index("syllable_bucket")
        idx_internal_density = header.index("internal_density")

        current_song_id = None
        current_in_keep = False
        current_artist = None
        current_title = None

        def close_song():
            nonlocal kept_song_count
            if current_in_keep:
                elite_f_txt.write("<END_SONG>\n\n")
                kept_song_count += 1

        for row in reader:
            song_id = row[idx_song_id]
            if song_id not in keep_ids:
                # If we are leaving a kept song, close it.
                if current_song_id is not None and current_in_keep and song_id != current_song_id:
                    close_song()
                    current_in_keep = False
                current_song_id = song_id
                continue

            artist = row[idx_artist]
            title = row[idx_title]
            line_index = row[idx_line_index]
            line_text = row[idx_line_text]
            rhyme_group_id = row[idx_rhyme_group_id]
            rhyme_letter = row[idx_rhyme_letter]
            syllable_count = row[idx_syllable_count]
            syllable_bucket = row[idx_syllable_bucket]
            internal_density = row[idx_internal_density]
            song_score = score_by_song_id.get(song_id, 0.0)

            # If we are entering a new kept song, write header meta line
            if song_id != current_song_id:
                # close previous if needed
                if current_song_id is not None and current_in_keep:
                    close_song()
                current_song_id = song_id
                current_in_keep = True
                current_artist = artist
                current_title = title

                meta_line = (
                    f"<ARTIST={artist}> <TITLE={title}> "
                    f"<SOURCE=genius_kaggle> <COHERENCE={song_score:.3f}>"
                )
                elite_f_txt.write(meta_line + "\n")

            # Write annotated bar to txt
            annotated_line = (
                f"[BAR] {line_text} "
                f"[RHY={rhyme_letter}] "
                f"[{syllable_bucket}] "
                f"[{internal_density}]"
            )
            elite_f_txt.write(annotated_line + "\n")

            # Write bar + annotations to csv
            elite_writer.writerow(
                [
                    artist,
                    title,
                    song_id,
                    f"{song_score:.6f}",
                    line_index,
                    line_text,
                    rhyme_group_id,
                    rhyme_letter,
                    syllable_count,
                    syllable_bucket,
                    internal_density,
                ]
            )
            kept_bar_count += 1

        # Close final song if needed
        if current_song_id is not None and current_in_keep:
            close_song()

    elite_f_csv.close()
    elite_f_txt.close()

    print("\n[PASS C SUMMARY]")
    print(f"  Elite songs kept      : {kept_song_count}")
    print(f"  Elite bars written    : {kept_bar_count}")
    print(f"  Elite CSV path        : {elite_csv}")
    print(f"  Elite TXT path        : {elite_txt}")


# ---------------------------------------------------------------------------
# ARGPARSE / ENTRYPOINT
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build elite, fully cleaned rap corpus from Kaggle Genius dataset (multi-stage)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "filter", "score", "export"],
        help="Which stages to run: all | filter | score | export.",
    )
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        default="carlosgdcj/genius-song-lyrics-with-language-information",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        required=True,
        help="Directory to download/unzip the Kaggle dataset into.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to FINAL elite CSV (one clean bar per row, with song_score and annotations).",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        required=True,
        help="Path to FINAL elite TXT corpus for training.",
    )
    parser.add_argument(
        "--siamese_model_dir",
        type=str,
        default="./rhyme_siamese",
        help="Directory containing trained Siamese model.",
    )
    parser.add_argument(
        "--rhyme_groups_csv",
        type=str,
        default=None,
        help="Path to rhymes_grouped.csv (word -> rhyme group) "
             "for structural annotation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Minimum song_score to keep a song in selection_mode=threshold.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="threshold",
        choices=["threshold", "top_k"],
        help="How to select final elite songs: 'threshold' or 'top_k'.",
    )
    parser.add_argument(
        "--target_num_songs",
        type=int,
        default=3000,
        help="If selection_mode=top_k, number of top-scoring songs to keep.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="pandas.read_csv chunksize (controls memory usage).",
    )
    parser.add_argument(
        "--min_bars",
        type=int,
        default=16,
        help="Minimum number of lyric lines required for a song in Pass A.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="In Pass B, log every N scored songs.",
    )
    parser.add_argument(
        "--expected_total_rows",
        type=int,
        default=5134856,
        help=(
            "Expected total rows in the CSV. Used only for ETA. "
            "For the Genius dataset this is about 5,134,856."
        ),
    )
    parser.add_argument(
        "--top_tier_bars_csv",
        type=str,
        default=None,
        help="Optional path for intermediate top-tier bars CSV. "
             "If not set, derived from output_csv.",
    )
    parser.add_argument(
        "--top_tier_meta_csv",
        type=str,
        default=None,
        help="Optional path for intermediate top-tier song meta CSV. "
             "If not set, derived from output_csv.",
    )
    parser.add_argument(
        "--scores_csv",
        type=str,
        default=None,
        help="Optional path for intermediate song scores CSV. "
             "If not set, derived from output_csv.",
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=32,
        help="Batch size for Siamese embed_batch in Pass B.",
    )
    parser.add_argument(
        "--auto_expand_rhyme_groups",
        action="store_true",
        help="After export, run update_rhyme_groups.py to refresh rhymes_grouped.csv.",
    )
    parser.add_argument(
        "--rhyme_update_corpus",
        type=str,
        default=None,
        help="Optional override for the corpus fed to the rhyme updater (defaults to output_txt).",
    )
    parser.add_argument(
        "--rhyme_update_min_count",
        type=int,
        default=4,
        help="Minimum frequency for a new ending when auto-expanding rhyme groups.",
    )
    parser.add_argument(
        "--rhyme_update_max_new",
        type=int,
        default=None,
        help="Optional cap on the number of endings to examine when auto-expanding.",
    )
    parser.add_argument(
        "--rhyme_update_output",
        type=str,
        default=None,
        help="Optional destination for the expanded rhyme CSV (defaults to --rhyme_groups_csv).",
    )
    parser.add_argument(
        "--rhyme_update_threshold",
        type=float,
        default=0.72,
        help="Siamese similarity threshold used during auto-expansion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Derive intermediate paths if not provided
    base_csv = os.path.splitext(args.output_csv)[0]
    top_tier_bars_csv = args.top_tier_bars_csv or (base_csv + "_top_tier_bars.csv")
    top_tier_meta_csv = args.top_tier_meta_csv or (base_csv + "_top_tier_song_meta.csv")
    scores_csv = args.scores_csv or (base_csv + "_rhyme_scores.csv")

    csv_path = ensure_kaggle_dataset(
        dataset=args.kaggle_dataset,
        download_dir=args.download_dir,
    )

    rhyme_lookup = None
    if args.mode in ("all", "filter"):
        if not args.rhyme_groups_csv:
            raise ValueError(
                "--rhyme_groups_csv is required when running mode 'all' or 'filter'."
            )
        print(f"[GLOBAL] Loading rhyme groups from: {args.rhyme_groups_csv}")
        rhyme_lookup = load_rhyme_lookup(args.rhyme_groups_csv)
        print(f"[GLOBAL] Loaded {len(rhyme_lookup):,} rhyme entries.")

        print(f"[GLOBAL] Loading Siamese model for rhyme clustering from: {args.siamese_model_dir}")
        siamese_for_rhymes = SiameseRhymeScorer(args.siamese_model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(siamese_for_rhymes, "to"):
            siamese_for_rhymes.to(device)
            siamese_for_rhymes.device = device
        elif hasattr(siamese_for_rhymes, "model"):
            siamese_for_rhymes.model.to(device)
            siamese_for_rhymes.device = device
        else:
            siamese_for_rhymes.device = device
        print(f"[GLOBAL] Siamese rhyme model ready on device: {siamese_for_rhymes.device}")


    if args.mode in ("all", "filter"):
        run_pass_filter(
            csv_path=csv_path,
            top_tier_bars_csv=top_tier_bars_csv,
            top_tier_meta_csv=top_tier_meta_csv,
            chunksize=args.chunksize,
            min_bars=args.min_bars,
            expected_total_rows=args.expected_total_rows,
            rhyme_lookup=rhyme_lookup,
            siamese_scorer=siamese_for_rhymes,
        )


    if args.mode in ("all", "score"):
        run_pass_score(
            top_tier_bars_csv=top_tier_bars_csv,
            scores_csv=scores_csv,
            siamese_model_dir=args.siamese_model_dir,
            batch_size=args.score_batch_size,
            progress_every=args.progress_every,
        )

    if args.mode in ("all", "export"):
        run_pass_export(
            top_tier_bars_csv=top_tier_bars_csv,
            scores_csv=scores_csv,
            elite_csv=args.output_csv,
            elite_txt=args.output_txt,
            threshold=args.threshold,
            selection_mode=args.selection_mode,
            target_num_songs=args.target_num_songs,
        )

    if args.auto_expand_rhyme_groups:
        if not args.rhyme_groups_csv:
            print("[AUTO-RHYME][WARN] --rhyme_groups_csv is required to auto-expand rhyme entries.")
        else:
            corpus_for_update = args.rhyme_update_corpus or args.output_txt
            if not corpus_for_update or not os.path.exists(corpus_for_update):
                print(
                    "[AUTO-RHYME][WARN] Corpus for rhyme update is missing. "
                    "Provide --rhyme_update_corpus or ensure --output_txt exists."
                )
            else:
                try:
                    from scripts.tools.update_rhyme_groups import expand_rhyme_groups
                except ImportError as exc:
                    print(f"[AUTO-RHYME][ERROR] Failed to import update_rhyme_groups: {exc}")
                else:
                    output_csv = args.rhyme_update_output or args.rhyme_groups_csv
                    print(
                        f"[AUTO-RHYME] Expanding rhyme groups using corpus {corpus_for_update} "
                        f"-> {output_csv}"
                    )
                    expand_rhyme_groups(
                        corpus_path=corpus_for_update,
                        existing_csv=args.rhyme_groups_csv,
                        output_csv=output_csv,
                        min_count=args.rhyme_update_min_count,
                        max_new_words=args.rhyme_update_max_new,
                        siamese_model_dir=args.siamese_model_dir,
                        siamese_threshold=args.rhyme_update_threshold,
                    )

if __name__ == "__main__":
    main()
