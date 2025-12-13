# rhyme_planner.py
import csv
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

WORD_RE = re.compile(r"[A-Za-z']+")


def load_rhyme_groups(csv_path: str) -> Dict[str, List[str]]:
    """
    Returns:
        group_to_words: dict[group_id(str) -> list[word(str)]]

    Compatible with either:
      - word,group
      - word,rhyme_group
    """
    group_to_words: Dict[str, List[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # detect which column name is used
        fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]
        if "rhyme_group" in fieldnames:
            group_col = next(fn for fn in reader.fieldnames if fn.lower() == "rhyme_group")
        elif "group" in fieldnames:
            group_col = next(fn for fn in reader.fieldnames if fn.lower() == "group")
        else:
            raise KeyError("No 'group' or 'rhyme_group' column found in rhyme CSV.")

        word_col = next(fn for fn in reader.fieldnames if fn.lower() == "word")

        for row in reader:
            w = row[word_col].strip().lower()
            g = str(row[group_col]).strip()
            if w and g:
                group_to_words[g].append(w)

    return group_to_words


def parse_scheme(scheme: str) -> List[str]:
    scheme = scheme.replace(" ", "").upper()
    return list(scheme) if scheme else ["A"]


def extract_topic_words(seed: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(seed)]


def plan_rhyme_endings(
    seed: str,
    scheme: str,
    group_to_words: Dict[str, List[str]],
    num_candidates_per_group: int = 6,
    per_letter_group_count: int = 2,
    random_seed: int | None = None,
) -> Dict[int, List[str]]:
    """
    Plan anchor end-words per line.

    Strategy:
      1) For each rhyme letter (A, B, C...), allocate a small pool of rhyme groups.
      2) As that letter repeats, rotate through its pool so anchors cycle across groups.
      3) For each line, sample distinct words from the chosen group's vocabulary.
    """
    scheme_letters = parse_scheme(scheme)
    topic_words = extract_topic_words(seed)  # currently unused, but ready for future bias

    # 1) Pre-filter groups that have enough words to be useful
    candidate_groups = [g for g, ws in group_to_words.items() if len(ws) >= 4]
    if not candidate_groups:
        raise RuntimeError("No rhyme groups with >= 4 words found in rhyme CSV.")

    # 2) Map letter -> multiple candidate groups (rotated later)
    import random

    if random_seed is None:
        random_seed = (hash((seed.lower(), scheme.upper())) & 0xFFFFFFFF)
    rnd = random.Random(random_seed)
    per_letter_group_count = max(1, int(per_letter_group_count))

    shuffled_groups = candidate_groups[:]
    rnd.shuffle(shuffled_groups)
    cursor = 0

    def next_group() -> str:
        nonlocal cursor, shuffled_groups
        if not shuffled_groups:
            shuffled_groups = candidate_groups[:]
            rnd.shuffle(shuffled_groups)
            cursor = 0
        if cursor >= len(shuffled_groups):
            rnd.shuffle(shuffled_groups)
            cursor = 0
        group_id = shuffled_groups[cursor]
        cursor += 1
        return group_id

    unique_letters = sorted(set(scheme_letters))
    letter_to_group_pool: Dict[str, List[str]] = {}
    for letter in unique_letters:
        pool: List[str] = []
        attempts = 0
        needed = min(per_letter_group_count, len(candidate_groups))
        while len(pool) < needed and attempts < len(candidate_groups) * 2:
            gid = next_group()
            if gid not in pool:
                pool.append(gid)
            attempts += 1
        if not pool:
            pool = [next_group()]
        letter_to_group_pool[letter] = pool

    # 3) For each line index, sample anchor words from that letter's group
    anchor_words_per_line: Dict[int, List[str]] = {}
    letter_counts = defaultdict(int)

    for idx, letter in enumerate(scheme_letters):
        pool = letter_to_group_pool[letter]
        occurrence = letter_counts[letter]
        group_id = pool[occurrence % len(pool)]
        letter_counts[letter] += 1

        group_words = group_to_words.get(group_id, [])
        if not group_words:
            anchor_words_per_line[idx] = []
            continue

        unique_words = list(dict.fromkeys(group_words))
        if len(unique_words) <= num_candidates_per_group:
            anchors = unique_words[:]
            rnd.shuffle(anchors)
        else:
            anchors = rnd.sample(unique_words, num_candidates_per_group)

        anchor_words_per_line[idx] = anchors

    return anchor_words_per_line
