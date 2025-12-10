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
) -> Dict[int, List[str]]:
    """
    Plan anchor end-words per line.

    Strategy:
      1) For each rhyme letter (A, B, C...), pick ONE rhyme group.
      2) All lines with that letter share that group, but can use
         different words from it as anchors.
    """
    scheme_letters = parse_scheme(scheme)
    topic_words = extract_topic_words(seed)  # currently unused, but ready for future bias

    # 1) Pre-filter groups that have enough words to be useful
    candidate_groups = [g for g, ws in group_to_words.items() if len(ws) >= 4]
    if not candidate_groups:
        raise RuntimeError("No rhyme groups with >= 4 words found in rhyme CSV.")

    # 2) Map letter -> chosen group
    import random
    rnd = random.Random(42)  # deterministic for debugging

    unique_letters = sorted(set(scheme_letters))
    letter_to_group: Dict[str, str] = {}
    for letter in unique_letters:
        letter_to_group[letter] = rnd.choice(candidate_groups)

    # 3) For each line index, sample anchor words from that letter's group
    anchor_words_per_line: Dict[int, List[str]] = {}
    used_words = set()

    for idx, letter in enumerate(scheme_letters):
        group_id = letter_to_group[letter]
        pool = group_to_words[group_id][:]  # copy
        rnd.shuffle(pool)

        anchors: List[str] = []
        for w in pool:
            if w in used_words:
                continue
            anchors.append(w)
            used_words.add(w)
            if len(anchors) >= num_candidates_per_group:
                break

        # If we ran out of new words, allow reuse
        if not anchors:
            anchors = group_to_words[group_id][:num_candidates_per_group]

        anchor_words_per_line[idx] = anchors

    return anchor_words_per_line
