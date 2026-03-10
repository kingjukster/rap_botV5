"""
evo_rhyme/generator.py

Seed couplet generation from templates and vocab. Loads rhyme groups from
rapbot.rhyme_scorer for end-word pairing. Builds tail_to_words (group_to_words)
index from rhymes_grouped.csv for rhyme targeting in mutations.

Also supports corpus-based random couplets for inject_random_immigrants.
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from evo_rhyme.individual import CoupletIndividual, analyze_individual

try:
    from rapbot.rhyme_scorer import RHYME_GROUPS, load_rhyme_groups
except ImportError:
    RHYME_GROUPS: Dict[str, int] = {}
    load_rhyme_groups = None  # type: ignore

# ---------------------------------------------------------------------------
# Paths for template/vocab seed generation
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
EVO_DATA = ROOT / "data" / "evo_rhyme"
VOCAB_DIR = EVO_DATA / "vocab"
TEMPLATES_PATH = EVO_DATA / "templates.txt"
RHYMES_CSV_PATH = ROOT / "data" / "rhymes_grouped.csv"

# ---------------------------------------------------------------------------
# Minimal seed data (used when vocab files don't exist)
# ---------------------------------------------------------------------------

_MINIMAL_NOUNS = [
    "pressure", "mask", "room", "truth", "path", "life", "way", "night",
    "light", "right", "back", "time", "fight", "mind", "game", "name",
]
_MINIMAL_VERBS = [
    "burn", "walk", "run", "fight", "talk", "move", "flow", "grow",
    "show", "know", "go", "hold", "break", "make", "take", "rise",
]
_MINIMAL_VERBS_PAST = [
    "burned", "walked", "ran", "fought", "talked", "moved", "flowed",
    "grew", "showed", "knew", "went", "held", "broke", "made", "took", "rose",
]
_MINIMAL_ADJECTIVES = [
    "dark", "bright", "right", "tight", "light", "high", "deep", "real",
    "true", "new", "cold", "bold", "strong", "long", "wrong", "young",
]
_MINIMAL_STATES = [
    "fire", "mode", "line", "time", "zone", "flow", "glow", "truth",
    "path", "lane", "game", "name", "pain", "chain", "rain",
]

# ---------------------------------------------------------------------------
# Template/vocab loaders for seed generation
# ---------------------------------------------------------------------------


def _ensure_vocab_files() -> None:
    """Create minimal vocab files if they don't exist."""
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    if not (VOCAB_DIR / "nouns.txt").exists():
        (VOCAB_DIR / "nouns.txt").write_text("\n".join(_MINIMAL_NOUNS), encoding="utf-8")
    if not (VOCAB_DIR / "verbs.txt").exists():
        (VOCAB_DIR / "verbs.txt").write_text("\n".join(_MINIMAL_VERBS), encoding="utf-8")
    if not (VOCAB_DIR / "verbs_past.txt").exists():
        (VOCAB_DIR / "verbs_past.txt").write_text("\n".join(_MINIMAL_VERBS_PAST), encoding="utf-8")
    if not (VOCAB_DIR / "adjectives.txt").exists():
        (VOCAB_DIR / "adjectives.txt").write_text("\n".join(_MINIMAL_ADJECTIVES), encoding="utf-8")
    if not (VOCAB_DIR / "states.txt").exists():
        (VOCAB_DIR / "states.txt").write_text("\n".join(_MINIMAL_STATES), encoding="utf-8")
    if not TEMPLATES_PATH.exists():
        _minimal_templates = [
            "{noun} in the {noun} while the {noun2} still {verb}",
            "I {verb_past} through the {noun} with my {noun} on {state}",
            "The {noun} got {adjective} when the {noun2} won't {verb}",
        ]
        TEMPLATES_PATH.write_text("\n".join(_minimal_templates), encoding="utf-8")


def _load_lines(path: Path) -> List[str]:
    """Load non-empty stripped lines from a text file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


_IRREGULAR_ING: Dict[str, str] = {
    "be": "being", "is": "being", "are": "being",
    "see": "seeing", "flee": "fleeing", "agree": "agreeing",
    "lie": "lying", "die": "dying", "tie": "tying", "vie": "vying",
}


def _verb_to_ing(verb: str) -> str:
    """Derive present participle from base verb."""
    v = verb.lower().strip()
    if not v:
        return ""
    if v in _IRREGULAR_ING:
        return _IRREGULAR_ING[v]
    if len(v) > 2 and v.endswith("e"):
        return v[:-1] + "ing"
    if len(v) >= 3:
        c, vc, cc = v[-3], v[-2], v[-1]
        if cc in "bdfgklmnprst" and vc in "aeiou" and c not in "aeiou":
            return v + cc + "ing"
    return v + "ing"


def load_templates() -> List[str]:
    """Load templates from data/evo_rhyme/templates.txt. Creates minimal file if missing."""
    _ensure_vocab_files()
    return _load_lines(TEMPLATES_PATH)


def load_vocab() -> Dict[str, List[str]]:
    """
    Load vocab from data/evo_rhyme/vocab/. Creates minimal files if missing.
    Returns dict with keys: nouns, verbs, adjectives, verbs_past, verbs_ing, states.
    """
    _ensure_vocab_files()
    verbs = _load_lines(VOCAB_DIR / "verbs.txt") or _MINIMAL_VERBS
    verbs_ing = _load_lines(VOCAB_DIR / "verbs_ing.txt")
    if not verbs_ing:
        # Derive -ing from verbs if verbs_ing.txt missing
        verbs_ing = [_verb_to_ing(v) for v in verbs]
    vocab: Dict[str, List[str]] = {
        "nouns": _load_lines(VOCAB_DIR / "nouns.txt") or _MINIMAL_NOUNS,
        "verbs": verbs,
        "adjectives": _load_lines(VOCAB_DIR / "adjectives.txt") or _MINIMAL_ADJECTIVES,
        "verbs_past": _load_lines(VOCAB_DIR / "verbs_past.txt") or _MINIMAL_VERBS_PAST,
        "verbs_ing": verbs_ing,
        "states": _load_lines(VOCAB_DIR / "states.txt") or _MINIMAL_STATES,
    }
    return vocab


# ---------------------------------------------------------------------------
# tail_to_words / group_to_words index from rhymes_grouped.csv
# ---------------------------------------------------------------------------

_GROUP_TO_WORDS: Optional[Dict[int, List[str]]] = None


def _build_group_to_words() -> Dict[int, List[str]]:
    """
    Build group_to_words index from rhymes_grouped.csv.
    Maps group_id -> list of unique words in that rhyme group.
    Used for rhyme targeting in seed generation and mutations.
    """
    global _GROUP_TO_WORDS
    if _GROUP_TO_WORDS is not None:
        return _GROUP_TO_WORDS

    group_to_words: Dict[int, List[str]] = defaultdict(list)
    seen: Set[tuple] = set()  # (group_id, word) to avoid duplicates

    if not RHYMES_CSV_PATH.exists():
        _GROUP_TO_WORDS = dict(group_to_words)
        return _GROUP_TO_WORDS

    with open(RHYMES_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip().lower()
            try:
                group = int(row.get("group", -1))
            except (ValueError, TypeError):
                continue
            if not word or group < 0:
                continue
            key = (group, word)
            if key not in seen:
                seen.add(key)
                group_to_words[group].append(word)

    _GROUP_TO_WORDS = {k: list(v) for k, v in group_to_words.items()}
    return _GROUP_TO_WORDS


def get_tail_to_words() -> Dict[int, List[str]]:
    """
    Return group_id -> words index for rhyme targeting.
    Alias for get_group_to_words(); named tail_to_words for consistency with mutation API.
    """
    return _build_group_to_words()


def get_group_to_words() -> Dict[int, List[str]]:
    """Return group_id -> list of words from rhymes_grouped.csv."""
    return _build_group_to_words()


# ---------------------------------------------------------------------------
# Template parsing and filling
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _parse_template(template: str) -> List[tuple]:
    """Return list of (placeholder_name, start, end) for each {placeholder} in template."""
    return [(m.group(1), m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(template)]


def _slot_mapping() -> Dict[str, str]:
    """Map placeholder names to vocab keys."""
    return {
        "noun": "nouns",
        "noun2": "nouns",
        "noun3": "nouns",
        "verb": "verbs",
        "verb_past": "verbs_past",
        "verb_ing": "verbs_ing",
        "adjective": "adjectives",
        "state": "states",
    }


def _pick_word(
    vocab: Dict[str, List[str]],
    slot: str,
    theme_keywords: Set[str],
    rhyme_group_words: Optional[Set[str]] = None,
) -> str:
    """Pick a word for the given slot, preferring theme matches."""
    key = _slot_mapping().get(slot, slot)
    words = vocab.get(key, vocab.get("nouns", []))
    if not words:
        return "thing"

    if rhyme_group_words:
        candidates = [w for w in words if w.lower() in rhyme_group_words]
        if not candidates:
            candidates = list(rhyme_group_words)[:20]
        if not candidates:
            candidates = words
    else:
        candidates = words

    if theme_keywords:
        themed = [w for w in candidates if w.lower() in theme_keywords]
        if themed:
            return random.choice(themed)

    return random.choice(candidates)


def _fill_template(
    template: str,
    vocab: Dict[str, List[str]],
    theme_keywords: Set[str],
    end_word: Optional[str] = None,
) -> str:
    """Fill template placeholders. If end_word is provided, use it for the last placeholder."""
    placeholders = _parse_template(template)
    if not placeholders:
        return template

    result = template
    for i in range(len(placeholders) - 1, -1, -1):
        name, start, end = placeholders[i]
        if end_word is not None and i == len(placeholders) - 1:
            word = end_word
        else:
            word = _pick_word(vocab, name, theme_keywords)
        result = result[:start] + word + result[end:]

    return result


def generate_seed_couplets(
    theme_keywords: Optional[List[str]] = None,
    count: int = 10,
    rhyme_targets: Optional[List[str]] = None,
    analyze: bool = True,
) -> List[CoupletIndividual]:
    """
    Generate seed couplets from templates with theme-aware word filling.
    End-word pairs are picked from the same rhyme group (from rapbot.rhyme_scorer).

    Args:
        theme_keywords: Optional list of theme words to prefer when filling slots.
        count: Number of couplets to generate.
        rhyme_targets: Optional list of words; end-word pairs chosen from
            words in rhyme_targets that share a rhyme group. If None, any rhyme group.
        analyze: If True, populate features1/features2 via analyze_individual.

    Returns:
        List of CoupletIndividual instances.
    """
    templates = load_templates()
    vocab = load_vocab()
    group_to_words = get_group_to_words()
    theme_set = set(w.lower() for w in (theme_keywords or []))

    if not templates:
        return []

    rhyme_groups = RHYME_GROUPS if RHYME_GROUPS else {}
    if load_rhyme_groups and not rhyme_groups and RHYMES_CSV_PATH.exists():
        rhyme_groups = load_rhyme_groups(RHYMES_CSV_PATH, validate=False)

    viable_groups = [
        gid for gid, words in group_to_words.items()
        if len(words) >= 2
    ]

    if not viable_groups:
        all_vocab = set()
        for v in vocab.values():
            all_vocab.update(w.lower() for w in v)
        for gid, words in group_to_words.items():
            in_vocab = [w for w in words if w in all_vocab]
            if len(in_vocab) >= 2:
                viable_groups.append(gid)

    if rhyme_targets:
        target_set = set(w.lower() for w in rhyme_targets)
        target_groups = [
            gid for gid in viable_groups
            if any(w in target_set for w in group_to_words.get(gid, []))
        ]
        if target_groups:
            viable_groups = target_groups

    results: List[CoupletIndividual] = []
    for _ in range(count):
        t1 = random.choice(templates)
        t2 = random.choice(templates)
        ph1 = _parse_template(t1)
        ph2 = _parse_template(t2)
        slot1 = ph1[-1][0] if ph1 else "noun"
        slot2 = ph2[-1][0] if ph2 else "noun"
        key1 = _slot_mapping().get(slot1, "nouns")
        key2 = _slot_mapping().get(slot2, "nouns")

        if viable_groups:
            gid = random.choice(viable_groups)
            words_in_group = group_to_words[gid]
            if rhyme_targets:
                target_set = set(w.lower() for w in rhyme_targets)
                candidates = [w for w in words_in_group if w in target_set]
                if len(candidates) < 2:
                    candidates = words_in_group
            else:
                candidates = words_in_group

            # Only use rhyme group words that are in our vocab for the slot type
            v1 = set(w.lower() for w in vocab.get(key1, []))
            v2 = set(w.lower() for w in vocab.get(key2, []))
            fit1 = [w for w in candidates if w in v1]
            fit2 = [w for w in candidates if w in v2]
            if fit1 and fit2:
                end1 = random.choice(fit1)
                end2 = random.choice([w for w in fit2 if w != end1] or fit2)
            else:
                # No vocab overlap: use random vocab words (coherent but may not rhyme)
                end1 = _pick_word(vocab, slot1, theme_set)
                end2 = _pick_word(vocab, slot2, theme_set)
        else:
            end1 = _pick_word(vocab, slot1, theme_set)
            end2 = _pick_word(vocab, slot2, theme_set)

        line1 = _fill_template(t1, vocab, theme_set, end_word=end1)
        line2 = _fill_template(t2, vocab, theme_set, end_word=end2)

        ind = CoupletIndividual(line1=line1, line2=line2)
        if analyze:
            analyze_individual(ind)
        results.append(ind)

    return results


def generate_random_couplets(
    theme_keywords: Optional[List[str]] = None,
    count: int = 10,
    analyze: bool = True,
) -> List[CoupletIndividual]:
    """
    Generate couplets from templates with random word filling (no rhyme targeting).
    Uses template_fill: pick random template, fill slots with random words from vocab.
    """
    templates = load_templates()
    vocab = load_vocab()
    theme_set = set(w.lower() for w in (theme_keywords or []))

    if not templates:
        return []

    results: List[CoupletIndividual] = []
    for _ in range(count):
        t1 = random.choice(templates)
        t2 = random.choice(templates)
        line1 = _fill_template(t1, vocab, theme_set, end_word=None)
        line2 = _fill_template(t2, vocab, theme_set, end_word=None)
        ind = CoupletIndividual(line1=line1, line2=line2)
        if analyze:
            analyze_individual(ind)
        results.append(ind)

    return results


def template_fill_couplets(
    theme_keywords: Optional[List[str]] = None,
    count: int = 10,
    analyze: bool = True,
) -> List[CoupletIndividual]:
    """
    Generate couplets from templates with rhyme-aware word filling.
    Alias for generate_seed_couplets for use in mixed population.
    """
    return generate_seed_couplets(
        theme_keywords=theme_keywords,
        count=count,
        rhyme_targets=None,
        analyze=analyze,
    )


# ---------------------------------------------------------------------------
# Corpus-based generation (for inject_random_immigrants)
# ---------------------------------------------------------------------------

_CORPUS_LINES: Optional[List[str]] = None


def _get_corpus_path() -> Path:
    """Path to verse corpus for random line sampling."""
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "phaseA_kaggle_verse.txt"


def _load_corpus_lines() -> List[str]:
    """Load and cache non-empty lines from verse corpus."""
    global _CORPUS_LINES
    if _CORPUS_LINES is not None:
        return _CORPUS_LINES

    path = _get_corpus_path()
    lines: List[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Split on newlines; also split long paragraphs into bar-sized chunks
        for para in content.split("\n"):
            para = para.strip()
            if not para:
                continue
            words = para.split()
            if len(words) < 3:
                continue
            # If paragraph is long, treat as multiple bars (roughly 8-16 words each)
            if len(words) <= 20:
                lines.append(para)
            else:
                for i in range(0, len(words), 12):
                    chunk = words[i : i + 16]
                    if len(chunk) >= 4:
                        lines.append(" ".join(chunk))

    _CORPUS_LINES = lines if lines else []
    return _CORPUS_LINES


def generate_couplet(
    theme_keywords: Optional[Set[str]] = None,
) -> CoupletIndividual:
    """
    Generate a random couplet from corpus lines.
    Picks two random lines. theme_keywords optional for future filtering.
    """
    lines = _load_corpus_lines()
    if len(lines) < 2:
        return CoupletIndividual(
            line1="I got the flow when I step in the spot",
            line2="You know I rock it hard when I hit the block",
        )

    line1 = random.choice(lines)
    line2 = random.choice(lines)
    while line2 == line1 and len(lines) > 1:
        line2 = random.choice(lines)

    return CoupletIndividual(line1=line1, line2=line2)


def get_generator() -> Callable:
    """Return a callable that generates CoupletIndividual (for inject_random_immigrants)."""
    return generate_couplet
