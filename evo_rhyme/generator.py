"""
evo_rhyme/generator.py

Seed couplet generation from templates and vocab. Loads rhyme groups from
evo_rhyme.rhyme_resources for end-word pairing. Builds tail_to_words (group_to_words)
index from rhymes_grouped.csv for rhyme targeting in mutations.

Also supports corpus-based random couplets for inject_random_immigrants.
"""

from __future__ import annotations

import csv
import random
import re
import re as _re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from evo_rhyme.individual import CoupletIndividual, analyze_individual

from evo_rhyme.rhyme_resources import RHYME_GROUPS, load_rhyme_groups

# ---------------------------------------------------------------------------
# Rhyme-tagged slot parsing
# ---------------------------------------------------------------------------

_RHYME_TAG_RE = _re.compile(r"^(\w+?)_rhyme_([A-Z])$")


def _parse_slot_name(name: str):
    """Parse a placeholder name into (base_type, rhyme_tag).

    'noun_rhyme_A' -> ('noun', 'A')
    'verb_rhyme_B' -> ('verb', 'B')
    'noun'         -> ('noun', None)
    """
    m = _RHYME_TAG_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    return name, None


@dataclass
class TemplateEntry:
    """A template with optional metadata."""
    text: str
    target_syllables: Optional[int] = None

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
    """Load non-empty, non-comment stripped lines from a text file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


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


_SYL_ANNOTATION_RE = re.compile(r"^#\s*@syl\s*:\s*(\d+)\s*$")


def load_templates() -> List[TemplateEntry]:
    """Load templates from data/evo_rhyme/templates.txt. Creates minimal file if missing.

    Returns list of TemplateEntry with text and optional target_syllables.
    Lines starting with '#' are treated as comments/annotations.
    A comment of the form '# @syl:N' sets target_syllables for the next template.
    """
    _ensure_vocab_files()
    if not TEMPLATES_PATH.exists():
        return []
    full_text = TEMPLATES_PATH.read_text(encoding="utf-8")
    entries: List[TemplateEntry] = []
    pending_syl: Optional[int] = None
    for raw in full_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = _SYL_ANNOTATION_RE.match(stripped)
            if m:
                pending_syl = int(m.group(1))
            continue
        entries.append(TemplateEntry(text=stripped, target_syllables=pending_syl))
        pending_syl = None
    return entries


def _load_verb_conjugations() -> tuple:
    """Load irregular verb conjugations from verbs.csv (present,past,participle)."""
    csv_path = VOCAB_DIR / "verbs.csv"
    extra_verbs: list = []
    extra_past: list = []
    if csv_path.exists():
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                extra_verbs.append(parts[0].strip().lower())
                extra_past.append(parts[1].strip().lower())
    return extra_verbs, extra_past


def load_vocab() -> Dict[str, List[str]]:
    """
    Load vocab from data/evo_rhyme/vocab/. Creates minimal files if missing.
    Returns dict with keys: nouns, verbs, adjectives, verbs_past, verbs_ing, states.
    Also supplements from verbs.csv for irregular conjugation coverage.
    """
    _ensure_vocab_files()
    verbs = _load_lines(VOCAB_DIR / "verbs.txt") or _MINIMAL_VERBS
    verbs_past = _load_lines(VOCAB_DIR / "verbs_past.txt") or _MINIMAL_VERBS_PAST

    csv_verbs, csv_past = _load_verb_conjugations()
    verb_set = set(v.lower() for v in verbs)
    past_set = set(v.lower() for v in verbs_past)
    for v in csv_verbs:
        if v not in verb_set:
            verbs.append(v)
            verb_set.add(v)
    for v in csv_past:
        if v not in past_set:
            verbs_past.append(v)
            past_set.add(v)

    verbs_ing = _load_lines(VOCAB_DIR / "verbs_ing.txt")
    if not verbs_ing:
        verbs_ing = [_verb_to_ing(v) for v in verbs]
    vocab: Dict[str, List[str]] = {
        "nouns": _load_lines(VOCAB_DIR / "nouns.txt") or _MINIMAL_NOUNS,
        "verbs": verbs,
        "adjectives": _load_lines(VOCAB_DIR / "adjectives.txt") or _MINIMAL_ADJECTIVES,
        "verbs_past": verbs_past,
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
    group_to_words: Optional[Dict[int, List[str]]] = None,
) -> str:
    """Fill template placeholders. If end_word is provided, use it for the last placeholder.

    Supports rhyme-tagged slots: {type_rhyme_TAG}. All slots sharing the same TAG
    are filled with words from the same rhyme group.
    """
    placeholders = _parse_template(template)
    if not placeholders:
        return template

    # Pass 1: identify rhyme tags and assign rhyme groups
    rhyme_group_assignments: Dict[str, Set[str]] = {}
    if group_to_words:
        tag_slots: Dict[str, List[str]] = {}
        for name, _, _ in placeholders:
            base_type, tag = _parse_slot_name(name)
            if tag:
                tag_slots.setdefault(tag, []).append(base_type)

        for tag, base_types in tag_slots.items():
            viable_groups = []
            for gid, words in group_to_words.items():
                if len(words) >= 2:
                    viable_groups.append((gid, words))
            if viable_groups:
                gid, words = random.choice(viable_groups)
                rhyme_group_assignments[tag] = set(w.lower() for w in words)

    # Pass 2: fill slots (reverse order to preserve indices)
    result = template
    for i in range(len(placeholders) - 1, -1, -1):
        name, start, end = placeholders[i]
        if end_word is not None and i == len(placeholders) - 1:
            word = end_word
        else:
            base_type, tag = _parse_slot_name(name)
            if tag and tag in rhyme_group_assignments:
                word = _pick_word(vocab, base_type, theme_keywords,
                                  rhyme_group_words=rhyme_group_assignments[tag])
            else:
                word = _pick_word(vocab, base_type, theme_keywords)
        result = result[:start] + word + result[end:]

    return result


def _select_template(
    templates: List[TemplateEntry],
    min_syl: int = 6,
    max_syl: int = 18,
) -> TemplateEntry:
    """Select a template, preferring those with target_syllables in range."""
    if not templates:
        return TemplateEntry(text="{noun} in the {noun}")

    preferred = [t for t in templates if t.target_syllables is not None
                 and min_syl <= t.target_syllables <= max_syl]
    if preferred and random.random() < 0.6:
        return random.choice(preferred)
    return random.choice(templates)


def _shuffled_template_cycle(
    templates: List[TemplateEntry],
) -> List[TemplateEntry]:
    """Return a shuffled copy of templates for round-robin iteration."""
    shuffled = list(templates)
    random.shuffle(shuffled)
    return shuffled


def generate_seed_couplets(
    theme_keywords: Optional[List[str]] = None,
    count: int = 10,
    rhyme_targets: Optional[List[str]] = None,
    analyze: bool = True,
    min_syllables: int = 6,
    max_syllables: int = 18,
) -> List[CoupletIndividual]:
    """
    Generate seed couplets from templates with theme-aware word filling.
    End-word pairs are picked from the same rhyme group (from evo_rhyme.rhyme_resources).

    Args:
        theme_keywords: Optional list of theme words to prefer when filling slots.
        count: Number of couplets to generate.
        rhyme_targets: Optional list of words; end-word pairs chosen from
            words in rhyme_targets that share a rhyme group. If None, any rhyme group.
        analyze: If True, populate features1/features2 via analyze_individual.
        min_syllables: Minimum target syllable count for template selection bias.
        max_syllables: Maximum target syllable count for template selection bias.

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

    shuffled = _shuffled_template_cycle(templates)
    n_templates = len(shuffled)

    results: List[CoupletIndividual] = []
    for idx in range(count):
        t1_entry = shuffled[(idx * 2) % n_templates]
        t2_entry = shuffled[(idx * 2 + 1) % n_templates]
        t1 = t1_entry.text
        t2 = t2_entry.text
        ph1 = _parse_template(t1)
        ph2 = _parse_template(t2)
        slot1_name = ph1[-1][0] if ph1 else "noun"
        slot2_name = ph2[-1][0] if ph2 else "noun"
        base1, _ = _parse_slot_name(slot1_name)
        base2, _ = _parse_slot_name(slot2_name)
        key1 = _slot_mapping().get(base1, "nouns")
        key2 = _slot_mapping().get(base2, "nouns")

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

            v1 = set(w.lower() for w in vocab.get(key1, []))
            v2 = set(w.lower() for w in vocab.get(key2, []))
            fit1 = [w for w in candidates if w in v1]
            fit2 = [w for w in candidates if w in v2]
            if fit1 and fit2:
                end1 = random.choice(fit1)
                end2 = random.choice([w for w in fit2 if w != end1] or fit2)
            else:
                end1 = _pick_word(vocab, base1, theme_set)
                end2 = _pick_word(vocab, base2, theme_set)
        else:
            end1 = _pick_word(vocab, base1, theme_set)
            end2 = _pick_word(vocab, base2, theme_set)

        line1 = _fill_template(t1, vocab, theme_set, end_word=end1,
                               group_to_words=group_to_words)
        line2 = _fill_template(t2, vocab, theme_set, end_word=end2,
                               group_to_words=group_to_words)

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
    group_to_words = get_group_to_words()
    theme_set = set(w.lower() for w in (theme_keywords or []))

    if not templates:
        return []

    shuffled = _shuffled_template_cycle(templates)
    n_templates = len(shuffled)

    results: List[CoupletIndividual] = []
    for idx in range(count):
        t1_entry = shuffled[(idx * 2) % n_templates]
        t2_entry = shuffled[(idx * 2 + 1) % n_templates]
        line1 = _fill_template(t1_entry.text, vocab, theme_set, end_word=None,
                               group_to_words=group_to_words)
        line2 = _fill_template(t2_entry.text, vocab, theme_set, end_word=None,
                               group_to_words=group_to_words)
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
