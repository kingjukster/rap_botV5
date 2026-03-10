"""
evo_rhyme/mutation.py

Evolution operators for couplet individuals. Builds tail_to_words index from
rhymes_grouped.csv + pronouncing for rhyme-family swaps. Each mutation makes
one conservative change and returns a new CoupletIndividual.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evo_rhyme.individual import CoupletIndividual, analyze_individual
from evo_rhyme.phonetics import (
    extract_rhyme_tail,
    extract_stressed_vowels_from_phones,
    get_pronunciations,
    get_vowel_to_words,
    multisyllable_overlap,
    syllable_count_word,
    tokenize_line,
)
from evo_rhyme.rhyme_graph import build_rhyme_graph

# Mutation type weights (must sum to 1.0)
MUTATION_WEIGHTS: Dict[str, float] = {
    "end_word_swap": 0.22,
    "internal_rhyme_insert": 0.18,
    "stressed_vowel_swap": 0.08,
    "syllable_adjust": 0.06,
    "semantic_swap": 0.10,
    "syntax_synonym": 0.10,
    "compression": 0.10,
    "expansion": 0.05,
    "phrase_replace": 0.02,
    "rhyme_graph_expand": 0.08,
    "stress_repair": 0.01,
}

# ---------------------------------------------------------------------------
# tail_to_words index: rhyme tail -> list of words
# ---------------------------------------------------------------------------

_TAIL_TO_WORDS: Optional[Dict[str, List[str]]] = None


def _get_rhymes_csv_path() -> Path:
    """Path to rhymes_grouped.csv."""
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "rhymes_grouped.csv"


def _build_tail_to_words() -> Dict[str, List[str]]:
    """
    Build tail_to_words index from rhymes_grouped.csv + pronouncing.
    Maps rhyme tail (phonetic string from last stressed vowel onward) -> list of words.
    Uses extract_rhyme_tail from phonetics; falls back to group_id for OOV words.
    """
    global _TAIL_TO_WORDS
    if _TAIL_TO_WORDS is not None:
        return _TAIL_TO_WORDS

    path = _get_rhymes_csv_path()
    tail_to_words: Dict[str, List[str]] = defaultdict(list)
    seen: Set[tuple] = set()  # (tail, word) to avoid duplicates

    if not path.exists():
        _TAIL_TO_WORDS = dict(tail_to_words)
        return _TAIL_TO_WORDS

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip().lower()
            group = row.get("group", "0")
            if not word:
                continue

            tail = extract_rhyme_tail(word)
            if tail is None:
                tail = f"group_{group}"

            key = (tail, word)
            if key not in seen:
                seen.add(key)
                tail_to_words[tail].append(word)

    _TAIL_TO_WORDS = {k: list(v) for k, v in tail_to_words.items()}
    return _TAIL_TO_WORDS


def get_tail_to_words() -> Dict[str, List[str]]:
    """Return the tail_to_words index (built on first call)."""
    return _build_tail_to_words()


# ---------------------------------------------------------------------------
# Mutation operators (each makes one conservative change)
# ---------------------------------------------------------------------------


def _copy_individual(ind: CoupletIndividual, line1: str, line2: str) -> CoupletIndividual:
    """Create new CoupletIndividual with given lines; reset features/scores."""
    return CoupletIndividual(
        line1=line1,
        line2=line2,
        features1=None,
        features2=None,
        scores=None,
        fitness=None,
        metadata=dict(ind.metadata),
    )


def _end_word_swap(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace end word with rhyme-family alternative from tail_to_words."""
    tokens1 = tokenize_line(individual.line1)
    tokens2 = tokenize_line(individual.line2)
    if not tokens1 or not tokens2:
        return None

    end1, end2 = tokens1[-1].lower(), tokens2[-1].lower()
    tail1 = extract_rhyme_tail(end1)
    tail2 = extract_rhyme_tail(end2)

    # Try swapping line1 end word
    corpus_vocab = config.get("corpus_vocab") if isinstance(config, dict) else None
    if tail1 and tail1 in tail_to_words:
        alts = [w for w in tail_to_words[tail1] if w != end1]
        if corpus_vocab:
            alts = [w for w in alts if w in corpus_vocab]
        if not alts:
            pass  # fall through to try line2
        else:
            new_end = random.choice(alts)
            new_tokens = tokens1[:-1] + [new_end]
            new_line1 = " ".join(new_tokens)
            return _copy_individual(individual, new_line1, individual.line2)

    # Try swapping line2 end word
    if tail2 and tail2 in tail_to_words:
        alts = [w for w in tail_to_words[tail2] if w != end2]
        if corpus_vocab:
            alts = [w for w in alts if w in corpus_vocab]
        if alts:
            new_end = random.choice(alts)
            new_tokens = tokens2[:-1] + [new_end]
            new_line2 = " ".join(new_tokens)
            return _copy_individual(individual, individual.line1, new_line2)

    return None


def _internal_rhyme_insert(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Swap middle word with one from same rhyme family."""
    for line, is_line1 in [(individual.line1, True), (individual.line2, False)]:
        tokens = tokenize_line(line)
        if len(tokens) < 3:
            continue

        mid_start, mid_end = 1, len(tokens) - 1
        if mid_end <= mid_start:
            continue

        idx = random.randint(mid_start, mid_end)
        word = tokens[idx].lower()
        tail = extract_rhyme_tail(word)

        if tail and tail in tail_to_words:
            alts = [w for w in tail_to_words[tail] if w != word]
            corpus_vocab = config.get("corpus_vocab") if isinstance(config, dict) else None
            if corpus_vocab:
                alts = [w for w in alts if w in corpus_vocab]
            if alts:
                new_tokens = tokens[:idx] + [random.choice(alts)] + tokens[idx + 1 :]
                new_line = " ".join(new_tokens)
                if is_line1:
                    return _copy_individual(individual, new_line, individual.line2)
                return _copy_individual(individual, individual.line1, new_line)
    return None


def _stressed_vowel_swap(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """
    Replace a word with another that shares the same stressed vowel.
    Uses vowel_to_words index; only proposes valid dictionary words from rhymes_grouped.csv.
    """
    vowel_to_words = get_vowel_to_words()
    if not vowel_to_words:
        return None

    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        tokens = tokenize_line(line)
        if not tokens:
            continue

        # Shuffle to randomize which word we try
        indices = list(range(len(tokens)))
        random.shuffle(indices)

        for idx in indices:
            word = tokens[idx].lower()
            phones_list = get_pronunciations(word)
            if not phones_list:
                continue

            stressed_vowels = extract_stressed_vowels_from_phones(phones_list[0])
            if not stressed_vowels:
                continue

            # Prefer primary/secondary stress (1, 2) over unstressed (0)
            stressed = [v for v in stressed_vowels if len(v) >= 2 and v[-1] in "12"]
            candidates = stressed if stressed else stressed_vowels

            for vowel in candidates:
                if vowel not in vowel_to_words:
                    continue
                alts = [w for w in vowel_to_words[vowel] if w != word]
                corpus_vocab = config.get("corpus_vocab") if isinstance(config, dict) else None
                if corpus_vocab:
                    alts = [w for w in alts if w in corpus_vocab]
                if not alts:
                    continue

                new_word = random.choice(alts)
                new_tokens = tokens[:idx] + [new_word] + tokens[idx + 1 :]
                new_line = " ".join(new_tokens)
                if is_line1:
                    return _copy_individual(individual, new_line, other)
                return _copy_individual(individual, other, new_line)

    return None


def _semantic_swap(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace generic word with theme word. Uses theme_keywords from config."""
    theme_keywords: Set[str] = set()
    if config and isinstance(config, dict) and "theme_keywords" in config:
        theme_keywords = set(w.lower() for w in config["theme_keywords"])
    if not theme_keywords:
        return None

    generic = {"thing", "stuff", "way", "place", "time", "life", "man", "people", "get", "make", "go", "see"}
    theme_word = random.choice(list(theme_keywords))

    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        tokens = tokenize_line(line)
        candidates = [t for t in tokens if t in generic and t not in theme_keywords]
        if not candidates:
            continue
        replace_word = random.choice(candidates)
        new_line = line.lower().replace(replace_word, theme_word, 1)
        if new_line != line.lower():
            if is_line1:
                return _copy_individual(individual, new_line, other)
            return _copy_individual(individual, other, new_line)
    return None


def _syllable_adjust(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Add or remove a short word to adjust syllable count."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        analyze_individual(individual)
        f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return None

    min_syl = 6
    max_syl = 18
    if config:
        if isinstance(config, dict):
            min_syl = config.get("min_syllables", min_syl)
            max_syl = config.get("max_syllables", max_syl)
        else:
            min_syl = getattr(config, "min_syllables", min_syl)
            max_syl = getattr(config, "max_syllables", max_syl)

    fillers = ["just", "so", "real", "all", "got", "now", "then", "yeah", "man"]
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    for line, feat, is_line1 in [
        (individual.line1, f1, True),
        (individual.line2, f2, False),
    ]:
        sc = feat.syllable_count
        if sc < min_syl and random.random() < 0.5:
            add_fillers = [f for f in fillers if f in corpus_vocab] if corpus_vocab else fillers
            if not add_fillers:
                continue
            add = random.choice(add_fillers)
            new_line = line + " " + add
            if is_line1:
                return _copy_individual(individual, new_line, individual.line2)
            return _copy_individual(individual, individual.line1, new_line)
        if sc > max_syl and len(feat.tokens) > 1:
            idx = random.randint(0, len(feat.tokens) - 2)
            drop = feat.tokens[idx]
            if syllable_count_word(drop) >= 1:
                new_tokens = feat.tokens[:idx] + feat.tokens[idx + 1 :]
                new_line = " ".join(new_tokens)
                if is_line1:
                    return _copy_individual(individual, new_line, individual.line2)
                return _copy_individual(individual, individual.line1, new_line)
    return None


def _syntax_synonym(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace word with simple synonym (small fixed set)."""
    synonyms: Dict[str, List[str]] = {
        "big": ["large", "huge", "massive"],
        "small": ["little", "tiny"],
        "good": ["great", "nice", "fine"],
        "bad": ["wrong", "rough"],
        "get": ["got", "grab"],
        "make": ["made", "build"],
    }
    text = f"{individual.line1} {individual.line2}".lower()
    tokens = tokenize_line(text)
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    for t in tokens:
        if t in synonyms:
            alts = [a for a in synonyms[t] if a in corpus_vocab] if corpus_vocab else synonyms[t]
            if not alts:
                continue
            alt = random.choice(alts)
            new_line1 = individual.line1.replace(t, alt) if t in individual.line1.lower() else individual.line1
            new_line2 = individual.line2.replace(t, alt) if t in individual.line2.lower() else individual.line2
            if new_line1 != individual.line1 or new_line2 != individual.line2:
                return _copy_individual(individual, new_line1, new_line2)
    return None


def _compression(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Remove one non-essential word to shorten line."""
    droppable = ["the", "a", "an", "just", "so", "all", "that", "and"]
    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        tokens = tokenize_line(line)
        if len(tokens) < 4:
            continue
        for i, t in enumerate(tokens):
            if t.lower() in droppable:
                new_tokens = tokens[:i] + tokens[i + 1 :]
                new_line = " ".join(new_tokens)
                if is_line1:
                    return _copy_individual(individual, new_line, other)
                return _copy_individual(individual, other, new_line)
    return None


def _expansion(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Insert one short word to expand line."""
    fillers = ["just", "so", "real", "all", "got"]
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    if corpus_vocab:
        fillers = [f for f in fillers if f in corpus_vocab]
    if not fillers:
        return None
    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        tokens = tokenize_line(line)
        if len(tokens) < 2:
            continue
        idx = random.randint(1, len(tokens) - 1)
        add = random.choice(fillers)
        new_tokens = tokens[:idx] + [add] + tokens[idx:]
        new_line = " ".join(new_tokens)
        if is_line1:
            return _copy_individual(individual, new_line, other)
        return _copy_individual(individual, other, new_line)
    return None


def _phrase_replace(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace short phrase with alternative (conservative 2-word swap)."""
    phrases: Dict[str, List[str]] = {
        "a lot": ["so much", "so many"],
        "got to": ["gotta", "have to"],
        "going to": ["gonna", "will"],
        "want to": ["wanna", "gonna"],
    }
    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        line_lower = line.lower()
        for phrase, alts in phrases.items():
            if phrase in line_lower:
                new_phrase = random.choice(alts)
                new_line = line_lower.replace(phrase, new_phrase, 1)
                if is_line1:
                    return _copy_individual(individual, new_line, other)
                return _copy_individual(individual, other, new_line)
    return None


def _stress_repair(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Swap weak end word with stressed alternative from rhyme family."""
    return _end_word_swap(individual, tail_to_words, config)


def _rhyme_graph_expand(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """
    Pick a random word in a random line. Build rhyme graph for that line.
    Find words with few edges. Replace one with a word from tail_to_words
    that rhymes with more words in the line.
    """
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None

    lines = [individual.line1, individual.line2]
    line_idx = random.randint(0, 1)
    line = lines[line_idx]
    tokens = tokenize_line(line)
    if len(tokens) < 2:
        return None

    edges = build_rhyme_graph(tokens)
    # Degree per vertex: count edges involving each index
    degree: List[int] = [0] * len(tokens)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    # Indices with few edges (candidates for replacement)
    min_degree = min(degree)
    low_indices = [i for i in range(len(tokens)) if degree[i] == min_degree]
    if not low_indices:
        return None

    idx = random.choice(low_indices)
    current_word = tokens[idx].lower()

    # Find tail that rhymes with most OTHER words in the line
    best_tail: Optional[str] = None
    best_count = -1
    other_indices = [j for j in range(len(tokens)) if j != idx]

    for tail, words_list in tail_to_words.items():
        if not words_list:
            continue
        count = 0
        for j in other_indices:
            tail_j = extract_rhyme_tail(tokens[j])
            if multisyllable_overlap(tail, tail_j) >= 2:
                count += 1
        if count > best_count:
            best_count = count
            best_tail = tail

    if best_tail is None or best_count < 1:
        return None

    alts = [w for w in tail_to_words[best_tail] if w != current_word]
    if corpus_vocab:
        alts = [w for w in alts if w in corpus_vocab]
    if not alts:
        return None

    new_word = random.choice(alts)
    new_tokens = tokens[:idx] + [new_word] + tokens[idx + 1 :]
    new_line = " ".join(new_tokens)
    if line_idx == 0:
        return _copy_individual(individual, new_line, individual.line2)
    return _copy_individual(individual, individual.line1, new_line)


# Registry of mutation functions
_MUTATION_FUNCS: Dict[str, Callable[..., Optional[CoupletIndividual]]] = {
    "end_word_swap": _end_word_swap,
    "internal_rhyme_insert": _internal_rhyme_insert,
    "stressed_vowel_swap": _stressed_vowel_swap,
    "syllable_adjust": _syllable_adjust,
    "semantic_swap": _semantic_swap,
    "syntax_synonym": _syntax_synonym,
    "compression": _compression,
    "expansion": _expansion,
    "phrase_replace": _phrase_replace,
    "rhyme_graph_expand": _rhyme_graph_expand,
    "stress_repair": _stress_repair,
}


def mutate(
    individual: CoupletIndividual,
    config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
) -> CoupletIndividual:
    """
    Apply one mutation to individual. Selects mutation type by MUTATION_WEIGHTS,
    makes one conservative change, returns new CoupletIndividual.
    If selected mutation fails, tries others; if all fail, returns copy unchanged.
    """
    w = weights or MUTATION_WEIGHTS
    tail_to_words = get_tail_to_words()

    choices = [k for k in w if w.get(k, 0) > 0 and k in _MUTATION_FUNCS]
    if not choices:
        return _copy_individual(individual, individual.line1, individual.line2)

    probs = [w[c] for c in choices]
    total = sum(probs)
    if total <= 0:
        return _copy_individual(individual, individual.line1, individual.line2)

    probs = [p / total for p in probs]
    order = list(range(len(choices)))
    random.shuffle(order)

    for i in order:
        name = choices[i]
        if random.random() > probs[i]:
            continue
        func = _MUTATION_FUNCS[name]
        result = func(individual, tail_to_words, config)
        if result is not None:
            return result

    # Fallback: try each mutation once
    for name in random.sample(choices, len(choices)):
        result = _MUTATION_FUNCS[name](individual, tail_to_words, config)
        if result is not None:
            return result

    return _copy_individual(individual, individual.line1, individual.line2)
