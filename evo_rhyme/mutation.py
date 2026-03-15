"""
evo_rhyme/mutation.py

Evolution operators for couplet individuals. Builds tail_to_words index from
rhymes_grouped.csv + pronouncing for rhyme-family swaps. Each mutation makes
one conservative change and returns a new CoupletIndividual.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evo_rhyme.individual import CoupletIndividual, analyze_individual
from evo_rhyme.lm_rewriter import BarRewriter, RewriterConfig
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

logger = logging.getLogger(__name__)

MUTATION_WEIGHTS: Dict[str, float] = {
    "lm_rhyme_rewrite": 0.10,
    "lm_internal_rhyme": 0.08,
    "lm_theme_rewrite": 0.08,
    "lm_paraphrase": 0.06,
    "lm_structural_rewrite": 0.06,
    "lm_metaphor_inject": 0.08,
    "lm_contrast_swap": 0.06,
    "lm_score_guided": 0.10,
    "lm_tighten": 0.04,
    "lm_expand": 0.02,
    "stressed_vowel_swap": 0.06,
    "syllable_adjust": 0.04,
    "rhyme_graph_expand": 0.04,
    "embedding_rhyme_walk": 0.05,
    "chain_extension": 0.06,
    "end_word_swap": 0.02,
    "multisyllable_rhyme": 0.02,
    "line_replace": 0.06,
    "block_replace": 0.04,
}

LEGACY_MUTATION_WEIGHTS: Dict[str, float] = {
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
# Shared BarRewriter instance (lazy-init)
# ---------------------------------------------------------------------------

_REWRITER: Optional[BarRewriter] = None


def get_rewriter(config: Optional[Dict] = None) -> BarRewriter:
    """Get or create the shared BarRewriter instance."""
    global _REWRITER
    if _REWRITER is None:
        rewriter_cfg = RewriterConfig()
        if config and isinstance(config, dict) and "rewriter" in config:
            rc = config["rewriter"]
            for k, v in rc.items():
                if hasattr(rewriter_cfg, k):
                    setattr(rewriter_cfg, k, v)
        _REWRITER = BarRewriter(rewriter_cfg)
    return _REWRITER

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


# ---------------------------------------------------------------------------
# LM-backed mutation operators (use BarRewriter)
# ---------------------------------------------------------------------------


def _lm_config_helpers(config: Any):
    """Extract common config values used by all LM mutations."""
    theme_keywords: List[str] = []
    if config and isinstance(config, dict):
        theme_keywords = config.get("theme_keywords", [])
    min_syl = config.get("min_syllables", 6) if isinstance(config, dict) else 6
    max_syl = config.get("max_syllables", 18) if isinstance(config, dict) else 18
    return theme_keywords, (min_syl, max_syl)


def _lm_rhyme_rewrite(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite a random line so its end word rhymes better with the other line."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)

        pick_line1 = random.random() < 0.5
        if pick_line1:
            line, other_line = individual.line1, individual.line2
        else:
            line, other_line = individual.line2, individual.line1

        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]

        candidates = rewriter.rhyme_rewrite(line, rhyme_target, theme_keywords, syllable_range)
        if not candidates:
            return None

        target_tail = extract_rhyme_tail(rhyme_target)
        best_line: Optional[str] = None
        best_score = -1
        for cand in candidates:
            tokens = tokenize_line(cand)
            if not tokens:
                continue
            cand_tail = extract_rhyme_tail(tokens[-1])
            score = multisyllable_overlap(cand_tail, target_tail)
            if score > best_score:
                best_score = score
                best_line = cand

        if best_line is None:
            return None

        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_rhyme_rewrite failed", exc_info=True)
        return None


def _lm_internal_rhyme(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite a line to add an internal rhyme at a middle word position."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)

        pick_line1 = random.random() < 0.5
        line = individual.line1 if pick_line1 else individual.line2

        tokens = tokenize_line(line)
        if len(tokens) < 3:
            return None

        mid_pos = random.randint(1, len(tokens) - 2)
        rhyme_with_word = tokens[mid_pos]
        rhyme_tail = extract_rhyme_tail(rhyme_with_word)
        if rhyme_tail is None:
            return None

        candidates = rewriter.internal_rhyme_rewrite(
            line, mid_pos, rhyme_with_word, theme_keywords, syllable_range,
        )
        if not candidates:
            return None

        target_tail = rhyme_tail
        best_line: Optional[str] = None
        best_score = -1
        for cand in candidates:
            cand_tokens = tokenize_line(cand)
            if len(cand_tokens) < 3:
                continue
            for i in range(1, len(cand_tokens) - 1):
                t = extract_rhyme_tail(cand_tokens[i])
                score = multisyllable_overlap(t, target_tail)
                if score > best_score:
                    best_score = score
                    best_line = cand

        if best_line is None:
            return None

        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_internal_rhyme failed", exc_info=True)
        return None


def _lm_theme_rewrite(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite the weaker-theme line to increase semantic relevance."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        if not theme_keywords:
            return None

        pick_line1 = True
        if individual.scores and "semantic1" in individual.scores and "semantic2" in individual.scores:
            pick_line1 = individual.scores["semantic1"] <= individual.scores["semantic2"]
        else:
            pick_line1 = random.random() < 0.5

        line = individual.line1 if pick_line1 else individual.line2

        candidates = rewriter.theme_rewrite(line, theme_keywords, syllable_range)
        if not candidates:
            return None

        new_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, new_line, individual.line2)
        return _copy_individual(individual, individual.line1, new_line)
    except Exception:
        logger.warning("_lm_theme_rewrite failed", exc_info=True)
        return None


def _lm_paraphrase(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Paraphrase a random line while preserving end rhyme."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        _, syllable_range = _lm_config_helpers(config)

        pick_line1 = random.random() < 0.5
        line = individual.line1 if pick_line1 else individual.line2

        candidates = rewriter.paraphrase(line, preserve_end_rhyme=True, syllable_range=syllable_range)
        if not candidates:
            return None

        new_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, new_line, individual.line2)
        return _copy_individual(individual, individual.line1, new_line)
    except Exception:
        logger.warning("_lm_paraphrase failed", exc_info=True)
        return None


def _lm_tighten(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Tighten the longer line by ~2 syllables."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        _, syllable_range = _lm_config_helpers(config)

        f1, f2 = individual.features1, individual.features2
        if not f1 or not f2:
            analyze_individual(individual)
            f1, f2 = individual.features1, individual.features2
        if not f1 or not f2:
            return None

        pick_line1 = f1.syllable_count >= f2.syllable_count
        if pick_line1:
            line, current_syl = individual.line1, f1.syllable_count
        else:
            line, current_syl = individual.line2, f2.syllable_count

        target_syl = max(syllable_range[0], current_syl - 2)
        candidates = rewriter.tighten(line, target_syllables=target_syl, syllable_range=syllable_range)
        if not candidates:
            return None

        new_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, new_line, individual.line2)
        return _copy_individual(individual, individual.line1, new_line)
    except Exception:
        logger.warning("_lm_tighten failed", exc_info=True)
        return None


def _lm_expand(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Expand the shorter line by ~2 syllables."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        _, syllable_range = _lm_config_helpers(config)

        f1, f2 = individual.features1, individual.features2
        if not f1 or not f2:
            analyze_individual(individual)
            f1, f2 = individual.features1, individual.features2
        if not f1 or not f2:
            return None

        pick_line1 = f1.syllable_count <= f2.syllable_count
        if pick_line1:
            line, current_syl = individual.line1, f1.syllable_count
        else:
            line, current_syl = individual.line2, f2.syllable_count

        target_syl = min(syllable_range[1], current_syl + 2)
        candidates = rewriter.expand(line, target_syllables=target_syl, syllable_range=syllable_range)
        if not candidates:
            return None

        new_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, new_line, individual.line2)
        return _copy_individual(individual, individual.line1, new_line)
    except Exception:
        logger.warning("_lm_expand failed", exc_info=True)
        return None


def _lm_structural_rewrite(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite a line with completely different sentence structure."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        pick_line1 = random.random() < 0.5
        if pick_line1:
            line, other_line = individual.line1, individual.line2
        else:
            line, other_line = individual.line2, individual.line1
        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]
        candidates = rewriter.structural_rewrite(line, rhyme_target, theme_keywords, syllable_range)
        if not candidates:
            return None
        best_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_structural_rewrite failed", exc_info=True)
        return None


def _lm_metaphor_inject(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite a line using vivid metaphor."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        pick_line1 = random.random() < 0.5
        if pick_line1:
            line, other_line = individual.line1, individual.line2
        else:
            line, other_line = individual.line2, individual.line1
        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]
        candidates = rewriter.metaphor_inject(line, rhyme_target, theme_keywords, syllable_range)
        if not candidates:
            return None
        best_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_metaphor_inject failed", exc_info=True)
        return None


def _lm_contrast_swap(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Rewrite a line with opposing emotional tone."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        pick_line1 = random.random() < 0.5
        if pick_line1:
            line, other_line = individual.line1, individual.line2
        else:
            line, other_line = individual.line2, individual.line1
        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]
        candidates = rewriter.contrast_swap(line, rhyme_target, theme_keywords, syllable_range)
        if not candidates:
            return None
        best_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_contrast_swap failed", exc_info=True)
        return None


def _lm_score_guided(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Score-aware mutation: find weakest dimension and target it."""
    try:
        scores = individual.scores or {}
        weakness_map = {
            "internal_rhyme": "internal_rhyme",
            "semantic": "semantic",
            "fluency": "fluency",
            "rhyme_graph": "rhyme_chain_density",
        }
        weakest_key = min(weakness_map, key=lambda k: scores.get(k, 0.5))
        weakest_score = scores.get(weakest_key, 0.5)
        if weakest_score >= 0.5:
            return None

        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        pick_line1 = random.random() < 0.5
        if pick_line1:
            line, other_line = individual.line1, individual.line2
        else:
            line, other_line = individual.line2, individual.line1
        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]

        candidates = rewriter.score_guided_rewrite(
            line, weakness_map[weakest_key], weakest_score,
            rhyme_target, theme_keywords, syllable_range,
        )
        if not candidates:
            return None
        best_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, best_line, individual.line2)
        return _copy_individual(individual, individual.line1, best_line)
    except Exception:
        logger.warning("_lm_score_guided failed", exc_info=True)
        return None


def _chain_extension(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Extend a phoneme chain by substituting a word with one sharing the same cluster."""
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    lines = [individual.line1, individual.line2]
    line_idx = random.randint(0, 1)
    other_idx = 1 - line_idx
    tokens_src = tokenize_line(lines[other_idx])
    tokens_tgt = tokenize_line(lines[line_idx])
    if len(tokens_src) < 2 or len(tokens_tgt) < 3:
        return None
    src_word = random.choice(tokens_src[:-1])
    src_tail = extract_rhyme_tail(src_word)
    if src_tail is None:
        return None
    if src_tail not in tail_to_words:
        return None
    alts = [w for w in tail_to_words[src_tail] if w != src_word.lower()]
    if corpus_vocab:
        alts = [w for w in alts if w in corpus_vocab]
    if not alts:
        return None
    new_word = random.choice(alts)
    mid_indices = list(range(1, len(tokens_tgt) - 1))
    if not mid_indices:
        return None
    swap_idx = random.choice(mid_indices)
    new_tokens = tokens_tgt[:swap_idx] + [new_word] + tokens_tgt[swap_idx + 1:]
    new_line = " ".join(new_tokens)
    if line_idx == 0:
        return _copy_individual(individual, new_line, individual.line2)
    return _copy_individual(individual, individual.line1, new_line)


def _multisyllable_rhyme(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace end word with one having higher multisyllable overlap with the paired line."""
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    tokens1 = tokenize_line(individual.line1)
    tokens2 = tokenize_line(individual.line2)
    if not tokens1 or not tokens2:
        return None
    end1, end2 = tokens1[-1].lower(), tokens2[-1].lower()
    tail2 = extract_rhyme_tail(end2)
    if tail2 is None:
        return None
    current_overlap = multisyllable_overlap(extract_rhyme_tail(end1), tail2)
    best_word = None
    best_overlap = current_overlap
    for tail, words in tail_to_words.items():
        for w in words:
            if w == end1:
                continue
            if corpus_vocab and w not in corpus_vocab:
                continue
            w_tail = extract_rhyme_tail(w)
            ov = multisyllable_overlap(w_tail, tail2)
            if ov > best_overlap:
                best_overlap = ov
                best_word = w
    if best_word is None:
        return None
    new_tokens = tokens1[:-1] + [best_word]
    new_line = " ".join(new_tokens)
    return _copy_individual(individual, new_line, individual.line2)


def _embedding_rhyme_walk(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace one token using nearest neighbors in rhyme embedding space."""
    try:
        from evo_rhyme.rhyme_embedding import get_rhyme_embedding_space
    except Exception:
        return None

    space = get_rhyme_embedding_space()
    corpus_vocab = config.get("corpus_vocab") if config and isinstance(config, dict) else None
    k = int(config.get("embedding_neighbor_k", 12)) if isinstance(config, dict) else 12
    min_cos = float(config.get("embedding_min_cosine", 0.6)) if isinstance(config, dict) else 0.6

    for line, other, is_line1 in [
        (individual.line1, individual.line2, True),
        (individual.line2, individual.line1, False),
    ]:
        tokens = tokenize_line(line)
        if len(tokens) < 3:
            continue
        idxs = list(range(1, len(tokens) - 1))
        random.shuffle(idxs)
        for idx in idxs:
            word = tokens[idx].lower()
            if not space.has_word(word):
                continue
            neighbors = space.rhyme_neighbors(word, k=k, min_cosine=min_cos)
            if corpus_vocab:
                neighbors = [w for w in neighbors if w in corpus_vocab]
            if not neighbors:
                continue
            new_word = random.choice(neighbors)
            if new_word == word:
                continue
            new_tokens = list(tokens)
            new_tokens[idx] = new_word
            new_line = " ".join(new_tokens)
            if is_line1:
                return _copy_individual(individual, new_line, other)
            return _copy_individual(individual, other, new_line)
    return None


def _line_replace_mutation(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace an entire line with a freshly LM-generated one. Medium-scale mutation."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)

        pick_line1 = random.random() < 0.5
        if pick_line1:
            other_line = individual.line2
        else:
            other_line = individual.line1

        other_tokens = tokenize_line(other_line)
        if not other_tokens:
            return None
        rhyme_target = other_tokens[-1]
        theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"

        candidates = rewriter.structural_rewrite(
            "write a new line",
            rhyme_target, theme_keywords, syllable_range,
        )
        if not candidates:
            candidates = rewriter.rhyme_rewrite(
                other_line, rhyme_target, theme_keywords, syllable_range,
            )
        if not candidates:
            return None

        new_line = candidates[0]
        if pick_line1:
            return _copy_individual(individual, new_line, individual.line2)
        return _copy_individual(individual, individual.line1, new_line)
    except Exception:
        logger.warning("_line_replace_mutation failed", exc_info=True)
        return None


def _block_replace_mutation(
    individual: CoupletIndividual,
    tail_to_words: Dict[str, List[str]],
    config: Any,
) -> Optional[CoupletIndividual]:
    """Replace the entire couplet with a freshly generated one. Large-scale mutation."""
    try:
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords, syllable_range = _lm_config_helpers(config)
        theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"

        verse_text = f"{individual.line1}\n{individual.line2}"
        prompt = (
            f'Write 2 completely new rap lines with fresh imagery.\n'
            f'Lines must rhyme with each other (end words rhyme).\n'
            f'Theme: {theme}\n'
            f'Each line: {syllable_range[0]}-{syllable_range[1]} syllables.\n\n'
            f'Write ONLY the 2 lines, nothing else.'
        )

        from evo_rhyme.lm_rewriter import _cache_key
        key = _cache_key(verse_text, "block_replace", (theme,))
        raw = rewriter._call_lm(prompt, key)

        new_lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if len(new_lines) < 2:
            return None

        return _copy_individual(individual, new_lines[0], new_lines[1])
    except Exception:
        logger.warning("_block_replace_mutation failed", exc_info=True)
        return None


# Registry of mutation functions
_MUTATION_FUNCS: Dict[str, Callable[..., Optional[CoupletIndividual]]] = {
    # LM-backed operators
    "lm_rhyme_rewrite": _lm_rhyme_rewrite,
    "lm_internal_rhyme": _lm_internal_rhyme,
    "lm_theme_rewrite": _lm_theme_rewrite,
    "lm_paraphrase": _lm_paraphrase,
    "lm_tighten": _lm_tighten,
    "lm_expand": _lm_expand,
    "lm_structural_rewrite": _lm_structural_rewrite,
    "lm_metaphor_inject": _lm_metaphor_inject,
    "lm_contrast_swap": _lm_contrast_swap,
    "lm_score_guided": _lm_score_guided,
    # Legacy operators (cheap fallbacks)
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
    "chain_extension": _chain_extension,
    "multisyllable_rhyme": _multisyllable_rhyme,
    "embedding_rhyme_walk": _embedding_rhyme_walk,
    "line_replace": _line_replace_mutation,
    "block_replace": _block_replace_mutation,
}


_SAFE_LEGACY_OPS: Set[str] = {"syllable_adjust", "compression"}


def mutate(
    individual: CoupletIndividual,
    config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
    lm_budget: Optional[Dict[str, int]] = None,
    lm_only: bool = False,
) -> CoupletIndividual:
    """
    Apply one mutation to individual. Selects mutation type by MUTATION_WEIGHTS,
    makes one conservative change, returns new CoupletIndividual.
    If selected mutation fails, tries others; if all fail, returns copy unchanged.

    *lm_budget*: if provided, a mutable dict like ``{"remaining": 200}``.
    LM operators (keys starting with ``lm_``) are excluded when the budget is
    exhausted, and each successful LM mutation decrements the counter by 1.
    Pass ``None`` for unlimited LM calls.

    *lm_only*: if True, only LM operators and safe legacy operators (syllable_adjust,
    compression) are used. Destructive legacy word-swaps are never tried, even as
    fallback. When an LM mutation fails, the individual is returned unchanged rather
    than corrupted by a random word swap.
    """
    w = weights or MUTATION_WEIGHTS
    tail_to_words = get_tail_to_words()

    lm_allowed = lm_budget is None or lm_budget.get("remaining", 0) > 0

    choices = [
        k for k in w
        if w.get(k, 0) > 0
        and k in _MUTATION_FUNCS
        and (lm_allowed or not k.startswith("lm_"))
    ]

    if lm_only:
        choices = [
            k for k in choices
            if k.startswith("lm_") or k in _SAFE_LEGACY_OPS
        ]

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
            if lm_budget is not None and name.startswith("lm_"):
                lm_budget["remaining"] -= 1
            return result

    # Fallback: try each mutation once (only from allowed choices)
    for name in random.sample(choices, len(choices)):
        result = _MUTATION_FUNCS[name](individual, tail_to_words, config)
        if result is not None:
            if lm_budget is not None and name.startswith("lm_"):
                lm_budget["remaining"] -= 1
            return result

    return _copy_individual(individual, individual.line1, individual.line2)
