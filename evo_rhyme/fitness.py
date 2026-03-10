"""
evo_rhyme/fitness.py

Fitness scoring for couplet individuals. Computes component scores and
aggregate fitness from weighted sum.
"""

from __future__ import annotations

import difflib
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from evo_rhyme.individual import CoupletIndividual, LineFeatures, VerseIndividual, VerseFeatures
from evo_rhyme.phonetics import (
    PhoneticFeature,
    extract_rhyme_tail,
    multisyllable_overlap,
    phonetic_similarity,
)
from evo_rhyme.rhyme_graph import score_line_rhyme_graph
from evo_rhyme.style_profile import StyleProfile

# MVP weights - rebalanced to avoid early saturation; max fitness rarely achieved
# lexical_validity + ngram_fluency protect against rhyme-optimized nonsense
# rhyme_family_repetition_penalty + repeated_shell_penalty block "truck duck fuck" collapse
DEFAULT_WEIGHTS: Dict[str, float] = {
    "end_rhyme": 0.16,
    "internal_rhyme": 0.28,
    "rhyme_graph": 0.10,
    "multisyllabic": 0.09,
    "syllable_balance": 0.07,
    "stress_alignment": 0.10,
    "semantic": 0.11,
    "fluency": 0.12,
    "lexical_validity": 0.12,
    "ngram_fluency": 0.10,
    "novelty": 0.07,
    "weak_tail_penalty": -0.03,
    "repetition_penalty": -0.03,
    "rhyme_family_repetition_penalty": -0.15,
    "identical_line_penalty": -0.20,
    "near_duplicate_penalty": -0.15,
    "template_penalty": -0.08,
    "corpus_overlap_penalty": -0.20,
    "theme_penalty": -0.18,
}

# Cap aggregate fitness to prevent saturation (e.g. truck duck fuck scoring >1.0)
FITNESS_CAP = 0.95


def _score_end_rhyme(f1: Optional[LineFeatures], f2: Optional[LineFeatures]) -> float:
    """
    End rhyme quality [0,1] between line endings.
    Uses soft scaling (power 0.8) to avoid saturation - perfect rhyme stays 1.0
    but near-perfect (0.9) becomes ~0.92, giving evolution room to improve.
    Capped at 0.92 to avoid saturation.
    """
    if not f1 or not f2 or not f1.end_tail or not f2.end_tail:
        return 0.0
    raw = phonetic_similarity(f1.end_tail, f2.end_tail) ** 0.8
    return min(raw, 0.92)


def _score_internal_rhyme(
    f1: Optional[LineFeatures], f2: Optional[LineFeatures]
) -> float:
    """Internal rhyme density [0,1] across both lines."""
    if not f1 or not f2:
        return 0.0
    tails: List[Optional[PhoneticFeature]] = []
    if f1.internal_tails:
        tails.extend(f1.internal_tails[:-1])  # exclude end (counted in end_rhyme)
    if f2.internal_tails:
        tails.extend(f2.internal_tails[:-1])
    valid = [t for t in tails if t is not None]
    if len(valid) < 2:
        return 0.0
    matches = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if phonetic_similarity(valid[i], valid[j]) >= 0.6:
                matches += 1
    max_pairs = len(valid) * (len(valid) - 1) // 2
    return min(1.0, matches / max(1, max_pairs) * 2.0) if max_pairs else 0.0


def _score_multisyllabic(f1: Optional[LineFeatures], f2: Optional[LineFeatures]) -> float:
    """Reward phoneme tail overlap between end words [0,1]."""
    if not f1 or not f2 or not f1.tokens or not f2.tokens:
        return 0.0
    tail1 = extract_rhyme_tail(f1.tokens[-1])
    tail2 = extract_rhyme_tail(f2.tokens[-1])
    overlap = multisyllable_overlap(tail1, tail2)
    if overlap >= 4:
        return 1.0
    if overlap == 3:
        return 0.7
    if overlap == 2:
        return 0.4
    if overlap == 1:
        return 0.2
    return 0.0


def _score_syllable_balance(f1: Optional[LineFeatures], f2: Optional[LineFeatures]) -> float:
    """Balance between line syllable counts [0,1]. Perfect when equal."""
    if not f1 or not f2:
        return 0.0
    diff = abs(f1.syllable_count - f2.syllable_count)
    return max(0.0, 1.0 - diff / 6.0)  # 0 diff = 1.0, 6+ diff = 0


def _score_stress_alignment(f1: Optional[LineFeatures], f2: Optional[LineFeatures]) -> float:
    """Stress pattern alignment [0,1] between lines."""
    if not f1 or not f2 or not f1.stress_pattern or not f2.stress_pattern:
        return 0.5  # neutral
    p1, p2 = f1.stress_pattern, f2.stress_pattern
    min_len = min(len(p1), len(p2))
    if min_len == 0:
        return 0.5
    matches = sum(1 for i in range(min_len) if p1[i] == p2[i])
    return matches / min_len


def _score_semantic(
    individual: CoupletIndividual,
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
) -> float:
    """
    Semantic relevance to theme [0,1].
    When semantic_scorer is provided and theme_string is non-empty:
      keyword_score = overlap with prompt_keywords
      embedding_score = cosine_similarity(embed(theme), embed(line1 + line2)) mapped to [0,1]
      semantic_score = (1 - embedding_weight) * keyword_score + embedding_weight * embedding_score
    Otherwise: keyword overlap only (or 0.5 neutral when no prompt).
    """
    # Keyword score
    if prompt_keywords:
        text = f"{individual.line1} {individual.line2}".lower()
        words = set(w.lower() for w in text.split() if len(w) > 2)
        overlap = len(words & prompt_keywords) / max(1, len(prompt_keywords))
        keyword_score = min(1.0, overlap * 2.0)
        if keyword_score == 0.0:
            # No keyword overlap but prompt provided: return 0.15 instead of 0
            if not (semantic_scorer and theme_string and theme_string.strip()):
                return 0.15
    else:
        keyword_score = 0.5  # neutral when no prompt

    # Embedding score (when scorer and theme available)
    if semantic_scorer and theme_string and theme_string.strip():
        line_text = f"{individual.line1} {individual.line2}"
        try:
            cos_sim = semantic_scorer.score_pair(theme_string.strip(), line_text)
            # Map cosine from [-1, 1] to [0, 1]
            embedding_score = (cos_sim + 1.0) / 2.0
        except Exception:
            embedding_score = keyword_score  # fallback on error
        alpha = 1.0 - embedding_weight
        return alpha * keyword_score + embedding_weight * embedding_score
    return keyword_score


def _score_identical_line_penalty(individual: CoupletIndividual) -> float:
    """Penalty [0,1]: 1.0 if line1 == line2 (trivial), else 0."""
    if individual.line1.strip().lower() == individual.line2.strip().lower():
        return 1.0
    return 0.0


def _score_template_penalty(individual: CoupletIndividual) -> float:
    """Penalty when both lines end with same word and have similar length (rigid template)."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2 or not f1.tokens or not f2.tokens:
        return 0.0
    t1, t2 = f1.tokens, f2.tokens
    if t1[-1].lower() != t2[-1].lower():
        return 0.0
    # Same end word + similar length = template-like
    if abs(len(t1) - len(t2)) <= 1:
        return 0.5  # moderate penalty
    return 0.0


def _score_corpus_overlap_penalty(
    individual: CoupletIndividual,
    corpus_lines: Optional[List[str]] = None,
) -> float:
    """
    Penalty when couplet lines are too similar to raw corpus lines (overfitting).
    Returns [0,1]: 0 = no overlap, 1 = near-copy of corpus.
    Uses word overlap (threshold 0.68) and character-level SequenceMatcher (threshold 0.85).
    """
    if not corpus_lines or len(corpus_lines) < 2:
        return 0.0
    import re
    word_re = re.compile(r"[A-Za-z']+")
    t1 = set(word_re.findall(individual.line1.lower()))
    t2 = set(word_re.findall(individual.line2.lower()))
    if not t1 and not t2:
        word_overlap_result = 0.0
    else:
        max_overlap = 0.0
        for corp in corpus_lines:
            tc = set(word_re.findall(corp.lower()))
            if not tc:
                continue
            for ind_tokens in (t1, t2):
                if not ind_tokens:
                    continue
                overlap = len(ind_tokens & tc) / max(len(ind_tokens), len(tc))
                max_overlap = max(max_overlap, overlap)
        if max_overlap <= 0.68:
            word_overlap_result = 0.0
        else:
            word_overlap_result = min(1.0, (max_overlap - 0.68) / 0.32)

    # Character-level SequenceMatcher check
    max_ratio = 0.0
    for line in (individual.line1, individual.line2):
        line_lower = line.lower()
        for corp in corpus_lines:
            corp_lower = corp.lower()
            ratio = difflib.SequenceMatcher(None, line_lower, corp_lower).ratio()
            max_ratio = max(max_ratio, ratio)
    if max_ratio > 0.85:
        sequence_penalty = min(1.0, (max_ratio - 0.85) / 0.15)
    else:
        sequence_penalty = 0.0

    return max(word_overlap_result, sequence_penalty)


def _score_theme_penalty(
    individual: CoupletIndividual,
    prompt_keywords: Optional[Set[str]] = None,
) -> float:
    """
    Penalty when theme keywords provided but neither line contains any.
    Returns 0.8 when no theme token, 0 otherwise. Encourages theme relevance.
    """
    if not prompt_keywords:
        return 0.0
    text = f"{individual.line1} {individual.line2}".lower()
    words = set(w.lower() for w in text.split())
    if words & prompt_keywords:
        return 0.0
    return 0.8


def _score_near_duplicate_penalty(individual: CoupletIndividual) -> float:
    """Penalty [0,1] when line1 and line2 are nearly identical (e.g. 1 word diff)."""
    import re
    word_re = re.compile(r"[A-Za-z']+")
    t1 = set(word_re.findall(individual.line1.lower()))
    t2 = set(word_re.findall(individual.line2.lower()))
    if not t1 or not t2:
        return 0.0
    shared = len(t1 & t2)
    total = max(len(t1), len(t2))
    overlap = shared / total
    # Penalize when >75% of words overlap (near-duplicate)
    if overlap <= 0.75:
        return 0.0
    return min(1.0, (overlap - 0.75) / 0.25)  # 0.75->0, 1.0->1.0


def _score_fluency(individual: CoupletIndividual) -> float:
    """Simple fluency heuristic [0,1]. MVP: syllable flow + valid word ratio."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return 0.5
    # Prefer lines in 6-18 range (constraints handle rejection)
    s1, s2 = f1.syllable_count, f2.syllable_count
    in_range = 1.0 if (6 <= s1 <= 18 and 6 <= s2 <= 18) else 0.5
    balance = _score_syllable_balance(f1, f2)
    flow = 0.6 * in_range + 0.4 * balance
    # Penalize unknown/nonsense words (no pronunciation)
    valid_ratio = _valid_word_ratio(individual)
    return 0.7 * flow + 0.3 * valid_ratio


def _valid_word_ratio(individual: CoupletIndividual) -> float:
    """Fraction of content words with valid pronunciation [0,1]. Penalizes nonsense."""
    from evo_rhyme.phonetics import phones_for_word
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    tokens = (f1.tokens if (f1 := individual.features1) else []) + (
        f2.tokens if (f2 := individual.features2) else []
    )
    content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
    if not content:
        return 1.0
    valid = sum(1 for w in content if phones_for_word(w))
    return valid / len(content)


def _score_lexical_validity(
    individual: CoupletIndividual,
    corpus_vocab: Optional[Set[str]] = None,
) -> float:
    """
    Fraction of content tokens that appear in corpus vocab [0,1].
    At least 80% in-vocab -> 1.0; below 80% scales linearly.
    Returns 0.5 (neutral) when corpus_vocab is None.
    """
    if not corpus_vocab:
        return 0.5
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    f1, f2 = individual.features1, individual.features2
    tokens = (f1.tokens if f1 else []) + (f2.tokens if f2 else [])
    content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
    if not content:
        return 1.0
    in_vocab = sum(1 for w in content if w in corpus_vocab)
    ratio = in_vocab / len(content)
    # 80%+ -> 1.0; 0% -> 0.0; linear in between
    if ratio >= 0.8:
        return 1.0
    return ratio / 0.8


def _score_ngram_fluency(
    individual: CoupletIndividual,
    ngram_model: Optional[Any] = None,
) -> float:
    """
    Phrase plausibility from corpus bigram/trigram model [0,1].
    Penalizes lines like "damn the cup" or "grad list chazer plan" where
    adjacent word pairs never appear in corpus.
    Returns 0.5 (neutral) when ngram_model is None.
    """
    if ngram_model is None:
        return 0.5
    return ngram_model.score_couplet(individual.line1, individual.line2)


def _score_novelty(individual: CoupletIndividual) -> float:
    """Novelty / diversity [0,1]. Inverse of repetition."""
    rep = _repetition_penalty_raw(individual)
    return max(0.0, 1.0 - rep)


def _weak_tail_penalty_raw(f1: Optional[LineFeatures], f2: Optional[LineFeatures]) -> float:
    """Penalty [0,1] for weak (unstressed) end tails."""
    if not f1 or not f2:
        return 0.0
    penalty = 0.0
    if f1.end_tail and f1.end_tail.stress == 0:
        penalty += 0.5
    if f2.end_tail and f2.end_tail.stress == 0:
        penalty += 0.5
    return penalty


def _repetition_penalty_raw(individual: CoupletIndividual) -> float:
    """Repetition penalty [0,1] for repeated tokens."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return 0.0
    tokens = f1.tokens + f2.tokens
    if not tokens:
        return 0.0
    counts = Counter(t.lower() for t in tokens)
    max_count = max(counts.values()) if counts else 0
    if max_count <= 2:
        return 0.0
    return min(1.0, (max_count - 2) / 3.0)  # 3+ = penalty


def _score_rhyme_family_repetition_penalty(individual: CoupletIndividual) -> float:
    """
    Penalty [0,1] when 3+ content words in a row share the same rhyme tail.
    Catches "truck duck fuck in buck truck fuck" - phonetically dense nonsense.
    """
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return 0.0

    def _max_consecutive_same_family(tokens: List[str]) -> int:
        content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
        if len(content) < 3:
            return 0
        tails = [extract_rhyme_tail(w) for w in content]
        max_run = 0
        run = 1
        for i in range(1, len(tails)):
            if tails[i] and tails[i - 1] and tails[i] == tails[i - 1]:
                run += 1
            else:
                max_run = max(max_run, run)
                run = 1
        max_run = max(max_run, run)
        return max_run

    max1 = _max_consecutive_same_family(f1.tokens) if f1.tokens else 0
    max2 = _max_consecutive_same_family(f2.tokens) if f2.tokens else 0
    worst = max(max1, max2)
    if worst < 3:
        return 0.0
    # 3 consecutive -> 0.5 penalty, 4 -> 0.7, 5+ -> 1.0
    return min(1.0, 0.3 + (worst - 3) * 0.25)


def score_style_similarity(
    individual: CoupletIndividual,
    style_profile: StyleProfile,
) -> float:
    """
    Compare individual's style to profile. Returns [0, 1] similarity score.

    Compares: syllables per line, avg word length, word choice overlap,
    rhyme tail overlap with profile distributions.
    """
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return 0.5  # neutral when no features

    # Syllable similarity: closer to profile avg = higher
    ind_avg_syl = (f1.syllable_count + f2.syllable_count) / 2.0
    syl_diff = abs(ind_avg_syl - style_profile.avg_syllables_per_line)
    scale = max(2.0, style_profile.line_length_std or 2.0)
    syl_sim = max(0.0, 1.0 - syl_diff / scale)

    # Word length similarity
    tokens = f1.tokens + f2.tokens
    if not tokens:
        word_len_sim = 0.5
    else:
        ind_avg_wlen = sum(len(t) for t in tokens) / len(tokens)
        wlen_diff = abs(ind_avg_wlen - style_profile.avg_word_length)
        word_len_sim = max(0.0, 1.0 - wlen_diff / 2.0)

    # Word choice overlap: Jaccard with profile's top words
    profile_words = {w for w, _ in style_profile.word_freq_dist}
    ind_words = {t.lower() for t in tokens if len(t) > 1}
    if not ind_words:
        word_sim = 0.5
    elif not profile_words:
        word_sim = 0.5
    else:
        overlap = len(ind_words & profile_words) / len(ind_words)
        word_sim = min(1.0, overlap * 2.0)  # scale: 50% overlap -> 1.0

    # Rhyme tail overlap: do individual's end tails appear in profile's top tails?
    profile_tails = {t for t, _ in style_profile.rhyme_tail_dist}
    tail1 = extract_rhyme_tail(f1.tokens[-1]) if f1.tokens else None
    tail2 = extract_rhyme_tail(f2.tokens[-1]) if f2.tokens else None
    tails_in_profile = sum(1 for t in (tail1, tail2) if t and t in profile_tails)
    if not profile_tails:
        tail_sim = 0.5
    else:
        tail_sim = tails_in_profile / 2.0  # 0, 0.5, or 1.0

    # Blend components
    return 0.25 * syl_sim + 0.25 * word_len_sim + 0.25 * word_sim + 0.25 * tail_sim


def score_couplet(
    individual: CoupletIndividual,
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
    corpus_lines: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all component scores for a couplet.

    Returns dict with: end_rhyme, internal_rhyme, multisyllabic, syllable_balance,
    stress_alignment, semantic, fluency, lexical_validity, ngram_fluency, novelty,
    weak_tail_penalty, repetition_penalty, etc.
    """
    import re
    f1, f2 = individual.features1, individual.features2
    kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None

    # Build corpus vocab and ngram model from corpus_lines when available
    corpus_vocab: Optional[Set[str]] = None
    ngram_model: Optional[Any] = None
    if corpus_lines and len(corpus_lines) >= 2:
        word_re = re.compile(r"[A-Za-z']+")
        corpus_vocab = set()
        for line in corpus_lines[:2000]:
            corpus_vocab.update(w.lower() for w in word_re.findall(line.lower()))
        from evo_rhyme.ngram_fluency import get_ngram_model
        ngram_model = get_ngram_model(corpus_lines)

    rhyme_graph = (
        score_line_rhyme_graph(individual.line1) + score_line_rhyme_graph(individual.line2)
    ) / 2.0

    return {
        "end_rhyme": _score_end_rhyme(f1, f2),
        "internal_rhyme": _score_internal_rhyme(f1, f2),
        "rhyme_graph": rhyme_graph,
        "multisyllabic": _score_multisyllabic(f1, f2),
        "syllable_balance": _score_syllable_balance(f1, f2),
        "stress_alignment": _score_stress_alignment(f1, f2),
        "semantic": _score_semantic(
            individual,
            prompt_keywords=kw,
            theme_string=theme_string,
            semantic_scorer=semantic_scorer,
            embedding_weight=embedding_weight,
        ),
        "fluency": _score_fluency(individual),
        "lexical_validity": _score_lexical_validity(individual, corpus_vocab),
        "ngram_fluency": _score_ngram_fluency(individual, ngram_model),
        "novelty": _score_novelty(individual),
        "weak_tail_penalty": _weak_tail_penalty_raw(f1, f2),
        "repetition_penalty": _repetition_penalty_raw(individual),
        "rhyme_family_repetition_penalty": _score_rhyme_family_repetition_penalty(individual),
        "identical_line_penalty": _score_identical_line_penalty(individual),
        "near_duplicate_penalty": _score_near_duplicate_penalty(individual),
        "template_penalty": _score_template_penalty(individual),
        "corpus_overlap_penalty": _score_corpus_overlap_penalty(individual, corpus_lines),
        "theme_penalty": _score_theme_penalty(individual, prompt_keywords=kw),
    }


def _get_rhyme_family(individual: CoupletIndividual) -> tuple:
    """Rhyme family = (end_tail_line1, end_tail_line2) for grouping/diversity."""
    f1, f2 = individual.features1, individual.features2
    w1 = f1.tokens[-1] if f1 and f1.tokens else ""
    w2 = f2.tokens[-1] if f2 and f2.tokens else ""
    t1 = extract_rhyme_tail(w1) if w1 else None
    t2 = extract_rhyme_tail(w2) if w2 else None
    return (t1 or "", t2 or "")


def _rhyme_family_diversity_penalty(
    individual: CoupletIndividual,
    population: List[CoupletIndividual],
) -> float:
    """
    Penalty for overused rhyme families. When many share same (tail1, tail2),
    subtract up to 0.35 * tail_freq from fitness. Secondary penalty when tail1
    or tail2 appears as end tail in >40% of population. Total penalty capped at 0.5.
    """
    if not population:
        return 0.0
    family = _get_rhyme_family(individual)
    tail1, tail2 = family[0], family[1]
    count = sum(1 for ind in population if _get_rhyme_family(ind) == family)
    tail_freq = count / max(1, len(population))

    # Secondary penalty: tail overused as end tail in population (>40%)
    extra_penalty = 0.0
    n = len(population)
    if tail1:
        tail1_count = sum(
            1 for ind in population
            if (_get_rhyme_family(ind)[0] == tail1 or _get_rhyme_family(ind)[1] == tail1)
        )
        if tail1_count / n > 0.4:
            extra_penalty += 0.1
    if tail2:
        tail2_count = sum(
            1 for ind in population
            if (_get_rhyme_family(ind)[0] == tail2 or _get_rhyme_family(ind)[1] == tail2)
        )
        if tail2_count / n > 0.4:
            extra_penalty += 0.1

    penalty = -0.35 * tail_freq - extra_penalty
    return max(penalty, -0.5)


def _get_skeleton(individual: CoupletIndividual, num_tokens: int = 4) -> str:
    """First num_tokens content words from each line, lowercased, joined. For shell diversity."""
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    f1, f2 = individual.features1, individual.features2
    tokens = (f1.tokens if f1 else []) + (f2.tokens if f2 else [])
    content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
    return " ".join(content[:num_tokens]) if content else ""


def _repeated_shell_penalty(
    individual: CoupletIndividual,
    population: List[CoupletIndividual],
) -> float:
    """
    Penalty when too many elites share the same 4-token skeleton.
    Blocks "truck duck truck in" from dominating the population.
    Returns negative value: -0.12 * shell_freq when freq > 0.3.
    """
    if not population:
        return 0.0
    skeleton = _get_skeleton(individual, num_tokens=4)
    if not skeleton:
        return 0.0
    count = sum(1 for ind in population if _get_skeleton(ind, 4) == skeleton)
    freq = count / max(1, len(population))
    if freq <= 0.25:
        return 0.0
    # 25%+ sharing -> up to -0.35 penalty
    return -0.35 * min(1.0, (freq - 0.25) / 0.75)


# ---------------------------------------------------------------------------
# Verse fitness (4-line)
# ---------------------------------------------------------------------------

VERSE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "rhyme_scheme_score": 0.30,
    "internal_rhyme": 0.18,
    "syllable_balance": 0.12,
    "fluency": 0.18,
    "semantic": 0.10,
    "identical_line_penalty": -0.20,
    "template_penalty": -0.12,
    "repetition_penalty": -0.10,
    "near_duplicate_penalty": -0.15,
}


def _score_verse_rhyme_scheme(
    features: Optional[VerseFeatures],
    scheme: str,
) -> float:
    """
    Rhyme scheme score [0,1].
    AABB: reward lines 1-2 and 3-4 matching end tails.
    ABAB: reward lines 1-3 and 2-4 matching end tails.
    """
    if not features or len(features.end_tails) != 4:
        return 0.0
    tails = features.end_tails
    scheme = (scheme or "AABB").upper()
    if scheme == "AABB":
        pair1 = _score_end_rhyme(
            LineFeatures("", [], [], 0, [], tails[0], []),
            LineFeatures("", [], [], 0, [], tails[1], []),
        )
        pair2 = _score_end_rhyme(
            LineFeatures("", [], [], 0, [], tails[2], []),
            LineFeatures("", [], [], 0, [], tails[3], []),
        )
        return (pair1 + pair2) / 2.0
    if scheme == "ABAB":
        pair1 = _score_end_rhyme(
            LineFeatures("", [], [], 0, [], tails[0], []),
            LineFeatures("", [], [], 0, [], tails[2], []),
        )
        pair2 = _score_end_rhyme(
            LineFeatures("", [], [], 0, [], tails[1], []),
            LineFeatures("", [], [], 0, [], tails[3], []),
        )
        return (pair1 + pair2) / 2.0
    return 0.0


def _score_verse_internal_rhyme_simple(features: Optional[VerseFeatures]) -> float:
    """Internal rhyme density across all 4 lines."""
    if not features or len(features.tokens_per_line) != 4:
        return 0.0
    all_tails: List[Optional[PhoneticFeature]] = []
    for i, tokens in enumerate(features.tokens_per_line):
        if not tokens:
            continue
        line_text = " ".join(tokens)
        from evo_rhyme.individual import _analyze_line
        lf = _analyze_line(line_text)
        if lf.internal_tails:
            all_tails.extend(lf.internal_tails[:-1])
    valid = [t for t in all_tails if t is not None]
    if len(valid) < 2:
        return 0.0
    matches = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if phonetic_similarity(valid[i], valid[j]) >= 0.6:
                matches += 1
    max_pairs = len(valid) * (len(valid) - 1) // 2
    return min(1.0, matches / max(1, max_pairs) * 2.0) if max_pairs else 0.0


def _score_verse_syllable_balance(features: Optional[VerseFeatures]) -> float:
    """Balance across 4 lines: max pairwise diff <= 2 ideal."""
    if not features or len(features.syllable_counts) != 4:
        return 0.0
    counts = features.syllable_counts
    diffs = [
        abs(counts[0] - counts[1]),
        abs(counts[1] - counts[2]),
        abs(counts[2] - counts[3]),
        abs(counts[0] - counts[2]),
        abs(counts[1] - counts[3]),
    ]
    max_diff = max(diffs) if diffs else 0
    return max(0.0, 1.0 - max_diff / 6.0)


def _score_verse_fluency(individual: VerseIndividual) -> float:
    """Fluency: syllable flow + valid word ratio across all lines."""
    f = individual.features
    if not f or len(f.syllable_counts) != 4:
        return 0.5
    in_range = sum(1 for s in f.syllable_counts if 6 <= s <= 18) / 4.0
    balance = _score_verse_syllable_balance(f)
    flow = 0.6 * in_range + 0.4 * balance
    from evo_rhyme.phonetics import phones_for_word
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    tokens = []
    for t in f.tokens_per_line:
        tokens.extend(t)
    content = [w.lower() for w in tokens if w.lower() not in stopwords and len(w) > 1]
    if not content:
        valid_ratio = 1.0
    else:
        valid_ratio = sum(1 for w in content if phones_for_word(w)) / len(content)
    return 0.7 * flow + 0.3 * valid_ratio


def _score_verse_semantic(
    individual: VerseIndividual,
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
) -> float:
    """Semantic relevance to theme for 4-line verse."""
    text = " ".join(individual.lines).lower()
    if prompt_keywords:
        words = set(w.lower() for w in text.split() if len(w) > 2)
        overlap = len(words & prompt_keywords) / max(1, len(prompt_keywords))
        keyword_score = min(1.0, overlap * 2.0)
    else:
        keyword_score = 0.5
    if semantic_scorer and theme_string and theme_string.strip():
        try:
            cos_sim = semantic_scorer.score_pair(theme_string.strip(), text)
            embedding_score = (cos_sim + 1.0) / 2.0
        except Exception:
            embedding_score = keyword_score
        alpha = 1.0 - embedding_weight
        return alpha * keyword_score + embedding_weight * embedding_score
    return keyword_score


def _score_verse_identical_line_penalty(individual: VerseIndividual) -> float:
    """Penalty when any two lines are identical."""
    lines = [l.strip().lower() for l in individual.lines]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i] == lines[j]:
                return 1.0
    return 0.0


def _score_verse_template_penalty(
    individual: VerseIndividual,
    prompt_keywords: Optional[Set[str]] = None,
) -> float:
    """Penalty when 3+ lines share same end word or theme keywords dominate."""
    import re
    word_re = re.compile(r"[A-Za-z']+")
    end_words: List[str] = []
    all_tokens: List[str] = []
    for line in individual.lines:
        words = word_re.findall(line.lower())
        if words:
            end_words.append(words[-1])
        all_tokens.extend(words)
    penalty = 0.0
    if len(end_words) >= 3:
        end_counts = Counter(end_words)
        if end_counts.most_common(1)[0][1] >= 3:
            penalty = 0.5
    if prompt_keywords:
        theme_count = sum(1 for t in all_tokens if t in prompt_keywords)
        if theme_count > 5:
            penalty = max(penalty, 0.3)
    return penalty


def _score_verse_repetition_penalty(individual: VerseIndividual) -> float:
    """Penalty when any content word appears 3+ times across 4 lines."""
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    tokens = []
    if individual.features and individual.features.tokens_per_line:
        for t in individual.features.tokens_per_line:
            tokens.extend(t)
    else:
        import re
        word_re = re.compile(r"[A-Za-z']+")
        for line in individual.lines:
            tokens.extend(word_re.findall(line.lower()))
    content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
    if not content:
        return 0.0
    counts = Counter(content)
    max_count = max(counts.values()) if counts else 0
    if max_count <= 2:
        return 0.0
    return min(1.0, (max_count - 2) / 3.0)


def _score_verse_near_duplicate_penalty(individual: VerseIndividual) -> float:
    """Penalty when any two lines have word overlap > 0.78."""
    import re
    word_re = re.compile(r"[A-Za-z']+")
    line_sets: List[Set[str]] = []
    for line in individual.lines:
        words = set(word_re.findall(line.lower()))
        line_sets.append(words)
    max_penalty = 0.0
    for i in range(len(line_sets)):
        for j in range(i + 1, len(line_sets)):
            t1, t2 = line_sets[i], line_sets[j]
            if not t1 or not t2:
                continue
            overlap = len(t1 & t2) / max(len(t1), len(t2))
            if overlap > 0.78:
                penalty = min(1.0, (overlap - 0.75) / 0.25)
                max_penalty = max(max_penalty, penalty)
    return max_penalty


def score_verse(
    individual: VerseIndividual,
    scheme: str = "AABB",
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
) -> Dict[str, float]:
    """Compute all component scores for a 4-line verse."""
    f = individual.features
    kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None
    return {
        "rhyme_scheme_score": _score_verse_rhyme_scheme(f, scheme),
        "internal_rhyme": _score_verse_internal_rhyme_simple(f),
        "syllable_balance": _score_verse_syllable_balance(f),
        "fluency": _score_verse_fluency(individual),
        "semantic": _score_verse_semantic(
            individual,
            prompt_keywords=kw,
            theme_string=theme_string,
            semantic_scorer=semantic_scorer,
            embedding_weight=embedding_weight,
        ),
        "identical_line_penalty": _score_verse_identical_line_penalty(individual),
        "template_penalty": _score_verse_template_penalty(individual, prompt_keywords=kw),
        "repetition_penalty": _score_verse_repetition_penalty(individual),
        "near_duplicate_penalty": _score_verse_near_duplicate_penalty(individual),
    }


def compute_verse_fitness(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate verse fitness from weighted sum of component scores."""
    w = weights or VERSE_DEFAULT_WEIGHTS
    total = 0.0
    for key, weight in w.items():
        if key in scores:
            total += weight * scores[key]
    return total


def compute_fitness(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    individual: Optional[CoupletIndividual] = None,
    population: Optional[List[CoupletIndividual]] = None,
    style_profile: Optional["StyleProfile"] = None,
    style_weight: float = 0.0,
) -> float:
    """
    Aggregate fitness from weighted sum of component scores.

    Uses DEFAULT_WEIGHTS if weights not provided.
    When individual and population are provided, adds rhyme_family_diversity_penalty
    and repeated_shell_penalty.
    Caps result at FITNESS_CAP to prevent saturation.
    """
    w = weights or DEFAULT_WEIGHTS
    total = 0.0
    for key, weight in w.items():
        if key in scores:
            total += weight * scores[key]
    if individual is not None and population is not None:
        total += _rhyme_family_diversity_penalty(individual, population)
        total += _repeated_shell_penalty(individual, population)
    if style_profile is not None and style_weight > 0 and individual is not None:
        total += style_weight * score_style_similarity(individual, style_profile)
    return min(total, FITNESS_CAP)
