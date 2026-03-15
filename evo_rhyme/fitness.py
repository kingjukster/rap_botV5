"""
evo_rhyme/fitness.py

Fitness scoring for couplet individuals. Computes component scores and
aggregate fitness from weighted sum.
"""

from __future__ import annotations

import difflib
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from evo_rhyme.individual import CoupletIndividual, LineFeatures, VerseIndividual, VerseFeatures
from evo_rhyme.phonetics import (
    PhoneticFeature,
    extract_rhyme_tail,
    multisyllable_overlap,
    phonetic_similarity,
)
from evo_rhyme.rhyme_graph import score_line_rhyme_graph
from evo_rhyme.scoring.coherence import score_coherence
from evo_rhyme.scoring.line_penalty import score_cliche_penalty, get_ngram_index
from evo_rhyme.scoring.punchline import score_punchline
from evo_rhyme.scoring.rhyme_chain import score_rhyme_chain_density
from evo_rhyme.scoring.rhyme_graph_network import score_rhyme_graph_metrics
from evo_rhyme.style_profile import StyleProfile

# MVP weights - rebalanced to avoid early saturation; max fitness rarely achieved
# lexical_validity + ngram_fluency protect against rhyme-optimized nonsense
# rhyme_family_repetition_penalty + repeated_shell_penalty block "truck duck fuck" collapse
#
# CRITICAL: ngram_fluency must be strong - it's the only signal that checks whether
# phrase structure appears in natural language. Without it, evolution exploits the
# loophole: valid words + rhyme + theme tokens = high score, but nonsense sequences.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "end_rhyme": 0.18,
    "internal_rhyme": 0.16,
    "rhyme_graph": 0.12,
    "multisyllabic": 0.08,
    "syllable_balance": 0.06,
    "stress_alignment": 0.10,
    "semantic": 0.10,
    "fluency": 0.08,
    "lexical_validity": 0.06,
    "ngram_fluency": 0.15,  # Strong: penalizes unnatural phrase sequences
    "novelty": 0.05,
    "weak_tail_penalty": -0.03,
    "repetition_penalty": -0.08,
    "theme_word_repetition_penalty": -0.12,
    "rhyme_family_repetition_penalty": -0.15,
    "identical_line_penalty": -0.20,
    "near_duplicate_penalty": -0.15,
    "template_penalty": -0.08,
    "corpus_overlap_penalty": -0.20,
    "theme_penalty": -0.18,
}

# Cap aggregate fitness to prevent saturation (e.g. truck duck fuck scoring >1.0)
FITNESS_CAP = 0.95

# Minimum ngram_fluency to survive - rejects candidates with unnatural phrase structure.
# When corpus is available, ngram_fluency < this means phrase never appears in real language.
NGRAM_FLOOR = 0.2

_VERSE_SCORE_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}
_VERSE_SCORE_CACHE_MAX = 20000


def _norm_verse_key(lines: List[str], scheme: str) -> Tuple[str, str]:
    text = "\n".join(line.strip().lower() for line in lines)
    return (scheme.upper(), text)


def _cache_put(key: Tuple[str, str], scores: Dict[str, float]) -> None:
    if len(_VERSE_SCORE_CACHE) >= _VERSE_SCORE_CACHE_MAX:
        # Lightweight bounded cache without external dependency.
        _VERSE_SCORE_CACHE.clear()
    _VERSE_SCORE_CACHE[key] = dict(scores)


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


def _score_theme_word_repetition_penalty(
    individual: CoupletIndividual,
    prompt_keywords: Optional[Set[str]] = None,
) -> float:
    """
    Penalty when theme keywords appear more than once in a couplet.
    E.g. "survival got tight but survival got" -> penalize repeated "survival".
    Returns [0,1]: 0 = no theme repeat, 1 = heavy theme word stuffing.
    """
    if not prompt_keywords:
        return 0.0
    import re
    word_re = re.compile(r"[A-Za-z']+")
    tokens = word_re.findall((individual.line1 + " " + individual.line2).lower())
    theme_counts = Counter(t for t in tokens if t in prompt_keywords)
    if not theme_counts:
        return 0.0
    max_theme_repeats = max(theme_counts.values())
    if max_theme_repeats <= 1:
        return 0.0
    return min(1.0, (max_theme_repeats - 1) / 2.0)  # 2->0.5, 3->1.0, 4+->1.0


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
    """Repetition penalty [0,1] for repeated content tokens (2+ = penalty)."""
    f1, f2 = individual.features1, individual.features2
    if not f1 or not f2:
        return 0.0
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    content = [t.lower() for t in (f1.tokens + f2.tokens) if t.lower() not in stopwords]
    if not content:
        return 0.0
    counts = Counter(content)
    max_count = max(counts.values()) if counts else 0
    if max_count <= 1:
        return 0.0
    return min(1.0, (max_count - 1) / 2.0)  # 2->0.5, 3->1.0, 4+->1.0


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
    use_lm_fluency: bool = False,
    lm_fluency_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all component scores for a couplet.

    Returns dict with: end_rhyme, internal_rhyme, multisyllabic, syllable_balance,
    stress_alignment, semantic, fluency, lexical_validity, ngram_fluency, novelty,
    weak_tail_penalty, repetition_penalty, etc.

    When use_lm_fluency=True, blends corpus ngram score with LM perplexity score
    for stronger phrase plausibility (catches nonsense like "survival pop rough").
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

    # Optional LM perplexity scorer for phrase plausibility (blends with ngram)
    lm_scorer: Optional[Any] = None
    if use_lm_fluency:
        try:
            from evo_rhyme.lm_fluency import get_lm_scorer
            lm_scorer = get_lm_scorer()
        except Exception:
            lm_scorer = None

    ngram_score = _score_ngram_fluency(individual, ngram_model)
    if lm_scorer is not None:
        lm_score = lm_scorer.score_couplet(individual.line1, individual.line2)
        ngram_fluency_val = (1.0 - lm_fluency_weight) * ngram_score + lm_fluency_weight * lm_score
    else:
        ngram_fluency_val = ngram_score

    rhyme_graph = (
        score_line_rhyme_graph(individual.line1) + score_line_rhyme_graph(individual.line2)
    ) / 2.0

    scores: Dict[str, float] = {
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
        "ngram_fluency": ngram_fluency_val,
        "novelty": _score_novelty(individual),
        "weak_tail_penalty": _weak_tail_penalty_raw(f1, f2),
        "repetition_penalty": _repetition_penalty_raw(individual),
        "theme_word_repetition_penalty": _score_theme_word_repetition_penalty(
            individual, prompt_keywords=kw
        ),
        "rhyme_family_repetition_penalty": _score_rhyme_family_repetition_penalty(individual),
        "identical_line_penalty": _score_identical_line_penalty(individual),
        "near_duplicate_penalty": _score_near_duplicate_penalty(individual),
        "template_penalty": _score_template_penalty(individual),
        "corpus_overlap_penalty": _score_corpus_overlap_penalty(individual, corpus_lines),
        "theme_penalty": _score_theme_penalty(individual, prompt_keywords=kw),
    }

    lines = [individual.line1, individual.line2]
    try:
        scores["coherence"] = score_coherence(lines)
    except Exception:
        scores["coherence"] = 0.0

    try:
        scores["punchline"] = score_punchline(lines)
    except Exception:
        scores["punchline"] = 0.0

    return scores


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
    "rhyme_scheme_score": 0.15,
    "internal_rhyme": 0.07,
    "rhyme_chain_density": 0.08,
    "global_rhyme_chain_score": 0.08,
    "internal_chain_score": 0.06,
    "rhyme_graph_density": 0.05,
    "rhyme_graph_cluster_coeff": 0.04,
    "rhyme_graph_chain_length": 0.04,
    "syllable_balance": 0.05,
    "fluency": 0.08,
    "lm_fluency": 0.12,
    "semantic": 0.08,
    "lexical_validity": 0.06,
    "coherence": 0.18,
    "punchline": 0.08,
    "identical_line_penalty": -0.40,
    "template_penalty": -0.15,
    "repetition_penalty": -0.12,
    "near_duplicate_penalty": -0.30,
    "filler_line_penalty": -0.30,
    "line_phrase_penalty": -0.12,
    "corpus_overlap_penalty": -0.20,
    "garbled_line_penalty": -0.35,
    "cliche_penalty": -0.25,
    "structural_repetition_penalty": -0.25,
    "cross_verse_repetition_penalty": -0.20,
    "novelty": 0.25,
    "flow_alignment": 0.10,
    "flow_continuity_score": 0.08,
    "style_adherence": 0.06,
    "prompt_adherence": 0.05,
}


def _score_verse_rhyme_scheme(
    features: Optional[VerseFeatures],
    scheme: str,
) -> float:
    """
    Rhyme scheme score [0,1].
    Supported schemes: AABB, ABAB, ABBA, ABCB, AABA, AAAA.
    """
    if not features or len(features.end_tails) != 4:
        return 0.0
    tails = features.end_tails
    scheme = (scheme or "AABB").upper()

    def _lf(idx: int) -> LineFeatures:
        return LineFeatures("", [], [], 0, [], tails[idx], [])

    if scheme == "AABB":
        pair1 = _score_end_rhyme(_lf(0), _lf(1))
        pair2 = _score_end_rhyme(_lf(2), _lf(3))
        return (pair1 + pair2) / 2.0
    if scheme == "ABAB":
        pair1 = _score_end_rhyme(_lf(0), _lf(2))
        pair2 = _score_end_rhyme(_lf(1), _lf(3))
        return (pair1 + pair2) / 2.0
    if scheme == "ABBA":
        pair1 = _score_end_rhyme(_lf(0), _lf(3))
        pair2 = _score_end_rhyme(_lf(1), _lf(2))
        return (pair1 + pair2) / 2.0
    if scheme == "ABCB":
        return _score_end_rhyme(_lf(1), _lf(3))
    if scheme == "AABA":
        p01 = _score_end_rhyme(_lf(0), _lf(1))
        p03 = _score_end_rhyme(_lf(0), _lf(3))
        p13 = _score_end_rhyme(_lf(1), _lf(3))
        return (p01 + p03 + p13) / 3.0
    if scheme == "AAAA":
        total = 0.0
        count = 0
        for i in range(4):
            for j in range(i + 1, 4):
                total += _score_end_rhyme(_lf(i), _lf(j))
                count += 1
        return total / count
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


def _score_global_rhyme_chain(lines: List[str]) -> float:
    tails: List[str] = []
    for line in lines:
        words = line.strip().split()
        if not words:
            continue
        tail = extract_rhyme_tail(words[-1].lower())
        tails.append(str(tail) if tail else "")
    if not tails:
        return 0.0
    counts = Counter(t for t in tails if t)
    if not counts:
        return 0.0
    max_chain = max(counts.values())
    return max_chain / max(1, len(lines))


def _score_internal_chain(lines: List[str]) -> float:
    try:
        from evo_rhyme.scoring.rhyme_chain import score_chain_structure
        cs = score_chain_structure(lines)
        return float(cs.get("internal_chain_density", 0.0))
    except Exception:
        return 0.0


def _score_flow_continuity(individual: VerseIndividual) -> float:
    counts: List[int] = []
    if individual.features and individual.features.syllable_counts:
        counts = list(individual.features.syllable_counts)
    else:
        try:
            from evo_rhyme.phonetics import syllable_count_line
            counts = [syllable_count_line(l) for l in individual.lines]
        except Exception:
            counts = []
    if len(counts) < 2:
        return 0.5
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = math.sqrt(var)
    return max(0.0, min(1.0, 1.0 - std / 3.0))


def _score_style_adherence(individual: VerseIndividual) -> float:
    labels = (individual.metadata or {}).get("style_genome_labels")
    if not isinstance(labels, dict):
        return 0.5
    parts: List[float] = []
    ir = (individual.scores or {}).get("internal_rhyme")
    if ir is not None:
        target = {
            "low": 0.10, "medium": 0.20, "high": 0.30, "very_high": 0.40,
        }.get(labels.get("internal_rhyme_density", "medium"), 0.20)
        parts.append(max(0.0, 1.0 - abs(float(ir) - target) / 0.35))
    syl = (individual.features.syllable_counts if individual.features else []) or []
    if syl:
        avg_syl = sum(syl) / len(syl)
        target_syl = {
            "sparse": 9.0, "normal": 11.0, "dense": 13.0, "very_dense": 14.0,
        }.get(labels.get("syllable_density", "normal"), 11.0)
        parts.append(max(0.0, 1.0 - abs(avg_syl - target_syl) / 4.0))
    metaphor = (individual.scores or {}).get("metaphor_density", (individual.scores or {}).get("cliche_penalty", 0.0))
    if metaphor is not None:
        target_m = {
            "literal": 0.05, "some": 0.20, "rich": 0.45, "very_rich": 0.60,
        }.get(labels.get("metaphor_density", "some"), 0.20)
        parts.append(max(0.0, 1.0 - abs(float(metaphor) - target_m) / 0.6))
    if not parts:
        return 0.5
    return sum(parts) / len(parts)


def _score_prompt_adherence(individual: VerseIndividual) -> float:
    pg = (individual.metadata or {}).get("prompt_genome")
    if not isinstance(pg, dict):
        return 0.5
    strictness = float(pg.get("strictness", 0.5))
    novelty_bias = float(pg.get("novelty_bias", 0.5))
    rep = (individual.scores or {}).get("repetition_penalty", 0.0)
    novelty = (individual.scores or {}).get("novelty", 0.5)
    structure = (individual.scores or {}).get("syllable_balance", 0.5)
    if strictness >= 0.6:
        strict_score = structure
    else:
        strict_score = 1.0 - abs(structure - 0.65)
    novelty_score = (novelty * 0.7) + ((1.0 - min(1.0, rep)) * 0.3)
    return (strictness * strict_score) + ((1.0 - strictness) * (novelty_bias * novelty_score + (1.0 - novelty_bias) * 0.6))


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
    embedding_weight: float = 0.4,
) -> float:
    """
    Semantic relevance to theme for 4-line verse.
    Applies per-line theme coverage scaling: when fewer than 2 of 4 lines
    contain any theme keyword, the semantic score is reduced proportionally.
    """
    text = " ".join(individual.lines).lower()
    if prompt_keywords:
        words = set(w.lower() for w in text.split() if len(w) > 2)
        overlap = len(words & prompt_keywords) / max(1, len(prompt_keywords))
        keyword_score = overlap
    else:
        keyword_score = 0.5

    if prompt_keywords:
        lines_with_kw = sum(
            1 for line in individual.lines
            if any(kw in set(w.lower() for w in line.split()) for kw in prompt_keywords)
        )
        distribution_factor = lines_with_kw / max(1, len(individual.lines))
        keyword_score *= (0.5 + 0.5 * distribution_factor)

    if semantic_scorer and theme_string and theme_string.strip():
        try:
            line_scores = []
            for line in individual.lines:
                cos_sim = semantic_scorer.score_pair(theme_string.strip(), line.lower())
                line_scores.append((cos_sim + 1.0) / 2.0)
            embedding_score = sum(line_scores) / len(line_scores) if line_scores else keyword_score
        except Exception:
            embedding_score = keyword_score
        alpha = 1.0 - embedding_weight
        base_score = alpha * keyword_score + embedding_weight * embedding_score
    else:
        base_score = keyword_score

    if prompt_keywords:
        lines_with_theme = sum(
            1 for line in individual.lines
            if any(kw in set(w.lower() for w in line.split()) for kw in prompt_keywords)
        )
        theme_coverage = lines_with_theme / max(1, len(individual.lines))
        if theme_coverage < 0.5:
            base_score *= theme_coverage * 2.0

    return base_score


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


def _score_verse_lexical_validity(individual: VerseIndividual) -> float:
    """Average valid-word ratio across all 4 lines [0,1]."""
    from evo_rhyme.phonetics import phones_for_word
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from", "go",
                 "had", "he", "her", "him", "his", "i", "in", "is", "it", "me",
                 "my", "no", "of", "on", "or", "our", "out", "she", "so", "that",
                 "the", "them", "then", "there", "they", "this", "to", "was",
                 "we", "you", "got", "like", "just", "all", "say", "said"}
    f = individual.features
    if not f or not f.tokens_per_line:
        return 0.5
    ratios = []
    for tokens in f.tokens_per_line:
        content = [t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 1]
        if not content:
            ratios.append(1.0)
        else:
            valid = sum(1 for w in content if phones_for_word(w))
            ratios.append(valid / len(content))
    return sum(ratios) / len(ratios) if ratios else 0.5


def _load_filler_tokens() -> set:
    """Load scat/filler tokens from data/evo_rhyme/filler_tokens.txt."""
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "evo_rhyme" / "filler_tokens.txt"
    if not path.exists():
        return set()
    tokens: set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = line.strip().lower()
            if t and not t.startswith("#"):
                tokens.add(t)
    return tokens


_FILLER_TOKENS: Optional[set] = None


def _get_filler_tokens() -> set:
    global _FILLER_TOKENS
    if _FILLER_TOKENS is None:
        _FILLER_TOKENS = _load_filler_tokens()
    return _FILLER_TOKENS


def _score_verse_filler_line_penalty(individual: VerseIndividual) -> float:
    """
    Penalty [0,1] for scat/filler lines: lines where >50% of tokens are
    single-char or in the filler set. Returns fraction of lines that are filler.
    """
    import re
    filler_set = _get_filler_tokens()
    word_re = re.compile(r"[A-Za-z']+")
    filler_count = 0
    for line in individual.lines:
        tokens = word_re.findall(line.lower())
        if not tokens:
            continue
        filler_hits = sum(
            1 for t in tokens if len(t) <= 1 or t in filler_set
        )
        if filler_hits / len(tokens) > 0.5:
            filler_count += 1
    return filler_count / max(1, len(individual.lines))


def _score_verse_line_phrase_penalty(individual: VerseIndividual) -> float:
    """Penalty for lines containing banned phrases from penalty_phrases.txt."""
    from evo_rhyme.scoring.line_penalty import apply_line_penalty
    return apply_line_penalty(individual.lines)


def _score_verse_corpus_overlap_penalty(
    individual: VerseIndividual,
    corpus_lines: Optional[List[str]] = None,
) -> float:
    """
    Penalty when verse lines are too similar to raw corpus lines.
    Checks all 4 lines via word overlap (threshold 0.68) and
    character-level SequenceMatcher (threshold 0.85).
    Returns max penalty across all lines [0,1].
    """
    if not corpus_lines or len(corpus_lines) < 2:
        return 0.0
    import re
    word_re = re.compile(r"[A-Za-z']+")
    line_token_sets = [set(word_re.findall(l.lower())) for l in individual.lines]

    max_word_overlap = 0.0
    for ind_tokens in line_token_sets:
        if not ind_tokens:
            continue
        for corp in corpus_lines:
            tc = set(word_re.findall(corp.lower()))
            if not tc:
                continue
            overlap = len(ind_tokens & tc) / max(len(ind_tokens), len(tc))
            max_word_overlap = max(max_word_overlap, overlap)
    if max_word_overlap <= 0.68:
        word_penalty = 0.0
    else:
        word_penalty = min(1.0, (max_word_overlap - 0.68) / 0.32)

    max_ratio = 0.0
    for line in individual.lines:
        line_lower = line.lower()
        for corp in corpus_lines:
            ratio = difflib.SequenceMatcher(None, line_lower, corp.lower()).ratio()
            max_ratio = max(max_ratio, ratio)
    if max_ratio > 0.85:
        seq_penalty = min(1.0, (max_ratio - 0.85) / 0.15)
    else:
        seq_penalty = 0.0

    return max(word_penalty, seq_penalty)


def _score_verse_structural_repetition_penalty(individual: VerseIndividual) -> float:
    """Penalty [0,1] for lines sharing the same syntactic template.

    Extracts the function-word subsequence from each line (dropping content words).
    Sequences with 3+ function words that repeat across lines are penalized.
    Also detects the common rap template "X prep possessive Y" opener pattern.
    """
    import re
    _FUNC_WORDS = frozenset({
        "i", "me", "my", "mine", "we", "our", "you", "your", "he", "she",
        "it", "they", "them", "his", "her", "its", "a", "an", "the",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "up", "out", "but", "and", "or", "so", "is", "am", "are",
        "was", "be", "been", "do", "don't", "can't", "won't", "ain't",
        "not", "no", "if", "that", "this", "like", "just", "got",
    })
    _PREPS = frozenset({"in", "on", "at", "of", "from", "with", "by", "to", "for"})
    _POSS = frozenset({"my", "your", "his", "her", "our", "their", "its"})
    word_re = re.compile(r"[a-z']+")

    func_seqs: list[str] = []
    opener_templates: list[str] = []

    for line in individual.lines:
        tokens = word_re.findall(line.lower())
        func_tokens = [t for t in tokens if t in _FUNC_WORDS]
        if len(func_tokens) >= 3:
            func_seqs.append(" ".join(func_tokens))
        else:
            func_seqs.append("")

        # Detect "X prep poss Y" opener: 2nd or 3rd token is prep, next is possessive
        if len(tokens) >= 3:
            for start in range(min(2, len(tokens) - 2)):
                if tokens[start] in _PREPS and tokens[start + 1] in _POSS:
                    opener_templates.append(f"{tokens[start]} {tokens[start+1]}")
                    break

    penalty = 0.0

    # Penalize repeated function-word sequences
    from collections import Counter
    seq_counts = Counter(s for s in func_seqs if s)
    if seq_counts:
        max_dup = max(seq_counts.values())
        if max_dup >= 3:
            penalty = max(penalty, 1.0)
        elif max_dup >= 2:
            penalty = max(penalty, 0.6)

    # Penalize repeated "prep+possessive" openers (the dominant attractor pattern)
    if len(opener_templates) >= 2:
        opener_counts = Counter(opener_templates)
        max_opener_dup = max(opener_counts.values())
        if max_opener_dup >= 3:
            penalty = max(penalty, 1.0)
        elif max_opener_dup >= 2:
            penalty = max(penalty, 0.5)
        # Even different prep+poss openers are formulaic if most lines use them
        if len(opener_templates) >= 3:
            penalty = max(penalty, 0.7)

    # Template identity check: penalize when multiple lines used the same template
    if hasattr(individual, 'template_ids') and individual.template_ids:
        template_counts = Counter(t for t in individual.template_ids if t)
        if template_counts:
            max_template_dup = max(template_counts.values())
            if max_template_dup >= 3:
                penalty = max(penalty, 1.0)
            elif max_template_dup >= 2:
                penalty = max(penalty, 0.4)

    return penalty


def _score_verse_lm_fluency(individual: VerseIndividual) -> float:
    """LM perplexity-based fluency [0,1]. Uses distilgpt2 to detect gibberish."""
    try:
        from evo_rhyme.lm_fluency import get_lm_scorer
        scorer = get_lm_scorer()
        if scorer is None:
            return 0.5
        scores = []
        for line in individual.lines:
            scores.append(scorer.score_line(line))
        return sum(scores) / len(scores) if scores else 0.5
    except Exception:
        return 0.5


def _score_verse_garbled_line_penalty(individual: VerseIndividual) -> float:
    """
    Penalty [0,1] for garbled/nonsensical lines.
    Detects: consecutive duplicate words, broken contractions ("i m", "s the"),
    articles/prepositions followed by articles ("the the", "of of"),
    and lines that are mostly stopwords with no content.
    Returns fraction of lines flagged as garbled.
    """
    import re
    _GARBLED_PATTERNS = [
        re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE),
        re.compile(r"\bi\s+m\b", re.IGNORECASE),
        re.compile(r"\bi\s+ve\b", re.IGNORECASE),
        re.compile(r"\bi\s+ll\b", re.IGNORECASE),
        re.compile(r"\bdon\s+t\b", re.IGNORECASE),
        re.compile(r"\bcan\s+t\b", re.IGNORECASE),
        re.compile(r"\bwon\s+t\b", re.IGNORECASE),
        re.compile(r"\bisn\s+t\b", re.IGNORECASE),
        re.compile(r"\bdidn\s+t\b", re.IGNORECASE),
        re.compile(r"\bain\s+t\b", re.IGNORECASE),
        re.compile(r"\bs\s+the\b", re.IGNORECASE),
        re.compile(r"\bs\s+an?\b", re.IGNORECASE),
        re.compile(r"\bs\s+my\b", re.IGNORECASE),
    ]
    stopwords = {"a", "an", "and", "at", "be", "but", "by", "for", "from",
                 "go", "had", "he", "her", "him", "his", "i", "in", "is",
                 "it", "me", "my", "no", "of", "on", "or", "our", "out",
                 "she", "so", "that", "the", "them", "then", "there", "they",
                 "this", "to", "was", "we", "you", "got", "like", "just",
                 "all", "say", "said", "with", "s", "m", "t", "ve", "ll",
                 "re", "d"}
    word_re = re.compile(r"[A-Za-z']+")
    garbled = 0
    for line in individual.lines:
        line_lower = line.lower()
        is_garbled = False
        for pat in _GARBLED_PATTERNS:
            if pat.search(line_lower):
                is_garbled = True
                break
        if not is_garbled:
            tokens = word_re.findall(line_lower)
            if tokens:
                content = [t for t in tokens if t not in stopwords and len(t) > 1]
                if len(content) < len(tokens) * 0.3:
                    is_garbled = True
        if is_garbled:
            garbled += 1
    return garbled / max(1, len(individual.lines))


def _score_cross_verse_repetition_penalty(
    individual: VerseIndividual,
    seen_lines: Optional[set] = None,
) -> float:
    """Penalty when lines appear in other archive entries."""
    if not seen_lines:
        return 0.0
    matches = 0
    for line in individual.lines:
        normalized = line.strip().lower()
        if normalized in seen_lines:
            matches += 1
    return min(1.0, matches / max(1, len(individual.lines)))


def score_verse(
    individual: VerseIndividual,
    scheme: str = "AABB",
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
    corpus_lines: Optional[List[str]] = None,
    seen_lines: Optional[set] = None,
    include_graph_metrics: bool = True,
    graph_edge_mode: str = "phonetic",
) -> Dict[str, float]:
    """Compute all component scores for a 4-line verse."""
    f = individual.features
    kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None
    scores: Dict[str, float] = {
        "rhyme_scheme_score": _score_verse_rhyme_scheme(f, scheme),
        "internal_rhyme": _score_verse_internal_rhyme_simple(f),
        "global_rhyme_chain_score": _score_global_rhyme_chain(individual.lines),
        "internal_chain_score": _score_internal_chain(individual.lines),
        "syllable_balance": _score_verse_syllable_balance(f),
        "fluency": _score_verse_fluency(individual),
        "lm_fluency": _score_verse_lm_fluency(individual),
        "semantic": _score_verse_semantic(
            individual,
            prompt_keywords=kw,
            theme_string=theme_string,
            semantic_scorer=semantic_scorer,
            embedding_weight=embedding_weight,
        ),
        "lexical_validity": _score_verse_lexical_validity(individual),
        "identical_line_penalty": _score_verse_identical_line_penalty(individual),
        "template_penalty": _score_verse_template_penalty(individual, prompt_keywords=kw),
        "repetition_penalty": _score_verse_repetition_penalty(individual),
        "near_duplicate_penalty": _score_verse_near_duplicate_penalty(individual),
        "filler_line_penalty": _score_verse_filler_line_penalty(individual),
        "line_phrase_penalty": _score_verse_line_phrase_penalty(individual),
        "corpus_overlap_penalty": _score_verse_corpus_overlap_penalty(individual, corpus_lines),
        "garbled_line_penalty": _score_verse_garbled_line_penalty(individual),
        "cliche_penalty": score_cliche_penalty(individual.lines),
        "rhyme_chain_density": score_rhyme_chain_density(individual.lines),
        "structural_repetition_penalty": _score_verse_structural_repetition_penalty(individual),
        "cross_verse_repetition_penalty": _score_cross_verse_repetition_penalty(individual, seen_lines),
        "flow_continuity_score": _score_flow_continuity(individual),
    }
    if include_graph_metrics:
        scores.update(score_rhyme_graph_metrics(individual.lines, edge_mode=graph_edge_mode))
    else:
        scores["rhyme_graph_density"] = 0.0
        scores["rhyme_graph_cluster_coeff"] = 0.0
        scores["rhyme_graph_chain_length"] = 0.0

    lines = individual.lines
    try:
        scores["coherence"] = score_coherence(lines)
    except Exception:
        scores["coherence"] = 0.0

    try:
        scores["punchline"] = score_punchline(lines)
    except Exception:
        scores["punchline"] = 0.0

    try:
        from evo_rhyme.flow import score_verse_flow
        scores["flow_alignment"] = score_verse_flow(lines, features=f)
    except Exception:
        scores["flow_alignment"] = 0.5

    # Adherence metrics depend on already-computed component scores.
    individual.scores = scores
    scores["style_adherence"] = _score_style_adherence(individual)
    scores["prompt_adherence"] = _score_prompt_adherence(individual)
    individual.scores = None

    return scores


def compute_verse_fitness(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate verse fitness from weighted sum of component scores."""
    w = weights or VERSE_DEFAULT_WEIGHTS
    total = 0.0
    for key, weight in w.items():
        if key in scores:
            if key == "cliche_penalty":
                cliche_val = scores.get("cliche_penalty", 0.0)
                cliche_scaled = min(1.0, cliche_val * 80)
                total += weight * cliche_scaled
            else:
                total += weight * scores[key]
    return total


# ---------------------------------------------------------------------------
# Multi-objective score vector
# ---------------------------------------------------------------------------

OBJECTIVE_KEYS: List[str] = [
    "end_rhyme",
    "internal_rhyme",
    "rhyme_chain_density",
    "global_rhyme_chain_score",
    "rhythm",
    "semantic",
    "fluency",
    "coherence",
    "originality",
    "style_match",
    "punchline",
    "novelty",
]


def score_vector(
    scores: Dict[str, float], keys: Optional[List[str]] = None
) -> List[float]:
    """Extract multi-objective score vector from a scores dict.

    Args:
        scores: The full scores dictionary from score_couplet or score_verse.
        keys: Which objectives to extract. Defaults to OBJECTIVE_KEYS.

    Returns:
        List of float values, one per objective. Missing keys default to 0.0.

    The 'rhythm' key is synthesized as mean of 'stress_alignment' and 'syllable_balance'.
    The 'originality' key is synthesized as 1.0 - scores.get('corpus_overlap_penalty', 0.0).
    The 'style_match' key maps to 'score_style_similarity' if present.
    """
    if keys is None:
        keys = OBJECTIVE_KEYS

    vec: List[float] = []
    for k in keys:
        if k == "rhythm":
            val = (
                scores.get("stress_alignment", 0.0)
                + scores.get("syllable_balance", 0.0)
            ) / 2.0
        elif k == "originality":
            val = 1.0 - abs(scores.get("corpus_overlap_penalty", 0.0))
        elif k == "style_match":
            val = scores.get("score_style_similarity", scores.get("style_adherence", 0.0))
        elif k == "end_rhyme":
            val = scores.get("end_rhyme", scores.get("rhyme_scheme_score", 0.0))
        elif k == "fluency":
            val = scores.get("ngram_fluency", scores.get("fluency", 0.0))
        else:
            val = scores.get(k, 0.0)
        vec.append(max(0.0, float(val)))
    return vec


def compute_fitness(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    individual: Optional[CoupletIndividual] = None,
    population: Optional[List[CoupletIndividual]] = None,
    style_profile: Optional["StyleProfile"] = None,
    style_weight: float = 0.0,
    ngram_floor: Optional[float] = NGRAM_FLOOR,
) -> float:
    """
    Aggregate fitness from weighted sum of component scores.

    Uses DEFAULT_WEIGHTS if weights not provided.
    When individual and population are provided, adds rhyme_family_diversity_penalty
    and repeated_shell_penalty.
    Caps result at FITNESS_CAP to prevent saturation.

    When ngram_floor is set (default 0.2), candidates with ngram_fluency below
    the floor get fitness 0 - they never appear in natural language.
    """
    if ngram_floor is not None and scores.get("ngram_fluency", 0.5) < ngram_floor:
        return 0.0
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


# ---------------------------------------------------------------------------
# Batched verse scoring
# ---------------------------------------------------------------------------

def score_verses_batch(
    individuals: List[VerseIndividual],
    scheme: str = "AABB",
    prompt_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    embedding_weight: float = 0.5,
    corpus_lines: Optional[List[str]] = None,
    seen_lines: Optional[set] = None,
    graph_top_k: int = 24,
    expensive_top_k: Optional[int] = None,
    fast_mode: bool = False,
    graph_edge_mode: str = "phonetic",
) -> List[Dict[str, float]]:
    """Score multiple verses with batched GPU calls for lm_fluency, punchline, coherence.

    Fast CPU-based metrics are computed per-verse. GPU-heavy metrics (lm_fluency,
    punchline, coherence) are batched across all verses for efficiency.
    """
    if not individuals:
        return []

    kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None
    all_scores: List[Dict[str, float]] = [{} for _ in individuals]
    missing_idxs: List[int] = []

    # Stage 1/2: cheap-medium features with cache.
    for idx, ind in enumerate(individuals):
        key = _norm_verse_key(ind.lines, scheme)
        cached = _VERSE_SCORE_CACHE.get(key)
        if cached is not None:
            all_scores[idx] = dict(cached)
            continue

        f = ind.features
        scores: Dict[str, float] = {
            "rhyme_scheme_score": _score_verse_rhyme_scheme(f, scheme),
            "internal_rhyme": _score_verse_internal_rhyme_simple(f),
            "global_rhyme_chain_score": _score_global_rhyme_chain(ind.lines),
            "internal_chain_score": _score_internal_chain(ind.lines),
            "syllable_balance": _score_verse_syllable_balance(f),
            "fluency": _score_verse_fluency(ind),
            "semantic": _score_verse_semantic(
                ind, prompt_keywords=kw, theme_string=theme_string,
                semantic_scorer=semantic_scorer, embedding_weight=embedding_weight,
            ),
            "lexical_validity": _score_verse_lexical_validity(ind),
            "identical_line_penalty": _score_verse_identical_line_penalty(ind),
            "template_penalty": _score_verse_template_penalty(ind, prompt_keywords=kw),
            "repetition_penalty": _score_verse_repetition_penalty(ind),
            "near_duplicate_penalty": _score_verse_near_duplicate_penalty(ind),
            "filler_line_penalty": _score_verse_filler_line_penalty(ind),
            "line_phrase_penalty": _score_verse_line_phrase_penalty(ind),
            "corpus_overlap_penalty": _score_verse_corpus_overlap_penalty(ind, corpus_lines),
            "garbled_line_penalty": _score_verse_garbled_line_penalty(ind),
            "cliche_penalty": score_cliche_penalty(ind.lines),
            "rhyme_chain_density": score_rhyme_chain_density(ind.lines),
            "cross_verse_repetition_penalty": _score_cross_verse_repetition_penalty(ind, seen_lines),
            "flow_continuity_score": _score_flow_continuity(ind),
            "rhyme_graph_density": 0.0,
            "rhyme_graph_cluster_coeff": 0.0,
            "rhyme_graph_chain_length": 0.0,
        }
        try:
            from evo_rhyme.flow import score_verse_flow
            scores["flow_alignment"] = score_verse_flow(ind.lines, features=f)
        except Exception:
            scores["flow_alignment"] = 0.5
        all_scores[idx] = scores
        missing_idxs.append(idx)

    if not missing_idxs:
        return all_scores

    # Stage 3a: graph metrics only for top-k by cheap score.
    if graph_top_k > 0:
        rank = sorted(
            missing_idxs,
            key=lambda i: (
                all_scores[i].get("rhyme_chain_density", 0.0)
                + all_scores[i].get("internal_rhyme", 0.0)
                + all_scores[i].get("global_rhyme_chain_score", 0.0)
                + all_scores[i].get("fluency", 0.0)
                - all_scores[i].get("garbled_line_penalty", 0.0)
            ),
            reverse=True,
        )[: min(graph_top_k, len(missing_idxs))]
        # Reserve some graph-budget for exploration so graph niches are not
        # dominated by only top-cheap-score candidates.
        remaining = [i for i in missing_idxs if i not in set(rank)]
        if remaining:
            explore_k = min(max(1, graph_top_k // 4), len(remaining))
            rank.extend(random.sample(remaining, k=explore_k))
        for i in rank:
            try:
                all_scores[i].update(
                    score_rhyme_graph_metrics(
                        individuals[i].lines,
                        edge_mode=graph_edge_mode,
                    )
                )
            except Exception:
                pass

    # Expensive stages can be gated in fast mode.
    expensive_candidates = list(missing_idxs)
    if fast_mode and expensive_top_k is not None:
        expensive_candidates = sorted(
            missing_idxs,
            key=lambda i: (
                all_scores[i].get("fluency", 0.0)
                + all_scores[i].get("semantic", 0.0)
                + all_scores[i].get("rhyme_scheme_score", 0.0)
            ),
            reverse=True,
        )[: max(0, min(expensive_top_k, len(missing_idxs)))]

    all_lines: List[str] = []
    verse_spans: Dict[int, Tuple[int, int]] = {}
    for idx in expensive_candidates:
        start = len(all_lines)
        all_lines.extend(individuals[idx].lines)
        verse_spans[idx] = (start, start + len(individuals[idx].lines))

    # LM fluency
    for i in missing_idxs:
        all_scores[i].setdefault("lm_fluency", 0.5)
    if all_lines:
        try:
            from evo_rhyme.lm_fluency import get_lm_scorer
            scorer = get_lm_scorer()
            if scorer is not None:
                line_fluencies = scorer.score_lines_batch(all_lines)
                for idx, (start, end) in verse_spans.items():
                    chunk = line_fluencies[start:end]
                    all_scores[idx]["lm_fluency"] = sum(chunk) / max(1, len(chunk))
        except Exception:
            pass

    # Punchline
    for i in missing_idxs:
        all_scores[i].setdefault("punchline", 0.0)
    if expensive_candidates:
        try:
            from evo_rhyme.scoring.punchline import score_punchlines_batch
            verses_lines = [individuals[i].lines for i in expensive_candidates]
            punch_scores = score_punchlines_batch(verses_lines)
            for idx, ps in zip(expensive_candidates, punch_scores):
                all_scores[idx]["punchline"] = ps
        except Exception:
            pass

    # Coherence
    for i in missing_idxs:
        all_scores[i].setdefault("coherence", 0.0)
    if all_lines:
        try:
            from evo_rhyme.scoring.coherence import score_coherence_with_embeddings
            from evo_rhyme.scoring.novelty import embed_texts as _embed
            line_embs = _embed(all_lines)
            for idx, (start, end) in verse_spans.items():
                emb_chunk = line_embs[start:end]
                all_scores[idx]["coherence"] = score_coherence_with_embeddings(emb_chunk)
        except Exception:
            for idx in expensive_candidates:
                try:
                    all_scores[idx]["coherence"] = score_coherence(individuals[idx].lines)
                except Exception:
                    pass

    # Adherence metrics and cache write.
    for idx in missing_idxs:
        individuals[idx].scores = all_scores[idx]
        all_scores[idx]["style_adherence"] = _score_style_adherence(individuals[idx])
        all_scores[idx]["prompt_adherence"] = _score_prompt_adherence(individuals[idx])
        individuals[idx].scores = None
        _cache_put(_norm_verse_key(individuals[idx].lines, scheme), all_scores[idx])

    return all_scores


# ---------------------------------------------------------------------------
# 16-bar verse scoring (block-level)
# ---------------------------------------------------------------------------

VERSE_16_WEIGHTS: Dict[str, float] = {
    "block_avg_fitness": 0.25,
    "block_coherence": 0.20,
    "theme_progression": 0.15,
    "punchline_placement": 0.10,
    "flow_consistency": 0.10,
    "structural_variety": 0.10,
    "block_rhyme_diversity": 0.10,
}


def _score_block_avg_fitness(block_fitnesses: List[float]) -> float:
    """Average fitness of constituent 4-bar blocks [0,1]."""
    if not block_fitnesses:
        return 0.0
    return sum(block_fitnesses) / len(block_fitnesses)


def _score_block_coherence(lines: List[str], block_size: int = 4) -> float:
    """Inter-block coherence: semantic similarity between consecutive blocks.

    Computes mean cosine similarity between the centroid embeddings of
    consecutive 4-bar blocks.
    """
    if len(lines) < block_size * 2:
        return 0.5
    try:
        from evo_rhyme.scoring.coherence import _get_model, _cosine_similarity
        import numpy as np
        model = _get_model()
        embeddings = model.encode(lines, convert_to_numpy=True)

        num_blocks = len(lines) // block_size
        block_centroids = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            centroid = embeddings[start:end].mean(axis=0)
            block_centroids.append(centroid)

        if len(block_centroids) < 2:
            return 0.5

        sims = []
        for i in range(len(block_centroids) - 1):
            sims.append(_cosine_similarity(block_centroids[i], block_centroids[i + 1]))

        raw = float(np.mean(sims))
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5


def _score_theme_progression(lines: List[str], block_size: int = 4) -> float:
    """Theme progression: semantic drift should increase from setup to punchline.

    Measures whether later blocks are more semantically distant from the first
    block, indicating thematic escalation.
    """
    if len(lines) < block_size * 2:
        return 0.5
    try:
        from evo_rhyme.scoring.coherence import _get_model, _cosine_similarity
        import numpy as np
        model = _get_model()
        embeddings = model.encode(lines, convert_to_numpy=True)

        num_blocks = len(lines) // block_size
        block_centroids = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            centroid = embeddings[start:end].mean(axis=0)
            block_centroids.append(centroid)

        if len(block_centroids) < 2:
            return 0.5

        dists = []
        for i in range(1, len(block_centroids)):
            dist = 1.0 - _cosine_similarity(block_centroids[0], block_centroids[i])
            dists.append(dist)

        if len(dists) < 2:
            return 0.5

        increasing_pairs = sum(
            1 for i in range(len(dists) - 1) if dists[i + 1] >= dists[i]
        )
        progression_ratio = increasing_pairs / max(1, len(dists) - 1)

        avg_dist = float(np.mean(dists))
        dist_quality = math.exp(-((avg_dist - 0.3) ** 2) / (2 * 0.15 ** 2))

        return 0.6 * progression_ratio + 0.4 * dist_quality
    except Exception:
        return 0.5


def _score_punchline_placement(lines: List[str], block_size: int = 4) -> float:
    """Score whether the strongest punchline content is in the final block."""
    try:
        from evo_rhyme.scoring.punchline import score_punchline_per_line
        per_line = score_punchline_per_line(lines)
        if not per_line or len(per_line) < block_size * 2:
            return 0.5

        num_blocks = len(per_line) // block_size
        block_avgs = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            chunk = per_line[start:end]
            block_avgs.append(sum(chunk) / max(1, len(chunk)))

        if not block_avgs:
            return 0.5

        max_block = max(range(len(block_avgs)), key=lambda i: block_avgs[i])
        if max_block == len(block_avgs) - 1:
            return 1.0
        elif max_block == len(block_avgs) - 2:
            return 0.6
        return 0.3
    except Exception:
        return 0.5


def _score_flow_consistency(lines: List[str]) -> float:
    """Flow consistency: low variance in stress pattern similarity across all lines."""
    try:
        from evo_rhyme.flow import score_flow
        scores = [score_flow(line) for line in lines]
        if len(scores) < 2:
            return 0.5
        import numpy as np
        mean_flow = float(np.mean(scores))
        std_flow = float(np.std(scores))
        consistency = max(0.0, 1.0 - std_flow * 3.0)
        return 0.5 * mean_flow + 0.5 * consistency
    except Exception:
        return 0.5


def _score_structural_variety_16(
    individual: VerseIndividual,
    block_size: int = 4,
) -> float:
    """Penalize when blocks share the same syntactic structure.

    Compares function-word subsequences across blocks. Different structures = higher score.
    """
    import re
    _FUNC_WORDS = frozenset({
        "i", "me", "my", "mine", "we", "our", "you", "your", "he", "she",
        "it", "they", "them", "his", "her", "its", "a", "an", "the",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "up", "out", "but", "and", "or", "so", "is", "am", "are",
        "was", "be", "been", "do", "don't", "can't", "won't", "ain't",
        "not", "no", "if", "that", "this", "like", "just", "got",
    })
    word_re = re.compile(r"[a-z']+")

    num_blocks = len(individual.lines) // block_size
    block_signatures: List[str] = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block_lines = individual.lines[start:end]
        func_tokens = []
        for line in block_lines:
            tokens = word_re.findall(line.lower())
            func_tokens.extend(t for t in tokens if t in _FUNC_WORDS)
        block_signatures.append(" ".join(func_tokens[:8]))

    if len(block_signatures) < 2:
        return 1.0

    unique = len(set(block_signatures))
    total = len(block_signatures)
    return unique / total


def _score_block_rhyme_diversity(
    individual: VerseIndividual,
    block_size: int = 4,
) -> float:
    """Reward diverse rhyme families across blocks."""
    try:
        from evo_rhyme.phonetics import extract_rhyme_tail, tokenize_line

        block_tails: List[set] = []
        num_blocks = len(individual.lines) // block_size

        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            tails = set()
            for line in individual.lines[start:end]:
                tokens = tokenize_line(line)
                if tokens:
                    tail = extract_rhyme_tail(tokens[-1])
                    if tail:
                        tails.add(str(tail))
            block_tails.append(tails)

        if len(block_tails) < 2:
            return 1.0

        all_tails = set()
        for tails in block_tails:
            all_tails.update(tails)

        max_possible = len(block_tails) * block_size
        diversity = len(all_tails) / max(1, max_possible)
        return min(1.0, diversity * 2.5)
    except Exception:
        return 0.5


def score_verse_16(
    individual: VerseIndividual,
    block_size: int = 4,
    block_fitnesses: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Score a 16-bar verse at the block level.

    Returns component scores for inter-block metrics. These are
    meant to complement (not replace) per-block scores from
    the existing 4-bar scoring pipeline.
    """
    lines = individual.lines
    scores: Dict[str, float] = {}

    scores["block_avg_fitness"] = _score_block_avg_fitness(
        block_fitnesses or []
    )
    scores["block_coherence"] = _score_block_coherence(lines, block_size)
    scores["theme_progression"] = _score_theme_progression(lines, block_size)
    scores["punchline_placement"] = _score_punchline_placement(lines, block_size)
    scores["flow_consistency"] = _score_flow_consistency(lines)
    scores["structural_variety"] = _score_structural_variety_16(individual, block_size)
    scores["block_rhyme_diversity"] = _score_block_rhyme_diversity(individual, block_size)

    return scores


def compute_verse_16_fitness(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate 16-bar verse fitness from weighted sum."""
    w = weights or VERSE_16_WEIGHTS
    total = 0.0
    for key, weight in w.items():
        if key in scores:
            total += weight * scores[key]
    return min(total, 0.95)
