"""
Line-level evolutionary loop for two-tier verse generation.

Evolves individual lines independently, scoring them on fluency, semantic
relevance, novelty, and rhyme chain potential. Populates a LineArchive
organized by rhyme group for verse assembly.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.line_archive import (
    LineArchive,
    ScoredLine,
    compute_line_composite,
)
from evo_rhyme.phonetics import (
    extract_rhyme_tail,
    phones_for_word,
    syllable_count_line,
    tokenize_line,
)

logger = logging.getLogger(__name__)


@dataclass
class LineEvolutionConfig:
    """Configuration for line-level evolution."""
    population_size: int = 1500
    num_generations: int = 3
    lm_seed_count: int = 80
    max_per_rhyme_group: int = 200
    mutation_rate: float = 0.8
    lm_mutation_budget: int = 15
    theme_keywords: List[str] = field(default_factory=list)
    min_syllables: int = 6
    max_syllables: int = 18
    use_embeddings: bool = True
    embedding_weight: float = 0.4


def score_line(
    text: str,
    theme_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    ngram_index: Optional[Dict[str, float]] = None,
    embedding_weight: float = 0.4,
) -> ScoredLine:
    """Score a single line on all dimensions and return a ScoredLine."""
    return score_lines_batch(
        [text], theme_keywords, theme_string, semantic_scorer,
        ngram_index, embedding_weight,
    )[0]


def score_lines_batch(
    texts: List[str],
    theme_keywords: Optional[Set[str]] = None,
    theme_string: Optional[str] = None,
    semantic_scorer: Optional[Any] = None,
    ngram_index: Optional[Dict[str, float]] = None,
    embedding_weight: float = 0.4,
) -> List[ScoredLine]:
    """Score multiple lines with batched embedding calls."""
    from evo_rhyme.scoring.rhyme_chain import score_line_chain_potential

    lines: List[ScoredLine] = []
    keyword_scores: List[float] = []

    for text in texts:
        line = ScoredLine.from_text(text)
        syl = line.syllables
        in_range = 1.0 if 6 <= syl <= 18 else max(0.0, 1.0 - abs(syl - 12) / 12.0)
        tokens = tokenize_line(text)
        content = [w for w in tokens if len(w) > 2]
        valid_ratio = sum(1 for w in content if phones_for_word(w)) / max(1, len(content)) if content else 1.0
        line.fluency = 0.6 * in_range + 0.4 * valid_ratio

        if theme_keywords:
            words = set(w.lower() for w in text.split() if len(w) > 2)
            kw_score = len(words & theme_keywords) / max(1, len(theme_keywords))
        else:
            kw_score = 0.5
        keyword_scores.append(kw_score)

        if ngram_index:
            from evo_rhyme.scoring.line_penalty import score_cliche_penalty
            cliche = score_cliche_penalty([text], ngram_index)
            line.novelty = max(0.0, 1.0 - cliche)
        else:
            line.novelty = 0.5

        line.chain_potential = score_line_chain_potential(text)
        lines.append(line)

    try:
        from evo_rhyme.lm_fluency import get_lm_scorer
        scorer = get_lm_scorer()
        if scorer is not None:
            lm_scores = scorer.score_lines_batch(texts)
            for line_obj, lm_s in zip(lines, lm_scores):
                line_obj.lm_fluency = lm_s
        else:
            for line_obj in lines:
                line_obj.lm_fluency = 0.5
    except Exception:
        for line_obj in lines:
            line_obj.lm_fluency = 0.5

    # Batch semantic scoring via score_pairs_batch
    if semantic_scorer and theme_string and texts:
        try:
            pairs = [(theme_string.strip(), t.lower()) for t in texts]
            cos_sims = semantic_scorer.score_pairs_batch(pairs)
            alpha = 1.0 - embedding_weight
            for line_obj, cos_sim, kw_score in zip(lines, cos_sims, keyword_scores):
                emb_score = (cos_sim + 1.0) / 2.0
                line_obj.semantic = alpha * kw_score + embedding_weight * emb_score
        except Exception:
            for line_obj, kw_score in zip(lines, keyword_scores):
                line_obj.semantic = kw_score
    else:
        for line_obj, kw_score in zip(lines, keyword_scores):
            line_obj.semantic = kw_score

    for line_obj in lines:
        line_obj.composite = compute_line_composite(line_obj)

    return lines


def seed_lines_from_lm(
    theme_keywords: List[str],
    count: int,
    min_syllables: int = 6,
    max_syllables: int = 18,
) -> List[str]:
    """Generate seed lines via LM using the existing BarProposer."""
    try:
        from evo_rhyme.lm_proposer import BarProposer, BarRequest, ProposerConfig

        proposer = BarProposer(ProposerConfig())
        lines: List[str] = []

        roles = ["setup", "flex", "punchline"]
        batches_per_role = max(1, (count + 9) // (10 * len(roles)))
        for role in roles:
            for _ in range(batches_per_role):
                if len(lines) >= count:
                    break
                try:
                    request = BarRequest(
                        theme_keywords=list(theme_keywords) if theme_keywords else ["hip-hop", "life"],
                        syllable_range=(min_syllables, max_syllables),
                        role=role,
                    )
                    batch = proposer.propose_bars(request)
                    for bar in batch:
                        bar = bar.strip()
                        if bar and min_syllables <= syllable_count_line(bar) <= max_syllables:
                            lines.append(bar)
                except Exception:
                    logger.warning("LM seed batch failed", exc_info=True)

        return lines[:count]
    except Exception:
        logger.warning("seed_lines_from_lm failed entirely", exc_info=True)
        return []


def _mutate_line_lm(
    line: ScoredLine,
    config: LineEvolutionConfig,
    lm_budget: Dict[str, int],
) -> Optional[str]:
    """Apply an LM mutation to a single line (sync fallback)."""
    if lm_budget.get("remaining", 0) <= 0:
        return None
    try:
        from evo_rhyme.mutation import get_rewriter
        rewriter = get_rewriter()
        theme_keywords = config.theme_keywords
        syl_range = (config.min_syllables, config.max_syllables)
        rhyme_target = line.end_word

        r = random.random()
        if r < 0.25:
            candidates = rewriter.structural_rewrite(line.text, rhyme_target, theme_keywords, syl_range)
        elif r < 0.45:
            candidates = rewriter.metaphor_inject(line.text, rhyme_target, theme_keywords, syl_range)
        elif r < 0.60:
            candidates = rewriter.theme_rewrite(line.text, theme_keywords, syl_range)
        elif r < 0.75:
            candidates = rewriter.rhyme_rewrite(line.text, rhyme_target, theme_keywords, syl_range)
        elif r < 0.85:
            candidates = rewriter.paraphrase(line.text, preserve_end_rhyme=True, syllable_range=syl_range)
        else:
            candidates = rewriter.contrast_swap(line.text, rhyme_target, theme_keywords, syl_range)

        lm_budget["remaining"] = lm_budget.get("remaining", 0) - 1

        if candidates:
            return candidates[0]
        return None
    except Exception:
        logger.warning("_mutate_line_lm failed", exc_info=True)
        return None


def _build_lm_mutation_request(
    line: ScoredLine,
    config: LineEvolutionConfig,
) -> Optional[Tuple[str, str, str]]:
    """Build a prompt/cache_key tuple for async batch rewriting. Returns None if no request needed."""
    from evo_rhyme.lm_rewriter import _cache_key
    theme_keywords = config.theme_keywords
    theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"
    syl_range = (config.min_syllables, config.max_syllables)
    rhyme_target = line.end_word
    n = 5

    r = random.random()
    if r < 0.25:
        method = "structural_rewrite"
        prompt = (
            f'Rewrite this rap bar with a different sentence structure.\n'
            f'The last word MUST rhyme with "{rhyme_target}".\n'
            f'Stay between {syl_range[0]}-{syl_range[1]} syllables.\nTheme: {theme}\n\n'
            f'Original: "{line.text}"\n\nWrite {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
    elif r < 0.45:
        method = "metaphor_inject"
        prompt = (
            f'Rewrite this rap bar using a vivid, original metaphor.\n'
            f'The last word MUST rhyme with "{rhyme_target}".\n'
            f'Stay between {syl_range[0]}-{syl_range[1]} syllables.\nTheme: {theme}\n\n'
            f'Original: "{line.text}"\n\nWrite {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
    elif r < 0.60:
        method = "theme_rewrite"
        prompt = (
            f'Rewrite this rap bar to better fit the theme: {theme}.\n'
            f'Keep the flow and rhyme scheme. Stay between {syl_range[0]}-{syl_range[1]} syllables.\n\n'
            f'Original: "{line.text}"\n\nWrite {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
    elif r < 0.75:
        method = "rhyme_rewrite"
        prompt = (
            f'Rewrite this rap bar. Keep the meaning and flow.\n'
            f'The last word MUST rhyme with "{rhyme_target}".\n'
            f'Stay between {syl_range[0]}-{syl_range[1]} syllables.\nTheme: {theme}\n\n'
            f'Original: "{line.text}"\n\nWrite {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
    elif r < 0.85:
        method = "paraphrase"
        prompt = (
            f'Paraphrase this rap bar. Preserve the end rhyme word.\n'
            f'Stay between {syl_range[0]}-{syl_range[1]} syllables.\n\n'
            f'Original: "{line.text}"\n\nWrite {n} paraphrases, one per line. Output ONLY the rewritten bars.'
        )
    else:
        method = "contrast_swap"
        prompt = (
            f'Rewrite this rap bar with an opposing emotional tone.\n'
            f'The last word MUST rhyme with "{rhyme_target}".\n'
            f'Stay between {syl_range[0]}-{syl_range[1]} syllables.\nTheme: {theme}\n\n'
            f'Original: "{line.text}"\n\nWrite {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )

    key = _cache_key(line.text, method, (rhyme_target, tuple(theme_keywords), syl_range))
    return (prompt, key, line.text)


def _mutate_line_legacy(line: ScoredLine, config: LineEvolutionConfig) -> Optional[str]:
    """Apply a legacy word-level mutation to a single line."""
    try:
        from evo_rhyme.mutation import get_tail_to_words
        tail_to_words = get_tail_to_words()
        tokens = tokenize_line(line.text)
        if len(tokens) < 3:
            return None

        corpus_vocab = None
        mutation_config: Dict[str, Any] = {"corpus_vocab": corpus_vocab}

        r = random.random()
        if r < 0.3:
            # End word swap
            tail = line.end_tail
            if tail and tail in tail_to_words:
                alts = [w for w in tail_to_words[tail] if w != line.end_word]
                if alts:
                    new_word = random.choice(alts)
                    new_tokens = tokens[:-1] + [new_word]
                    return " ".join(new_tokens)
        elif r < 0.6:
            # Stressed vowel swap at random internal position
            from evo_rhyme.phonetics import extract_stressed_vowels_from_phones, get_vowel_to_words
            mid_indices = list(range(1, len(tokens) - 1))
            if mid_indices:
                idx = random.choice(mid_indices)
                word = tokens[idx].lower()
                phones_list = phones_for_word(word)
                if phones_list:
                    vowels = extract_stressed_vowels_from_phones(phones_list[0])
                    if vowels:
                        v2w = get_vowel_to_words()
                        vowel = vowels[0]
                        if vowel in v2w:
                            alts = [w for w in v2w[vowel] if w != word]
                            if alts:
                                new_word = random.choice(alts[:20])
                                new_tokens = tokens[:idx] + [new_word] + tokens[idx + 1:]
                                return " ".join(new_tokens)
        else:
            # Chain extension: find a word from other rhyme family at internal position
            mid_indices = list(range(1, len(tokens) - 1))
            if mid_indices:
                idx = random.choice(mid_indices)
                end_tail = line.end_tail
                if end_tail and end_tail in tail_to_words:
                    alts = [w for w in tail_to_words[end_tail] if w != tokens[idx].lower()]
                    if alts:
                        new_word = random.choice(alts[:10])
                        new_tokens = tokens[:idx] + [new_word] + tokens[idx + 1:]
                        return " ".join(new_tokens)
        return None
    except Exception:
        logger.warning("_mutate_line_legacy failed", exc_info=True)
        return None


def evolve_lines(
    config: LineEvolutionConfig,
    initial_lines: Optional[List[ScoredLine]] = None,
    semantic_scorer: Optional[Any] = None,
    novelty_archive: Optional[Any] = None,
    existing_archive: Optional[LineArchive] = None,
) -> LineArchive:
    """Run line-level evolution and return a populated LineArchive.

    If existing_archive is provided, reuses it (incremental evolution).
    """
    kw_set = set(w.lower() for w in config.theme_keywords) if config.theme_keywords else None
    theme_string = " ".join(config.theme_keywords) if config.theme_keywords else None
    ngram_index = None
    try:
        from evo_rhyme.scoring.line_penalty import get_ngram_index
        ngram_index = get_ngram_index()
    except Exception:
        pass

    archive = existing_archive if existing_archive is not None else LineArchive(max_per_group=config.max_per_rhyme_group)

    # 1. Seed population
    population: List[ScoredLine] = []
    if initial_lines:
        population.extend(initial_lines)

    if len(population) < config.population_size and config.lm_seed_count > 0:
        needed = config.population_size - len(population)
        seed_count = min(needed, config.lm_seed_count)
        raw_lines = seed_lines_from_lm(
            config.theme_keywords, seed_count,
            config.min_syllables, config.max_syllables,
        )
        if raw_lines:
            scored = score_lines_batch(
                raw_lines, kw_set, theme_string, semantic_scorer,
                ngram_index, config.embedding_weight,
            )
            population.extend(scored)

    # 2. Score initial population
    for sl in population:
        if sl.composite == 0.0:
            sl.composite = compute_line_composite(sl)

    # Add all to archive
    archive.add_batch(population)

    logger.info(
        "Line evolution: initial pop=%d archive=%d groups=%d",
        len(population), archive.size(), archive.group_count(),
    )

    # 3. Evolve
    for gen in range(config.num_generations):
        lm_remaining = config.lm_mutation_budget
        offspring: List[ScoredLine] = []
        pending_texts: List[str] = []

        # Phase 1: Select parents and split into LM vs legacy mutations
        lm_parents: List[ScoredLine] = []
        legacy_parents: List[ScoredLine] = []

        for _ in range(config.population_size):
            candidates = random.choices(population, k=min(4, len(population)))
            parent = max(candidates, key=lambda l: l.composite)
            if random.random() > config.mutation_rate:
                continue
            if random.random() < 0.7 and lm_remaining > 0:
                lm_parents.append(parent)
                lm_remaining -= 1
            else:
                legacy_parents.append(parent)

        # Phase 2: Batch all LM mutations concurrently
        if lm_parents:
            from evo_rhyme.mutation import get_rewriter
            import re as _re
            rewriter = get_rewriter()
            _NUMBERING = _re.compile(r"^\s*(?:\d+[\.\)\-]|\-|\*)\s*")

            requests = []
            for parent in lm_parents:
                req = _build_lm_mutation_request(parent, config)
                if req:
                    requests.append(req)

            if requests:
                raw_responses = rewriter.batch_rewrite(requests, concurrency=20)
                for raw_resp in raw_responses:
                    if not raw_resp:
                        continue
                    for cand_line in raw_resp.strip().splitlines():
                        cand = _NUMBERING.sub("", cand_line).strip().strip('"').strip("'").strip()
                        if not cand:
                            continue
                        syl = syllable_count_line(cand)
                        if config.min_syllables <= syl <= config.max_syllables:
                            pending_texts.append(cand)
                            break

        # Phase 3: Legacy mutations (instant, no API calls)
        for parent in legacy_parents:
            new_text = _mutate_line_legacy(parent, config)
            if new_text is None:
                continue
            syl = syllable_count_line(new_text)
            if not (config.min_syllables <= syl <= config.max_syllables):
                continue
            pending_texts.append(new_text)

        # Phase 3b: Batch score all new texts at once
        if pending_texts:
            offspring = score_lines_batch(
                pending_texts, kw_set, theme_string, semantic_scorer,
                ngram_index, config.embedding_weight,
            )

        # Phase 4: Batch novelty scoring
        if novelty_archive is not None and offspring:
            try:
                from evo_rhyme.scoring.novelty import embed_texts
                texts = [c.text for c in offspring]
                embeddings = embed_texts(texts)
                novelties = novelty_archive.compute_novelty_batch(embeddings)
                for child, nov in zip(offspring, novelties):
                    child.novelty = 0.5 * child.novelty + 0.5 * float(nov)
                    child.composite = compute_line_composite(child)
            except Exception:
                pass

        # Update archive and population
        added = archive.add_batch(offspring)

        all_lines = list(population) + offspring
        all_lines.sort(key=lambda l: l.composite, reverse=True)
        population = all_lines[: config.population_size]

        logger.info(
            "Line gen %d: offspring=%d added_to_archive=%d archive_size=%d groups=%d",
            gen, len(offspring), added, archive.size(), archive.group_count(),
        )

    return archive
