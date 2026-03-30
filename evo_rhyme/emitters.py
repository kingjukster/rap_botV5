"""
evo_rhyme/emitters.py

Emitter-based MAP-Elites for quality-diversity rap verse evolution.
Multiple specialized emitters generate candidates through different strategies,
coordinated by an adaptive scheduler that allocates budget based on success.
"""

from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from evo_rhyme.individual import VerseIndividual, analyze_verse_individual
from evo_rhyme.style_genome import (
    StyleGenome,
    random_style_genome,
    mutate_style_genome,
    crossover_style_genome,
    style_to_prompt_directives,
)
from evo_rhyme.prompt_genome import (
    PromptGenome,
    random_prompt_genome,
    mutate_prompt_genome,
    crossover_prompt_genome,
    prompt_directives,
)

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ["AABB", "ABAB", "ABBA", "ABCB", "AABA", "AAAA"]


def _constraint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build constraint config with optional prompt_keywords for orphan check (Plan 2)."""
    cc = {
        "min_syllables": config.get("min_syllables", 6),
        "max_syllables": config.get("max_syllables", 18),
    }
    theme_kw = config.get("theme_keywords", [])
    if theme_kw:
        cc["prompt_keywords"] = list(theme_kw)
    return cc


def _mutation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build standard mutation config from emitter config, including corpus_vocab."""
    mc: Dict[str, Any] = {
        "theme_keywords": config.get("theme_keywords", []),
        "min_syllables": config.get("min_syllables", 6),
        "max_syllables": config.get("max_syllables", 18),
        "use_structural_mutations": config.get("use_structural_mutations", False),
        "embedding_neighbor_k": config.get("embedding_neighbor_k", 12),
        "embedding_min_cosine": config.get("embedding_min_cosine", 0.58),
    }
    if config.get("corpus_vocab"):
        mc["corpus_vocab"] = config["corpus_vocab"]
    return mc


@dataclass
class EmitResult:
    """Result of an emitter batch."""
    candidates: List[VerseIndividual]
    new_niches: int = 0
    improved_niches: int = 0
    total_inserted: int = 0
    elapsed_ms: float = 0.0
    generated: int = 0


class BaseEmitter(ABC):
    """Abstract base for MAP-Elites emitters."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._total_new_niches = 0
        self._total_improved = 0

    @abstractmethod
    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        """Generate a batch of candidate verse individuals."""
        ...

    def update(self, result: EmitResult) -> None:
        """Update internal state after batch evaluation."""
        self._total_new_niches += result.new_niches
        self._total_improved += result.improved_niches


def _genome_from_metadata(ind: VerseIndividual) -> Tuple[StyleGenome, PromptGenome]:
    md = ind.metadata or {}
    style_raw = md.get("style_genome")
    prompt_raw = md.get("prompt_genome")
    try:
        style = StyleGenome.from_dict(style_raw) if isinstance(style_raw, dict) else random_style_genome()
    except Exception:
        style = random_style_genome()
    try:
        prompt = PromptGenome.from_dict(prompt_raw) if isinstance(prompt_raw, dict) else random_prompt_genome()
    except Exception:
        prompt = random_prompt_genome()
    return style, prompt


def _attach_genomes(ind: VerseIndividual, style: Optional[StyleGenome] = None, prompt: Optional[PromptGenome] = None) -> None:
    style = style or random_style_genome()
    prompt = prompt or random_prompt_genome()
    ind.metadata.setdefault("style_genome", style.to_dict())
    ind.metadata.setdefault("style_genome_labels", style.to_labels())
    ind.metadata.setdefault("prompt_genome", prompt.to_dict())
    ind.metadata.setdefault("prompt_directives", {
        **style_to_prompt_directives(style),
        **{
            "strictness": f"{prompt.strictness:.2f}",
            "novelty_bias": f"{prompt.novelty_bias:.2f}",
            "metaphor_boost": f"{prompt.metaphor_boost:.2f}",
            "internal_rhyme_boost": f"{prompt.internal_rhyme_boost:.2f}",
            "punchline_bias": f"{prompt.punchline_bias:.2f}",
        },
    })


class RandomEmitter(BaseEmitter):
    """Pure exploration: random templates, random schemes, random vocab."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("random", config)
        self._corpus_path = config.get("corpus_path")
        self._theme_keywords = config.get("theme_keywords", [])

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.population import VerseSeedGenerator
        candidates = []
        schemes_to_try = self.config.get("schemes", ALLOWED_SCHEMES)

        prompt_llm_fraction = float(self.config.get("prompt_llm_fraction", 0.0))
        for _ in range(batch_size):
            style = random_style_genome()
            prompt = random_prompt_genome()
            style_labels = style.to_labels()
            scheme = style_labels.get("rhyme_scheme", random.choice(schemes_to_try))
            if scheme == "freeform":
                scheme = random.choice(schemes_to_try)

            use_llm_prompt = prompt_llm_fraction > 0 and random.random() < prompt_llm_fraction
            try:
                if use_llm_prompt:
                    # Optional prompt-space generation path (runtime-gated by fraction).
                    gen = VerseSeedGenerator(
                        corpus_path=self._corpus_path,
                        init_mode="lm",
                        scheme=scheme,
                        proposer_config=self.config.get("proposer_config"),
                    )
                    theme = list(self._theme_keywords)
                    theme.append(style_labels.get("imagery_mode", "street"))
                    style_ctrl = {
                        **style_to_prompt_directives(style),
                        **prompt_directives(prompt),
                    }
                    verses = gen.generate_seed_verses(
                        theme_keywords=theme,
                        size=1,
                        style_directives=style_ctrl,
                    )
                else:
                    gen = VerseSeedGenerator(
                        corpus_path=self._corpus_path,
                        init_mode="mixed",
                        scheme=scheme,
                    )
                    verses = gen.generate_seed_verses(
                        theme_keywords=self._theme_keywords,
                        size=1,
                    )
                if verses:
                    v = verses[0]
                    v.metadata["origin"] = "random_emitter"
                    v.metadata["scheme"] = scheme
                    _attach_genomes(v, style=style, prompt=prompt)
                    candidates.append(v)
            except Exception:
                logger.debug("RandomEmitter: generation failed", exc_info=True)

        return candidates


class MutationEmitter(BaseEmitter):
    """Mutate archive elites at multiple scales: small/medium/large/huge."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("mutation", config)

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.verse_evolution import (
            select_crossover_parents,
            verse_crossover,
            verse_mutate,
        )
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.mutation import MUTATION_WEIGHTS

        parents = archive.sample_parents(batch_size * 2)
        if not parents:
            return []

        mutation_config = _mutation_config(self.config)
        constraint_config = _constraint_config(self.config)
        crossover_rate = self.config.get("crossover_rate", 0.5)
        semantic_pairing = self.config.get("semantic_crossover_pairing", False)
        lm_budget_per_gen = self.config.get("lm_mutation_budget_per_gen", 0)
        lm_budget = {"remaining": lm_budget_per_gen} if lm_budget_per_gen > 0 else {"remaining": 0}

        candidates = []
        attempts = 0
        max_attempts = batch_size * 3

        while len(candidates) < batch_size and attempts < max_attempts:
            attempts += 1
            p1, p2 = select_crossover_parents(
                parents, semantic_pairing=semantic_pairing,
            )
            s1, pr1 = _genome_from_metadata(p1)
            s2, pr2 = _genome_from_metadata(p2)

            if random.random() < crossover_rate and len(parents) >= 2:
                child = verse_crossover(p1, p2, {})
                style = crossover_style_genome(s1, s2)
                prompt = crossover_prompt_genome(pr1, pr2)
            else:
                child = VerseIndividual(lines=list(p1.lines))
                style = s1
                prompt = pr1

            r = random.random()
            if r < 0.10 and lm_budget.get("remaining", 0) > 0:
                from evo_rhyme.verse_evolution import _lm_verse_rewrite
                rewritten = _lm_verse_rewrite(child, mutation_config)
                if rewritten:
                    child = rewritten
                else:
                    child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS,
                                         constraint_config=constraint_config, lm_budget=lm_budget)
            elif r < 0.25:
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS,
                                     constraint_config=constraint_config, lm_budget=lm_budget)
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS,
                                     constraint_config=constraint_config, lm_budget=lm_budget)
            elif r < 0.50:
                heavy_weights = dict(MUTATION_WEIGHTS)
                for k in heavy_weights:
                    if k in ("line_replace", "block_replace"):
                        heavy_weights[k] *= 2.0
                child = verse_mutate(child, mutation_config, heavy_weights,
                                     constraint_config=constraint_config, lm_budget=lm_budget)
            else:
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS,
                                     constraint_config=constraint_config, lm_budget=lm_budget)

            if passes_verse_constraints(child, constraint_config):
                analyze_verse_individual(child)
                child.metadata["origin"] = "mutation_emitter"
                p1_fit = p1.fitness or 0.0
                p2_fit = p2.fitness or 0.0
                child.metadata["_parent_fitness"] = max(p1_fit, p2_fit)
                style = mutate_style_genome(style, mutation_rate=0.20)
                prompt = mutate_prompt_genome(prompt, mutation_scale=0.12)
                _attach_genomes(child, style=style, prompt=prompt)
                candidates.append(child)

        return candidates


class DirectedMutationEmitter(BaseEmitter):
    """Base class for target-directed mutation emitters."""

    def __init__(self, name: str, config: Dict[str, Any], focus: str):
        super().__init__(name, config)
        self.focus = focus

    def _focused_weights(self, base: Dict[str, float]) -> Dict[str, float]:
        out = dict(base)
        if self.focus == "internal_rhyme":
            for k in ("rhyme_graph_expand", "chain_extension", "embedding_rhyme_walk", "multisyllable_rhyme", "lm_internal_rhyme"):
                out[k] = out.get(k, 0.0) * 2.5
        elif self.focus == "narrative":
            for k in ("lm_theme_rewrite", "lm_paraphrase", "line_replace", "lm_structural_rewrite"):
                out[k] = out.get(k, 0.0) * 2.2
        elif self.focus == "punchline":
            for k in ("lm_contrast_swap", "lm_score_guided", "lm_metaphor_inject", "line_replace"):
                out[k] = out.get(k, 0.0) * 2.3
        elif self.focus == "flow":
            for k in ("syllable_adjust", "lm_tighten", "lm_expand", "lm_structural_rewrite"):
                out[k] = out.get(k, 0.0) * 2.2
        elif self.focus == "imagery":
            for k in ("lm_metaphor_inject", "lm_theme_rewrite", "lm_paraphrase"):
                out[k] = out.get(k, 0.0) * 3.0
        return out

    def _style_bias(self, style: StyleGenome) -> StyleGenome:
        d = style.to_dict()
        if self.focus == "internal_rhyme":
            d["internal_rhyme_density"] = min(3, d["internal_rhyme_density"] + 1)
            d["multisyllabic_rhyme_density"] = min(3, d["multisyllabic_rhyme_density"] + 1)
        elif self.focus == "narrative":
            d["narrativity"] = 2
        elif self.focus == "punchline":
            d["tone"] = random.choice([2, 3, 4])
        elif self.focus == "flow":
            d["syllable_density"] = random.choice([1, 2, 3])
        elif self.focus == "imagery":
            d["metaphor_density"] = min(3, d["metaphor_density"] + 1)
            d["imagery_mode"] = random.choice([2, 3, 5])  # mythic/industrial/surreal
        return StyleGenome.from_dict(d)

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.verse_evolution import (
            select_crossover_parents,
            verse_crossover,
            verse_mutate,
        )
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.mutation import MUTATION_WEIGHTS

        parents = archive.sample_parents(batch_size * 2)
        if not parents:
            return []

        mutation_config = _mutation_config(self.config)
        constraint_config = _constraint_config(self.config)
        crossover_rate = self.config.get("crossover_rate", 0.5)
        semantic_pairing = self.config.get("semantic_crossover_pairing", False)
        weights = self._focused_weights(MUTATION_WEIGHTS)
        lm_budget_per_gen = self.config.get("lm_mutation_budget_per_gen", 0)
        lm_budget = {"remaining": lm_budget_per_gen} if lm_budget_per_gen > 0 else {"remaining": 0}

        candidates: List[VerseIndividual] = []
        attempts = 0
        max_attempts = batch_size * 4
        while len(candidates) < batch_size and attempts < max_attempts:
            attempts += 1
            p1, p2 = select_crossover_parents(
                parents, semantic_pairing=semantic_pairing,
            )
            s1, pr1 = _genome_from_metadata(p1)
            s2, pr2 = _genome_from_metadata(p2)

            if random.random() < crossover_rate and len(parents) >= 2:
                child = verse_crossover(p1, p2, {})
                style = crossover_style_genome(s1, s2)
                prompt = crossover_prompt_genome(pr1, pr2)
            else:
                child = VerseIndividual(lines=list(p1.lines))
                style = s1
                prompt = pr1

            child = verse_mutate(child, mutation_config, weights,
                                 constraint_config=constraint_config,
                                 lm_budget=lm_budget)
            if random.random() < 0.25:
                child = verse_mutate(child, mutation_config, weights,
                                     constraint_config=constraint_config,
                                     lm_budget=lm_budget)

            if passes_verse_constraints(child, constraint_config):
                analyze_verse_individual(child)
                child.metadata["origin"] = f"{self.name}_emitter"
                style = mutate_style_genome(self._style_bias(style), mutation_rate=0.15)
                prompt = mutate_prompt_genome(prompt, mutation_scale=0.10)
                _attach_genomes(child, style=style, prompt=prompt)
                candidates.append(child)
        return candidates


class InternalRhymeEmitter(DirectedMutationEmitter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("internal_rhyme", config, focus="internal_rhyme")


class NarrativeEmitter(DirectedMutationEmitter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("narrative", config, focus="narrative")


class PunchlineEmitter(DirectedMutationEmitter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("punchline", config, focus="punchline")


class FlowEmitter(DirectedMutationEmitter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("flow", config, focus="flow")


class ImageryEmitter(DirectedMutationEmitter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("imagery", config, focus="imagery")


class NicheTargetingEmitter(BaseEmitter):
    """Deliberately target empty niches by mutating nearest occupied elite."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("niche_targeting", config)
        self._target_queue: List[Tuple[int, ...]] = []

    def _refresh_targets(self, archive: Any) -> None:
        """Refresh the queue of empty niches to target."""
        # Sampling avoids O(total_niches) scans when archive is high-dimensional.
        sample = self.config.get("niche_targeting_sample", 600)
        queue_size = self.config.get("niche_targeting_queue", 400)
        empty = archive.sample_empty_niches(sample)
        random.shuffle(empty)
        self._target_queue = empty[:queue_size]

    def _get_niche_mutation_hints(
        self, target_labels: Dict[str, str], archive: Any,
    ) -> Dict[str, Any]:
        """Convert niche labels into mutation guidance."""
        hints: Dict[str, Any] = {}
        theme_kw = list(self.config.get("theme_keywords", []))

        intensity = target_labels.get("intensity", "")
        if intensity in ("aggressive", "very_aggressive"):
            hints["force_aggressive"] = True
            hints["extra_keywords"] = [
                "kill", "slay", "venom", "strike", "crush", "burn", "fire",
                "flex", "king", "reign", "throne", "conquer", "savage",
            ]

        rhyme_d = target_labels.get("rhyme_density", "")
        if rhyme_d in ("high", "very_high"):
            hints["force_internal_rhyme"] = True
            hints["prefer_ops"] = ["rhyme_graph_expand", "chain_extension", "embedding_rhyme_walk"]

        chain = target_labels.get("chain_length", "")
        if chain in ("long", "very_long"):
            hints["prefer_ops"] = hints.get("prefer_ops", []) + ["chain_extension", "multisyllable_rhyme"]
            if chain == "very_long":
                hints["prefer_ops_heavy"] = ["chain_extension", "multisyllable_rhyme"]

        graph_d = target_labels.get("graph_density", "")
        if graph_d in ("dense", "very_dense"):
            hints["prefer_ops"] = hints.get("prefer_ops", []) + ["rhyme_graph_expand", "embedding_rhyme_walk"]

        syl = target_labels.get("syllable_tightness", "")
        if syl == "dense":
            hints["target_syllables"] = (12, 18)
        elif syl == "sparse":
            hints["target_syllables"] = (6, 9)

        sentiment = target_labels.get("sentiment_polarity", "")
        if sentiment == "dark":
            hints["extra_keywords"] = hints.get("extra_keywords", []) + [
                "pain", "shadow", "bleed", "drown", "cold", "grave",
            ]
        elif sentiment == "hopeful":
            hints["extra_keywords"] = hints.get("extra_keywords", []) + [
                "rise", "light", "dream", "shine", "free", "glory",
            ]

        metaphor = target_labels.get("metaphor_density", "")
        if metaphor == "rich":
            hints["force_metaphor"] = True
            hints["extra_keywords"] = hints.get("extra_keywords", []) + [
                "like", "as", "flow", "river", "fire", "storm", "shadow", "light",
            ]

        return hints

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.verse_evolution import verse_mutate
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.mutation import MUTATION_WEIGHTS

        if not self._target_queue or generation % 3 == 0:
            self._refresh_targets(archive)

        if not self._target_queue:
            logger.info("NicheTargetingEmitter: no empty niches left")
            return []

        mutation_config = _mutation_config(self.config)
        mutation_config["theme_keywords"] = list(mutation_config.get("theme_keywords", []))
        constraint_config = _constraint_config(self.config)

        candidates = []
        targets_tried = 0

        while len(candidates) < batch_size and self._target_queue and targets_tried < batch_size * 3:
            targets_tried += 1
            target_coord = self._target_queue.pop(0)
            target_labels = archive.niche_label(target_coord)

            parent = archive.nearest_occupied(target_coord)
            if parent is None:
                continue
            parent_style, parent_prompt = _genome_from_metadata(parent)

            hints = self._get_niche_mutation_hints(target_labels, archive)

            mut_cfg = dict(mutation_config)
            if hints.get("extra_keywords"):
                mut_cfg["theme_keywords"] = mut_cfg["theme_keywords"] + hints["extra_keywords"]
            if hints.get("target_syllables"):
                mut_cfg["min_syllables"] = hints["target_syllables"][0]
                mut_cfg["max_syllables"] = hints["target_syllables"][1]
                constraint_config_local = {
                    "min_syllables": hints["target_syllables"][0],
                    "max_syllables": hints["target_syllables"][1],
                }
                theme_kw = self.config.get("theme_keywords", [])
                if theme_kw:
                    constraint_config_local["prompt_keywords"] = list(theme_kw)
            else:
                constraint_config_local = constraint_config

            lm_budget_per_gen = self.config.get("lm_mutation_budget_per_gen", 0)
            niche_lm_budget = {"remaining": lm_budget_per_gen} if lm_budget_per_gen > 0 else {"remaining": 0}

            heavy_weights = dict(MUTATION_WEIGHTS)
            for k in heavy_weights:
                if k in ("line_replace", "block_replace"):
                    heavy_weights[k] *= 3.0
            if lm_budget_per_gen > 0:
                for k in heavy_weights:
                    if k.startswith("lm_"):
                        heavy_weights[k] *= 2.0
            for k in hints.get("prefer_ops", []):
                if k in heavy_weights:
                    heavy_weights[k] *= 2.0
            for k in hints.get("prefer_ops_heavy", []):
                if k in heavy_weights:
                    heavy_weights[k] *= 3.0
            if hints.get("force_metaphor") and lm_budget_per_gen > 0:
                if "lm_metaphor_inject" in heavy_weights:
                    heavy_weights["lm_metaphor_inject"] *= 3.0

            for attempt in range(3):
                child = VerseIndividual(lines=list(parent.lines), metadata={"origin": "niche_targeting"})
                child = verse_mutate(child, mut_cfg, heavy_weights,
                                     constraint_config=constraint_config_local,
                                     lm_budget=niche_lm_budget)

                if hints.get("force_aggressive"):
                    aggressive_words = hints.get("extra_keywords", [])
                    if aggressive_words:
                        line_idx = random.randint(0, len(child.lines) - 1)
                        tokens = child.lines[line_idx].split()
                        if len(tokens) >= 3:
                            inject_pos = random.randint(1, len(tokens) - 2)
                            tokens[inject_pos] = random.choice(aggressive_words)
                            child.lines[line_idx] = " ".join(tokens)

                if passes_verse_constraints(child, constraint_config_local):
                    analyze_verse_individual(child)
                    child.metadata["target_niche"] = list(target_coord)
                    child.metadata["target_labels"] = target_labels
                    style = mutate_style_genome(parent_style, mutation_rate=0.25)
                    prompt = mutate_prompt_genome(parent_prompt, mutation_scale=0.14)
                    _attach_genomes(child, style=style, prompt=prompt)
                    candidates.append(child)
                    break

        return candidates


class RepairEmitter(BaseEmitter):
    """Repair broken-but-promising candidates via LM rewriting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("repair", config)
        self._broken_pool: List[VerseIndividual] = []

    def add_broken(self, individuals: List[VerseIndividual]) -> None:
        """Add broken candidates to the repair pool."""
        self._broken_pool.extend(individuals)
        if len(self._broken_pool) > 500:
            self._broken_pool = self._broken_pool[-500:]

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.phonetics import tokenize_line

        if not self._broken_pool:
            return []

        mutation_config = _mutation_config(self.config)
        constraint_config = _constraint_config(self.config)

        candidates = []
        to_repair = self._broken_pool[:batch_size]
        self._broken_pool = self._broken_pool[batch_size:]

        try:
            from evo_rhyme.mutation import get_rewriter
            rewriter = get_rewriter(mutation_config)
        except Exception:
            logger.warning("RepairEmitter: could not get rewriter")
            return []

        min_syl = mutation_config.get("min_syllables", 6)
        max_syl = mutation_config.get("max_syllables", 18)

        for individual in to_repair:
            try:
                repaired_lines = list(individual.lines)
                did_repair = False

                for i in range(0, len(individual.lines), 2):
                    if i + 1 >= len(individual.lines):
                        break

                    line1 = individual.lines[i]
                    line2 = individual.lines[i + 1]

                    tokens2 = tokenize_line(line2)
                    rhyme_target = tokens2[-1] if tokens2 else "night"

                    result = rewriter.repair(
                        line1, rhyme_target, (min_syl, max_syl),
                    )
                    if result:
                        repaired_lines[i] = result[0]
                        did_repair = True

                if did_repair:
                    style, prompt = _genome_from_metadata(individual)
                    new_ind = VerseIndividual(
                        lines=repaired_lines,
                        metadata={"origin": "repair_emitter"},
                    )
                    if passes_verse_constraints(new_ind, constraint_config):
                        analyze_verse_individual(new_ind)
                        _attach_genomes(new_ind, style=style, prompt=prompt)
                        candidates.append(new_ind)
            except Exception:
                logger.debug("RepairEmitter: repair failed for one individual", exc_info=True)

        return candidates


@dataclass
class EmitterStats:
    """Track per-emitter success over a sliding window."""
    new_niches: deque = field(default_factory=lambda: deque(maxlen=10))
    improved: deque = field(default_factory=lambda: deque(maxlen=10))
    elapsed_ms: deque = field(default_factory=lambda: deque(maxlen=10))
    generated: deque = field(default_factory=lambda: deque(maxlen=10))

    def record(self, new_n: int, improved_n: int, elapsed_ms: float = 0.0, generated: int = 0) -> None:
        self.new_niches.append(new_n)
        self.improved.append(improved_n)
        self.elapsed_ms.append(float(max(0.0, elapsed_ms)))
        self.generated.append(int(max(0, generated)))

    def success_score(self) -> float:
        """success = 2*(new niches) + 1*(improved elites) over window."""
        quality = 2.0 * sum(self.new_niches) + 1.0 * sum(self.improved)
        total_ms = sum(self.elapsed_ms) or 1.0
        efficiency = (sum(self.new_niches) + 0.5 * sum(self.improved)) / (total_ms / 1000.0)
        return quality + 0.3 * efficiency


class EmitterScheduler:
    """Adaptive scheduler: allocates budget proportional to emitter success."""

    def __init__(
        self,
        emitters: List[BaseEmitter],
        initial_weights: Optional[Dict[str, float]] = None,
        min_weight: float = 0.05,
        coverage_target: Optional[float] = None,
        coverage_boost_threshold: float = 0.35,
    ):
        self.emitters = emitters
        self._stats: Dict[str, EmitterStats] = {e.name: EmitterStats() for e in emitters}
        self._min_weight = min_weight
        self._coverage_target = coverage_target
        self._coverage_boost_threshold = coverage_boost_threshold

        if initial_weights:
            self._weights = {e.name: initial_weights.get(e.name, 1.0 / len(emitters)) for e in emitters}
        else:
            self._weights = {e.name: 1.0 / len(emitters) for e in emitters}

    def choose(self) -> BaseEmitter:
        """Choose an emitter proportional to current weights."""
        names = [e.name for e in self.emitters]
        weights = [self._weights[n] for n in names]
        chosen_name = random.choices(names, weights=weights, k=1)[0]
        return next(e for e in self.emitters if e.name == chosen_name)

    def allocate_budget(self, total_budget: int, archive: Any = None) -> Dict[str, int]:
        """Allocate batch sizes to each emitter based on weights.

        When coverage_target is set and archive.coverage() < coverage_boost_threshold,
        niche_targeting weight is temporarily scaled by 1.5x to accelerate exploration.
        """
        weights = dict(self._weights)
        if (
            archive is not None
            and self._coverage_target is not None
            and archive.coverage() < self._coverage_boost_threshold
        ):
            if "niche_targeting" in weights:
                weights = dict(weights)
                weights["niche_targeting"] = min(0.5, weights["niche_targeting"] * 1.5)
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}

        allocation: Dict[str, int] = {}
        remaining = total_budget
        emitter_list = list(self.emitters)

        for i, emitter in enumerate(emitter_list):
            if i == len(emitter_list) - 1:
                allocation[emitter.name] = remaining
            else:
                share = max(1, int(total_budget * weights[emitter.name]))
                share = min(share, remaining - (len(emitter_list) - i - 1))
                allocation[emitter.name] = share
                remaining -= share

        return allocation

    def update(self, emitter: BaseEmitter, result: EmitResult) -> None:
        """Update stats and recompute weights."""
        self._stats[emitter.name].record(
            result.new_niches,
            result.improved_niches,
            elapsed_ms=result.elapsed_ms,
            generated=result.generated,
        )

        total_success = sum(s.success_score() for s in self._stats.values())

        if total_success > 0:
            for name, stats in self._stats.items():
                raw = stats.success_score() / total_success
                self._weights[name] = max(self._min_weight, raw)

        total_w = sum(self._weights.values())
        if total_w > 0:
            self._weights = {k: v / total_w for k, v in self._weights.items()}

    def summary(self) -> Dict[str, Any]:
        """Return current scheduler state."""
        return {
            "weights": dict(self._weights),
            "stats": {
                name: {
                    "success_score": stats.success_score(),
                    "recent_new_niches": list(stats.new_niches),
                    "recent_improved": list(stats.improved),
                    "recent_elapsed_ms": list(stats.elapsed_ms),
                    "recent_generated": list(stats.generated),
                }
                for name, stats in self._stats.items()
            },
        }


def create_emitters(config: Dict[str, Any]) -> Tuple[List[BaseEmitter], EmitterScheduler]:
    """Factory: create specialized emitters and the adaptive scheduler.

    config should contain:
        theme_keywords: List[str]
        corpus_path: Optional[Path]
        min_syllables: int
        max_syllables: int
        schemes: List[str]  (allowed rhyme schemes)
        crossover_rate: float
    """
    random_emitter = RandomEmitter(config)
    rhyme_emitter = InternalRhymeEmitter(config)
    narrative_emitter = NarrativeEmitter(config)
    punchline_emitter = PunchlineEmitter(config)
    flow_emitter = FlowEmitter(config)
    imagery_emitter = ImageryEmitter(config)
    niche_emitter = NicheTargetingEmitter(config)

    lm_budget_per_gen = config.get("lm_mutation_budget_per_gen", 0)

    emitters: List[BaseEmitter] = [
        random_emitter,
        rhyme_emitter,
        narrative_emitter,
        punchline_emitter,
        flow_emitter,
        imagery_emitter,
        niche_emitter,
    ]

    initial_weights: Dict[str, float] = {
        "random": 0.12,
        "internal_rhyme": 0.15,
        "narrative": 0.13,
        "punchline": 0.13,
        "flow": 0.13,
        "imagery": 0.13,
        "niche_targeting": 0.21,
    }

    if lm_budget_per_gen > 0:
        repair_emitter = RepairEmitter(config)
        emitters.append(repair_emitter)
        total = sum(initial_weights.values())
        scale = 0.95 / total
        initial_weights = {k: v * scale for k, v in initial_weights.items()}
        initial_weights["repair"] = 0.05

    scheduler = EmitterScheduler(
        emitters,
        initial_weights=initial_weights,
        coverage_target=config.get("coverage_target"),
        coverage_boost_threshold=config.get("coverage_boost_threshold", 0.35),
    )
    return emitters, scheduler


def run_emitter_generation(
    archive: Any,
    emitters: List[BaseEmitter],
    scheduler: EmitterScheduler,
    total_budget: int,
    generation: int,
    score_fn: Callable,
    fitness_fn: Callable,
    novelty_weight: float = 0.3,
    min_coherence: float = 0.0,
) -> Dict[str, Any]:
    """Run one generation of emitter-based MAP-Elites.

    Args:
        archive: MAPElitesArchive instance
        emitters: List of emitter instances
        scheduler: EmitterScheduler
        total_budget: Total candidates to generate this generation
        generation: Current generation number
        score_fn: Callable(List[VerseIndividual]) -> List[Dict[str, float]]
        fitness_fn: Callable(Dict[str, float]) -> float
        novelty_weight: Weight for novelty in effective fitness (0.0-1.0)

    Returns:
        Dict with generation stats
    """
    allocation = scheduler.allocate_budget(total_budget, archive=archive)

    all_candidates: List[VerseIndividual] = []
    emitter_map: List[Tuple[BaseEmitter, int, int]] = []
    broken_candidates: List[VerseIndividual] = []

    for emitter in emitters:
        budget = allocation.get(emitter.name, 0)
        if budget <= 0:
            continue

        emit_start = time.perf_counter()
        batch = emitter.emit(archive, budget, generation)
        emit_ms = (time.perf_counter() - emit_start) * 1000.0
        start = len(all_candidates)
        all_candidates.extend(batch)
        emitter_map.append((emitter, start, len(all_candidates), emit_ms))

    if not all_candidates:
        return {"total_candidates": 0, "inserted": 0, "new_niches": 0}

    try:
        all_scores = score_fn(all_candidates)
        for cand, sc in zip(all_candidates, all_scores):
            cand.scores = sc
        try:
            from evo_rhyme.fitness import compute_parent_improvement
            for cand in all_candidates:
                parent_fit = (cand.metadata or {}).get("_parent_fitness")
                if parent_fit is not None and cand.scores:
                    cand.scores["parent_improvement"] = compute_parent_improvement(
                        cand.scores, parent_fit,
                    )
                elif cand.scores:
                    cand.scores.setdefault("parent_improvement", 0.5)
        except Exception:
            logger.debug("Parent improvement scoring skipped", exc_info=True)
        for cand in all_candidates:
            cand.fitness = fitness_fn(cand.scores)
    except Exception:
        logger.warning("Batch scoring failed", exc_info=True)
        return {"total_candidates": len(all_candidates), "inserted": 0, "new_niches": 0}

    if novelty_weight > 0:
        try:
            from evo_rhyme.scoring.novelty import embed_texts, NoveltyArchive
            texts = [" ".join(c.lines) for c in all_candidates]
            embs = embed_texts(texts)
            archive_texts = [" ".join(ind.lines) for ind in archive.top_k(min(50, archive.occupied_niches()))]
            if archive_texts:
                archive_embs = embed_texts(archive_texts)
                import numpy as np
                for i, cand in enumerate(all_candidates):
                    dists = [
                        float(np.linalg.norm(embs[i] - ae))
                        for ae in archive_embs
                    ]
                    novelty = min(dists) if dists else 0.5
                    cand.scores["novelty"] = min(1.0, novelty)
                    effective = (1.0 - novelty_weight) * (cand.fitness or 0) + novelty_weight * novelty
                    cand.metadata["effective_fitness"] = effective
        except Exception:
            logger.debug("Novelty computation failed, using raw fitness", exc_info=True)

    occupied_before = archive.occupied_niches()

    per_emitter_results: Dict[str, EmitResult] = {}

    for emitter, start, end, emit_ms in emitter_map:
        batch = all_candidates[start:end]
        new_niches = 0
        improved = 0

        for cand in batch:
            if min_coherence > 0:
                coh = (cand.scores or {}).get("coherence")
                if coh is not None and float(coh) < float(min_coherence):
                    continue
            occ_pre = archive.occupied_niches()
            inserted = archive.add(cand)
            if inserted:
                if archive.occupied_niches() > occ_pre:
                    new_niches += 1
                else:
                    improved += 1
            elif cand.scores and cand.scores.get("garbled_line_penalty", 0) > 0.15:
                broken_candidates.append(cand)

        result = EmitResult(
            candidates=batch,
            new_niches=new_niches,
            improved_niches=improved,
            total_inserted=new_niches + improved,
            elapsed_ms=emit_ms,
            generated=len(batch),
        )
        per_emitter_results[emitter.name] = result
        emitter.update(result)
        scheduler.update(emitter, result)

    repair = next((e for e in emitters if isinstance(e, RepairEmitter)), None)
    if repair and broken_candidates:
        repair.add_broken(broken_candidates)

    total_new = archive.occupied_niches() - occupied_before
    total_inserted = sum(r.total_inserted for r in per_emitter_results.values())

    return {
        "total_candidates": len(all_candidates),
        "inserted": total_inserted,
        "new_niches": total_new,
        "per_emitter": {
            name: {
                "budget": allocation.get(name, 0),
                "generated": len(r.candidates),
                "new_niches": r.new_niches,
                "improved": r.improved_niches,
                "elapsed_ms": r.elapsed_ms,
                "niches_per_sec": (
                    (r.new_niches + 0.5 * r.improved_niches) / max(1e-6, r.elapsed_ms / 1000.0)
                ),
            }
            for name, r in per_emitter_results.items()
        },
        "scheduler_weights": scheduler._weights,
        "broken_pool_size": len(repair._broken_pool) if repair else 0,
    }
