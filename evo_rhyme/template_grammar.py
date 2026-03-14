"""
evo_rhyme/template_grammar.py

Evolutionary template pool. Templates are structural patterns like
``I {verb_past} through the {noun} while the {noun2} still {verb}``
that get filled with vocabulary to produce rap lines.

This module evolves the *templates themselves* via mutation operators that
modify connectors, determiners, verb forms, and slot counts, plus a
crossover operator that recombines two templates at connector boundaries.

A TemplatePool tracks per-template fitness statistics so that
high-performing structures are sampled more often and low-performing ones
are mutated or replaced.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

CONNECTORS: Set[str] = {"while", "when", "but", "till", "as", "where", "and", "or", "if", "then"}
DETERMINERS: Set[str] = {"the", "my", "this", "no", "every", "your", "our", "a"}
VERB_FORMS: Set[str] = {"verb", "verb_past", "verb_ing"}
SLOT_TYPES: List[str] = ["noun", "verb", "adjective"]

_MAX_SLOTS = 7
_MIN_SLOTS_FOR_REMOVE = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_slots(template: str) -> int:
    return len(_PLACEHOLDER_RE.findall(template))


def _pick_other(current: str, options: Set[str]) -> str:
    """Pick a random element from *options* that differs from *current*."""
    alternatives = [o for o in options if o != current]
    return random.choice(alternatives) if alternatives else current


def _tokenize(template: str) -> List[str]:
    """Split template into tokens preserving whitespace structure."""
    return template.split()


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


def swap_connector(template: str) -> str:
    """Replace a random connector word with another from the connector set."""
    tokens = _tokenize(template)
    indices = [i for i, t in enumerate(tokens) if t.lower() in CONNECTORS]
    if not indices:
        return template
    idx = random.choice(indices)
    old = tokens[idx]
    new = _pick_other(old.lower(), CONNECTORS)
    tokens[idx] = new if old.islower() else new.capitalize()
    return " ".join(tokens)


def swap_determiner(template: str) -> str:
    """Replace a random determiner with another from the determiner set."""
    tokens = _tokenize(template)
    indices = [i for i, t in enumerate(tokens) if t.lower() in DETERMINERS]
    if not indices:
        return template
    idx = random.choice(indices)
    old = tokens[idx]
    new = _pick_other(old.lower(), DETERMINERS)
    tokens[idx] = new if old.islower() else new.capitalize()
    return " ".join(tokens)


def swap_verb_form(template: str) -> str:
    """Change a random verb placeholder to a different verb tense form.

    Valid forms cycle among ``{verb}``, ``{verb_past}``, ``{verb_ing}``.
    Numbered suffixes (e.g. ``{verb2}``) are preserved.
    """
    matches = list(_PLACEHOLDER_RE.finditer(template))
    verb_matches = []
    for m in matches:
        name = m.group(1)
        base = re.sub(r"\d+$", "", name)
        if base in VERB_FORMS:
            verb_matches.append(m)
    if not verb_matches:
        return template

    m = random.choice(verb_matches)
    name = m.group(1)
    suffix_match = re.search(r"(\d+)$", name)
    num_suffix = suffix_match.group(1) if suffix_match else ""
    base = re.sub(r"\d+$", "", name)
    new_base = _pick_other(base, VERB_FORMS)
    new_name = f"{new_base}{num_suffix}"
    return template[: m.start()] + "{" + new_name + "}" + template[m.end() :]


def insert_slot(template: str) -> str:
    """Insert a new placeholder at a random position between words.

    Only proceeds if the resulting template would have <= 7 slots.
    """
    if _count_slots(template) >= _MAX_SLOTS:
        return template

    tokens = _tokenize(template)
    if len(tokens) < 2:
        return template

    slot_base = random.choice(SLOT_TYPES)
    existing_names = set(_PLACEHOLDER_RE.findall(template))
    name = slot_base
    counter = 2
    while name in existing_names:
        name = f"{slot_base}{counter}"
        counter += 1

    insert_pos = random.randint(1, len(tokens) - 1)
    tokens.insert(insert_pos, "{" + name + "}")
    return " ".join(tokens)


def remove_slot(template: str) -> str:
    """Remove a random placeholder if the template has >= 4 slots."""
    matches = list(_PLACEHOLDER_RE.finditer(template))
    if len(matches) < _MIN_SLOTS_FOR_REMOVE:
        return template

    m = random.choice(matches)
    before = template[: m.start()].rstrip()
    after = template[m.end() :].lstrip()
    result = before + (" " if before and after else "") + after
    return re.sub(r"  +", " ", result).strip()


def recombine(template1: str, template2: str) -> str:
    """Crossover two templates by splitting each at a connector and swapping halves."""
    tokens1 = _tokenize(template1)
    tokens2 = _tokenize(template2)

    conn_idx1 = [i for i, t in enumerate(tokens1) if t.lower() in CONNECTORS]
    conn_idx2 = [i for i, t in enumerate(tokens2) if t.lower() in CONNECTORS]

    if not conn_idx1 or not conn_idx2:
        return template1

    split1 = random.choice(conn_idx1)
    split2 = random.choice(conn_idx2)

    head = tokens1[: split1 + 1]
    tail = tokens2[split2 + 1 :]

    if not tail:
        return template1

    result = " ".join(head + tail)
    return result


# ---------------------------------------------------------------------------
# All mutation operators (for random selection)
# ---------------------------------------------------------------------------

_MUTATIONS = [swap_connector, swap_determiner, swap_verb_form, insert_slot, remove_slot]


def _random_mutation(template: str) -> str:
    """Apply a single random mutation operator."""
    fn = random.choice(_MUTATIONS)
    return fn(template)


# ---------------------------------------------------------------------------
# ScoredTemplate / TemplatePool
# ---------------------------------------------------------------------------


@dataclass
class ScoredTemplate:
    """A template string with cumulative fitness tracking."""

    text: str
    fitness_sum: float = 0.0
    usage_count: int = 0

    @property
    def avg_fitness(self) -> float:
        return self.fitness_sum / max(1, self.usage_count)


class TemplatePool:
    """Evolvable pool of template strings with fitness-weighted sampling.

    Parameters
    ----------
    base_templates:
        Initial set of template strings.
    max_size:
        Hard cap on pool size after evolution.
    """

    def __init__(self, base_templates: List[str], max_size: int = 100) -> None:
        seen: Set[str] = set()
        self._templates: List[ScoredTemplate] = []
        for t in base_templates:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                self._templates.append(ScoredTemplate(text=t))
        self._max_size = max_size
        self._initial_size = len(self._templates)
        self._text_index: Dict[str, ScoredTemplate] = {st.text: st for st in self._templates}
        logger.info("TemplatePool initialized with %d templates (max %d)", len(self._templates), max_size)

    # -- sampling ----------------------------------------------------------

    def sample(self, k: int = 1) -> List[str]:
        """Sample *k* templates, weighted by fitness.

        Templates that have never been used get a small positive weight so
        they are still discoverable.
        """
        if not self._templates:
            return []
        k = min(k, len(self._templates))

        weights: List[float] = []
        for st in self._templates:
            w = st.avg_fitness if st.usage_count > 0 else 0.1
            weights.append(max(w, 0.01))

        chosen = random.choices(self._templates, weights=weights, k=k)
        return [st.text for st in chosen]

    # -- fitness recording -------------------------------------------------

    def record_fitness(self, template_text: str, fitness: float) -> None:
        """Record that a line generated from *template_text* achieved *fitness*."""
        st = self._text_index.get(template_text)
        if st is None:
            logger.debug("record_fitness called for unknown template; ignoring")
            return
        st.fitness_sum += fitness
        st.usage_count += 1

    # -- evolution ---------------------------------------------------------

    def evolve(self, mutation_rate: float = 0.2, crossover_rate: float = 0.1) -> None:
        """Evolve the pool in-place.

        1. Sort by avg_fitness.
        2. Bottom 20 %: replace with mutated variants.
        3. Top 20 %: produce offspring via crossover.
        4. Deduplicate, trim to max_size, back-fill if under initial size.
        """
        if len(self._templates) < 3:
            return

        self._templates.sort(key=lambda st: st.avg_fitness)
        n = len(self._templates)
        bottom_k = max(1, int(n * mutation_rate))
        top_k = max(1, int(n * crossover_rate))

        # mutate bottom slice
        for i in range(bottom_k):
            old_text = self._templates[i].text
            new_text = _random_mutation(old_text)
            if new_text != old_text:
                self._templates[i] = ScoredTemplate(text=new_text)

        # crossover from top slice
        top_slice = self._templates[-top_k:]
        new_offspring: List[ScoredTemplate] = []
        for _ in range(top_k):
            if len(top_slice) >= 2:
                p1, p2 = random.sample(top_slice, 2)
            else:
                p1 = p2 = top_slice[0]
            child_text = recombine(p1.text, p2.text)
            new_offspring.append(ScoredTemplate(text=child_text))
        self._templates.extend(new_offspring)

        # deduplicate
        seen: Set[str] = set()
        deduped: List[ScoredTemplate] = []
        for st in self._templates:
            if st.text not in seen:
                seen.add(st.text)
                deduped.append(st)
        self._templates = deduped

        # trim to max_size (keep highest fitness first)
        if len(self._templates) > self._max_size:
            self._templates.sort(key=lambda st: st.avg_fitness, reverse=True)
            self._templates = self._templates[: self._max_size]

        # back-fill if below initial size
        while len(self._templates) < self._initial_size:
            donor = random.choice(self._templates[:max(1, len(self._templates))])
            mutated = _random_mutation(donor.text)
            if mutated not in {st.text for st in self._templates}:
                self._templates.append(ScoredTemplate(text=mutated))
            else:
                break

        # rebuild index
        self._text_index = {st.text: st for st in self._templates}
        logger.info("TemplatePool evolved: %d templates", len(self._templates))

    # -- introspection -----------------------------------------------------

    def top_k(self, k: int = 10) -> List[ScoredTemplate]:
        """Return top *k* templates by average fitness (descending)."""
        ranked = sorted(self._templates, key=lambda st: st.avg_fitness, reverse=True)
        return ranked[:k]

    def size(self) -> int:
        """Current number of templates in the pool."""
        return len(self._templates)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_POOL: Optional[TemplatePool] = None


def get_template_pool(base_templates: Optional[List[str]] = None) -> TemplatePool:
    """Get or create the global template pool.

    If no pool exists yet, one is created from *base_templates*.  When
    *base_templates* is ``None``, templates are loaded via
    ``evo_rhyme.generator.load_templates()``.
    """
    global _POOL
    if _POOL is None:
        if base_templates is None:
            from evo_rhyme.generator import load_templates
            entries = load_templates()
            base_templates = [e.text for e in entries]
        _POOL = TemplatePool(base_templates)
    return _POOL
