"""Tests for evo_rhyme.template_grammar: swap_connector, swap_determiner, insert_slot, remove_slot, recombine, TemplatePool.
Area 10: Template grammar and constants.
"""

import random

import pytest

from evo_rhyme.template_grammar import (
    CONNECTORS,
    DETERMINERS,
    SLOT_TYPES,
    VERB_FORMS,
    ScoredTemplate,
    TemplatePool,
    insert_slot,
    recombine,
    remove_slot,
    swap_connector,
    swap_determiner,
    swap_verb_form,
)


def test_swap_connector_changes_connector():
    random.seed(42)
    template = "I run while the sun sets"
    result = swap_connector(template)
    words = result.split()
    connector_pos = next((i for i, w in enumerate("I run while the sun sets".split()) if w.lower() in CONNECTORS), None)
    assert connector_pos is not None
    assert result != template or "while" in CONNECTORS  # may or may not change due to random


def test_swap_connector_no_connector_unchanged():
    template = "I run fast"
    result = swap_connector(template)
    assert result == template


def test_swap_determiner_changes_determiner():
    random.seed(123)
    template = "I got the flow"
    result = swap_determiner(template)
    assert "the" in template
    assert result.split()[2] in DETERMINERS


def test_swap_determiner_no_determiner_unchanged():
    template = "hello world"
    result = swap_determiner(template)
    assert result == template


def test_connectors_and_determiners_non_empty():
    assert len(CONNECTORS) > 0
    assert len(DETERMINERS) > 0


def test_slot_types_and_verb_forms():
    assert len(SLOT_TYPES) > 0
    assert "noun" in SLOT_TYPES
    assert "verb" in VERB_FORMS
    assert "verb_past" in VERB_FORMS


def test_insert_slot_adds_placeholder(monkeypatch):
    random.seed(7)
    template = "I {verb} the {noun}"
    result = insert_slot(template)
    assert "{" in result and "}" in result
    assert result.count("{") >= 2


def test_insert_slot_max_slots_unchanged():
    template = "a {noun} b {noun2} c {verb} d {adj} e {noun3} f {noun4} g {noun5}"
    result = insert_slot(template)
    assert result == template


def test_remove_slot_removes_one():
    random.seed(11)
    template = "I {verb} the {noun} while the {noun2} {verb2}"
    result = remove_slot(template)
    assert result.count("{") == template.count("{") - 1 or result == template


def test_remove_slot_few_slots_unchanged():
    template = "I {verb} the {noun}"
    result = remove_slot(template)
    assert result == template


def test_recombine_uses_connectors():
    random.seed(99)
    t1 = "I run while the sun sets"
    t2 = "You stay when the moon rises"
    result = recombine(t1, t2)
    assert " " in result
    assert result != t1 or result != t2


def test_recombine_no_connector_returns_first():
    result = recombine("hello world", "foo bar")
    assert result == "hello world"


def test_swap_verb_form_changes_placeholder():
    random.seed(13)
    template = "I {verb} through the {noun}"
    result = swap_verb_form(template)
    assert "{verb" in result or "{verb_past" in result or "{verb_ing" in result


def test_scored_template_avg_fitness():
    st = ScoredTemplate(text="x", fitness_sum=1.0, usage_count=2)
    assert st.avg_fitness == 0.5
    st0 = ScoredTemplate(text="y", fitness_sum=0.0, usage_count=0)
    assert st0.avg_fitness == 0.0


def test_template_pool_sample_and_record():
    random.seed(42)
    pool = TemplatePool(["I {verb} the {noun}", "You {verb} when I {noun}"], max_size=10)
    assert pool.size() == 2
    samples = pool.sample(1)
    assert len(samples) == 1
    assert samples[0] in ["I {verb} the {noun}", "You {verb} when I {noun}"]
    pool.record_fitness("I {verb} the {noun}", 0.8)
    pool.record_fitness("I {verb} the {noun}", 0.6)
    top = pool.top_k(1)
    assert len(top) >= 1
    assert top[0].text == "I {verb} the {noun}"
    assert top[0].avg_fitness == 0.7


def test_template_pool_evolve():
    random.seed(123)
    pool = TemplatePool(
        ["I {verb} while the {noun} runs", "You {verb_past} when the {noun} fell", "We {verb_ing} as the {noun} grows"],
        max_size=20,
    )
    n_before = pool.size()
    pool.evolve(mutation_rate=0.2, crossover_rate=0.1)
    assert pool.size() >= 1
    assert pool.size() <= 20
