"""Tests for mutation quality: corpus_vocab filtering and garbled-line rejection.

Verifies that:
- Rule-based mutations respect corpus_vocab when provided
- verse_mutate rejects garbled output via the post-mutation gate
"""

import random

import pytest

from evo_rhyme.individual import CoupletIndividual, VerseIndividual
from evo_rhyme.mutation import (
    _end_word_swap,
    _internal_rhyme_insert,
    _stressed_vowel_swap,
    get_tail_to_words,
    mutate,
    LEGACY_MUTATION_WEIGHTS,
)


@pytest.fixture
def sample_couplet():
    return CoupletIndividual(
        line1="I spit pressure when the rival ignite",
        line2="mask on my face till I'm out of sight",
    )


@pytest.fixture
def corpus_vocab():
    return {
        "spit", "pressure", "rival", "ignite", "mask", "face", "sight",
        "flow", "show", "night", "right", "fight", "light", "might",
        "tight", "bright", "street", "heat", "beat", "feat", "seat",
        "dream", "team", "cream", "scheme", "real", "feel", "deal",
        "steel", "wheel", "kill", "still", "will", "chill", "ill",
        "grill", "skill", "thrill", "spill", "drill", "brain",
        "pain", "rain", "chain", "main", "game", "fame", "flame",
        "name", "aim", "claim", "time", "rhyme", "crime", "prime",
        "dime", "climb", "grind", "mind", "find", "kind", "blind",
        "crown", "down", "town", "frown", "ground", "sound", "round",
        "bound", "found", "pound", "war", "star", "bar", "car", "far",
        "got", "shot", "hot", "lot", "not", "spot", "top", "drop",
        "stop", "pop", "hop", "cop", "rock", "block", "clock", "knock",
        "talk", "walk", "stalk", "king", "ring", "thing", "bring",
        "sing", "swing", "sting", "spring", "string", "wing",
    }


class TestCorpusVocabFiltering:
    def test_end_word_swap_filters_by_corpus(self, sample_couplet, corpus_vocab):
        """end_word_swap should only pick words from corpus_vocab when set."""
        random.seed(42)
        tail_to_words = get_tail_to_words()
        config = {"corpus_vocab": corpus_vocab, "theme_keywords": ["pressure"]}

        successes = 0
        for _ in range(30):
            result = _end_word_swap(sample_couplet, tail_to_words, config)
            if result is not None:
                successes += 1
                words1 = result.line1.lower().split()
                words2 = result.line2.lower().split()
                for w in words1 + words2:
                    if w not in sample_couplet.line1.lower().split() and \
                       w not in sample_couplet.line2.lower().split():
                        assert w in corpus_vocab, (
                            f"New word '{w}' not in corpus_vocab"
                        )

    def test_internal_rhyme_insert_filters_by_corpus(self, sample_couplet, corpus_vocab):
        """internal_rhyme_insert should only pick words from corpus_vocab when set."""
        random.seed(123)
        tail_to_words = get_tail_to_words()
        config = {"corpus_vocab": corpus_vocab, "theme_keywords": ["pressure"]}

        for _ in range(30):
            result = _internal_rhyme_insert(sample_couplet, tail_to_words, config)
            if result is not None:
                orig_words = set(
                    sample_couplet.line1.lower().split()
                    + sample_couplet.line2.lower().split()
                )
                new_words = set(
                    result.line1.lower().split() + result.line2.lower().split()
                )
                introduced = new_words - orig_words
                for w in introduced:
                    assert w in corpus_vocab, (
                        f"New word '{w}' not in corpus_vocab"
                    )

    def test_mutate_with_corpus_vocab_no_archaic_words(self, sample_couplet, corpus_vocab):
        """mutate() with corpus_vocab should avoid archaic/rare words."""
        random.seed(7)
        archaic = {"bask", "beset", "smote", "dyke", "byre", "doth", "hath", "wert"}
        config = {"corpus_vocab": corpus_vocab, "theme_keywords": ["pressure"]}

        for _ in range(50):
            result = mutate(sample_couplet, config, weights=LEGACY_MUTATION_WEIGHTS)
            all_words = set(result.line1.lower().split() + result.line2.lower().split())
            overlap = all_words & archaic
            assert not overlap, f"Archaic words in output: {overlap}"


class TestGarbledGate:
    def test_verse_mutate_rejects_garbled(self):
        """verse_mutate should reject mutations that produce garbled lines."""
        from evo_rhyme.verse_evolution import verse_mutate
        from evo_rhyme.fitness import _score_verse_garbled_line_penalty

        garbled_verse = VerseIndividual(
            lines=[
                "I spit pressure when the rival ignite",
                "mask on my face and the night feel right",
                "survival got me running through the fight",
                "they said the crown bound but I moved the light",
            ],
        )

        random.seed(0)
        config = {"theme_keywords": ["pressure", "mask"], "min_syllables": 6, "max_syllables": 18}

        garbled_count = 0
        total = 30
        for _ in range(total):
            result = verse_mutate(garbled_verse, config)
            penalty = _score_verse_garbled_line_penalty(result)
            if penalty > 0.25:
                garbled_count += 1

        assert garbled_count == 0, (
            f"{garbled_count}/{total} mutations passed with garbled_line_penalty > 0.25"
        )
