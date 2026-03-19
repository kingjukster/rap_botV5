# Evolution Strategy
(See docs/CONTEXT.md)

## Genome

### CoupletIndividual

- `line1`, `line2` — text
- `features1`, `features2` — LineFeatures (tokens, phonemes, syllable_count, stress_pattern, end_tail)
- `scores` — component scores (end_rhyme, internal_rhyme, etc.)
- `fitness` — weighted sum

### VerseIndividual

- `lines` — list of strings (4, 8, or 16)
- `features` — VerseFeatures (tokens_per_line, phonemes, syllable_counts, end_tails, stress_patterns)
- `structure` — VerseStructure (scheme, roles, callbacks)
- `scores`, `fitness`

---

## Fitness Function

F = weighted sum of components (capped at 0.95)

### Positive components (config/evolution.yaml)

| Component | Weight | Role |
|-----------|--------|------|
| end_rhyme | 0.18 | End-word rhyme quality |
| internal_rhyme | 0.16 | Internal rhyme density |
| rhyme_graph | 0.12 | Rhyme family graph |
| multisyllabic | 0.08 | Multisyllabic rhyme overlap |
| syllable_balance | 0.06 | Line length consistency |
| stress_alignment | 0.10 | Stress pattern match |
| semantic | 0.10 | Theme relevance |
| fluency | 0.08 | General fluency |
| lexical_validity | 0.06 | Word validity |
| ngram_fluency | 0.15 | Phrase appears in corpus (anti-nonsense) |
| novelty | 0.05 | Distance from corpus |

### Penalties (negative weights)

weak_tail_penalty, repetition_penalty, theme_word_repetition_penalty, rhyme_family_repetition_penalty, identical_line_penalty, near_duplicate_penalty, template_penalty, corpus_overlap_penalty, theme_penalty.

Verse evolution uses `VERSE_DEFAULT_WEIGHTS` in fitness.py with optional coherence, punchline, rhyme_chain.

---

## Mutation Operators

### LM-backed (mutation.MUTATION_WEIGHTS)

lm_rhyme_rewrite, lm_internal_rhyme, lm_theme_rewrite, lm_paraphrase, lm_structural_rewrite, lm_metaphor_inject, lm_contrast_swap, lm_score_guided, lm_tighten, lm_expand, stressed_vowel_swap, syllable_adjust, rhyme_graph_expand, embedding_rhyme_walk, chain_extension, end_word_swap, multisyllable_rhyme, line_replace, block_replace.

### Legacy (mutation.LEGACY_MUTATION_WEIGHTS)

end_word_swap, internal_rhyme_insert, stressed_vowel_swap, semantic_swap, syntax_synonym, compression, expansion, phrase_replace, rhyme_graph_expand, stress_repair.

---

## Crossover

- **Line-level** — line1 from parent A, line2 from parent B (or vice versa)
- **Phrase-slice** — when structure matches (syllable counts similar), take first half of line from A and second half from B (config: phrase_slice)

---

## Selection

- **Elitism** — Top N by fitness survive
- **Tournament** — k-way tournament for parents
- **Random immigrants** — Inject fresh individuals each generation

---

## MAP-Elites (QD)

### Behavioral dimensions

- **rhyme_density** — internal_rhyme score → very_low … very_high
- **intensity** — aggressive keyword ratio → very_calm … very_aggressive
- **syllable_tightness** — avg syllables per line
- **theme_balance** — semantic score > 0.3
- **line_length_variance** — std of syllable counts

### Archive modes

- **compact_style** — reduced dimension grid
- **ultra_compact** — minimal grid
- **default** — phonetic graph mode

---

## Future

- Policy-guided mutation (learned vs static)
- Learned fitness weights (verse weight tuner)
- Embedding-based rhyme search
