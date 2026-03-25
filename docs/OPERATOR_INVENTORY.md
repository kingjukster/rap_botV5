# Mutation & Crossover Operator Inventory + Instrumentation Proposal

This document enumerates mutation/crossover operators currently implemented in the repo and proposes an operator-level delta logging design so we can measure which operators actually improve fitness.

---

## 1) Couplet-level mutation operators (`evo_rhyme/mutation.py`)

All entries below share a common callable shape:
- **Inputs:** `(individual: CoupletIndividual, tail_to_words: Dict[str, List[str]], config: Any)`
- **Outputs:** `Optional[CoupletIndividual]` (`None` means operator failed/no-op)
- **Global constraints:** operator should preserve 2-line structure; if no valid edit exists, returns `None`; `corpus_vocab` constraints may filter candidate words.

### 1.1 Legacy / symbolic operators

| Operator key | Expected effect | Key constraints / failure modes |
|---|---|---|
| `end_word_swap` | Swap end word of one line with rhyme-family alternative. | Needs tokenized lines and rhyme tail alternatives; may fail if no alternatives. |
| `internal_rhyme_insert` | Replace internal token with same-tail token to increase internal rhyme. | Requires line length >= 3 and available same-tail alternatives. |
| `stressed_vowel_swap` | Replace token with another sharing stressed vowel. | Requires pronunciation + stressed vowels + vowel index hits. |
| `semantic_swap` | Replace generic terms with theme keywords. | Requires `config.theme_keywords`; only touches predefined generic words. |
| `syllable_adjust` | Add filler when too short or remove token when too long. | Depends on extracted features/syllable counts; bounded by min/max syllables. |
| `syntax_synonym` | Replace small set of words with synonyms. | Only predefined synonym map entries mutate. |
| `compression` | Remove a droppable function word. | Requires line length >= 4 and droppable token presence. |
| `expansion` | Insert a short filler token. | Requires non-empty filler list (possibly corpus-filtered). |
| `phrase_replace` | Swap fixed multiword phrases with alternatives. | Only triggers on fixed phrase inventory. |
| `rhyme_graph_expand` | Replace low-rhyme-connectivity token with better-rhyming one. | Requires enough tokens and graph/tail overlap opportunities. |
| `stress_repair` | Alias of `end_word_swap` for stressed/rhyme repair intent. | Inherits `end_word_swap` constraints. |
| `chain_extension` | Inject token from source rhyme tail into other line to extend chain motifs. | Needs source tail + target middle token slot + alternatives. |
| `multisyllable_rhyme` | Improve end rhyme by maximizing multisyllabic overlap with paired line. | Needs a candidate with strictly better overlap. |
| `embedding_rhyme_walk` | Replace internal token using embedding-space rhyme neighbors. | Requires embedding subsystem availability + neighbor hits. |

### 1.2 LM-backed operators

| Operator key | Expected effect | Key constraints / failure modes |
|---|---|---|
| `lm_rhyme_rewrite` | Rewrite one line to better rhyme with paired line’s end word. | Requires LM candidates; best candidate selected by tail overlap. |
| `lm_internal_rhyme` | Rewrite one line to improve internal rhyme at mid position. | Needs line length >= 3 and LM outputs with improved internal tail overlap. |
| `lm_theme_rewrite` | Rewrite weaker-semantic line toward theme relevance. | Requires `theme_keywords`; otherwise returns `None`. |
| `lm_paraphrase` | Paraphrase one line while preserving end rhyme. | Requires non-empty paraphrase candidates. |
| `lm_tighten` | Reduce syllable load of longer line (~2 syllables). | Depends on computed features/syllable range + LM response. |
| `lm_expand` | Increase syllable load of shorter line (~2 syllables). | Depends on computed features/syllable range + LM response. |
| `lm_structural_rewrite` | Reframe sentence structure while keeping rhyme target. | Needs opposite line end token + LM candidate. |
| `lm_metaphor_inject` | Increase imagery/metaphorical vividness while preserving rhyme target. | Needs LM candidates. |
| `lm_contrast_swap` | Shift emotional tone to opposing affect while preserving rhyme target. | Needs LM candidates. |
| `lm_score_guided` | Target weakest scoring dimension (internal rhyme/semantic/fluency/rhyme graph). | Requires `individual.scores` weakness < 0.5 and LM candidates. |
| `line_replace` | Replace full line with freshly generated LM line. | Needs opposite line’s rhyme target and successful LM call. |
| `block_replace` | Replace entire couplet with LM-generated new pair. | Requires LM response containing >= 2 clean lines. |

### 1.3 Mutation dispatcher

`mutate(...)` selects from weighted keys in `_MUTATION_FUNCS`, filters by LM budget, optionally `lm_only`, and tries weighted attempts + fallback pass. On success it returns one mutated `CoupletIndividual`; otherwise copy of input individual.

---

## 2) Couplet crossover operators (`evo_rhyme/crossover.py`)

| Function | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `crossover(parent1, parent2, config)` | Two `CoupletIndividual`s; optional `config` with `phrase_slice`. | New `CoupletIndividual`. | Phrase-slice branch only if syllable/structure checks pass and random gate passes. | Baseline recombination: line-level mix, optional phrase-level blending. |
| `_phrase_slice_crossover(parent_a, parent_b)` | Two `CoupletIndividual`s. | `Optional[CoupletIndividual]`. | Needs both selected lines length >= 4 and resulting line length >= 3. | Fine-grained phrase recombination for stylistic novelty. |

---

## 3) Verse (4-bar) mutation/crossover operators (`evo_rhyme/verse_evolution.py`)

| Function / operator | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `verse_crossover(...)` | Two `VerseIndividual`s + optional config. | New `VerseIndividual`. | Assumes 4-line verse for full behavior. | Mix verses using half-swap, single-line swap, best-of-each, optional phrase-slice. |
| `_verse_phrase_slice_crossover(...)` | Two 4-line verses. | `Optional[VerseIndividual]`. | Requires matching line index token length >= 4. | Phrase-level blend on one line to increase novelty. |
| `verse_mutate(...)` | `VerseIndividual`, mutation config/weights, optional constraints + LM budget. | New `VerseIndividual`. | Optional structural branch; may perform couplet mutate(s), LM rewrite, LM repair. | Multi-scale mutation pipeline at verse level. |
| Structural branch: `swap_couplets` | 4-line `VerseIndividual`. | New `VerseIndividual`. | Requires >=4 lines. | Reorders couplet blocks to explore macro flow. |
| Structural branch: `rewrite_transition_line` | Verse + boundary index + LM budget dict. | `Optional[VerseIndividual]`. | Requires budget and valid boundary index. | Improve coherence at couplet boundary via LM rewrite. |
| LM branch: `_lm_verse_rewrite` | Verse + config + LM budget. | `Optional[VerseIndividual]`. | Requires budget and exactly 4 resulting lines. | High-variance global rewrite preserving scheme/theme intent. |

---

## 4) Structural operators module (`evo_rhyme/structural_mutations.py`)

| Function | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `swap_couplets` | `VerseIndividual`. | `VerseIndividual`. | Needs 4 lines for true swap. | Swap first/second couplet positions. |
| `rewrite_transition_line` | Verse + `boundary_idx` + `lm_budget`. | `Optional[VerseIndividual]`. | Budget > 0 and valid boundary index. | Rewrite transition line for coherence while preserving rhyme context. |
| `couplet_swap_crossover` | Two `VerseIndividual`s. | `VerseIndividual`. | Needs both with >=4 lines for normal behavior. | Crossover at couplet granularity across parents. |

---

## 5) 16-bar block operators (`evo_rhyme/verse_composer.py`)

| Function | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `block_swap_crossover` | Two long-form `VerseIndividual`s + `block_size`. | `VerseIndividual`. | Requires at least 2 blocks. | One-point crossover at block boundary. |
| `single_block_swap` | Two verses + `block_size`. | `VerseIndividual`. | Requires >=1 block. | Replace exactly one block in parent1 with parent2’s aligned block. |
| `verse_16_crossover` | Two verses + `block_size`. | `VerseIndividual`. | Delegates to above two strategies. | Strategy-randomized 16-bar crossover. |
| `block_mutate` | Verse + `block_size` + config + LM budget. | `VerseIndividual`. | Requires at least one complete block and valid couplet slot. | Mutate a random couplet within one block using couplet `mutate()`. |
| `verse_16_mutate` | Verse + params. | `VerseIndividual`. | Wrapper around `block_mutate`. | Entry-point mutation for 16-bar evolution. |

---

## 6) Template-grammar operators (`evo_rhyme/template_grammar.py`)

| Operator | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `swap_connector` | `template: str` | `str` | Requires connector token in template. | Modify discourse flow word while preserving structure. |
| `swap_determiner` | `template: str` | `str` | Requires determiner token. | Lexical variation in noun phrase framing. |
| `swap_verb_form` | `template: str` | `str` | Requires verb placeholder (`{verb*}`). | Alter tense/aspect in template skeleton. |
| `insert_slot` | `template: str` | `str` | Requires slot count < 7 and at least 2 tokens. | Increase expressive capacity via extra placeholder slot. |
| `remove_slot` | `template: str` | `str` | Requires slot count >= 4. | Simplify template structure. |
| `_random_mutation` | `template: str` | `str` | Picks one function above. | One-step stochastic structural mutation. |
| `recombine(template1, template2)` | Two templates. | `str`. | Both need connector indices and non-empty recombined tail. | Template-level crossover at connector boundary. |

---

## 7) Style genome operators (`evo_rhyme/style_genome.py`)

| Operator | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `mutate_style_genome` | `StyleGenome`, `mutation_rate`. | `StyleGenome`. | Per-gene mutation gated by rate; value must change to another categorical index. | Controlled categorical style exploration. |
| `crossover_style_genome` | Two `StyleGenome`s. | `StyleGenome`. | Uniform per-gene parent pick. | Recombine style traits. |

---

## 8) Prompt genome operators (`evo_rhyme/prompt_genome.py`)

| Operator | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `mutate_prompt_genome` | `PromptGenome`, `mutation_scale`. | `PromptGenome`. | 35% per-gene mutate chance; clipped to [0,1]. | Smooth exploration in continuous prompt-control space. |
| `crossover_prompt_genome` | Two `PromptGenome`s. | `PromptGenome`. | Uniform per-gene parent pick. | Trait recombination for prompt strategy evolution. |

---

## 9) Weight genome operators

### 9.1 Couplet scorer weight tuner (`evo_rhyme/weight_tuner.py`)

| Operator | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `mutate_weight` | scalar weight + `sigma` + penalty flag | scalar weight | Positive clipped [0.01, 0.5], penalties clipped [-0.5, -0.01]. | Controlled Gaussian perturbation. |
| `mutate_genome` | weight dict + rate | mutated dict | Per-key Bernoulli mutation. | Explore weight settings. |
| `crossover_genomes` | two weight dicts | child dict | Uses defaults for missing keys; merges values. | Blend parental score-weight policies. |

### 9.2 Verse scorer weight tuner (`evo_rhyme/verse_weight_tuner.py`)

Same operator semantics as above (`mutate_weight`, `mutate_genome`, `crossover_genomes`) but applied to verse-level scoring weights.

---

## 10) Line-evolution mutation strategies (`evo_rhyme/line_evolution.py`)

These are not exported as a single registry, but they are real operator choices in line evolution.

| Strategy | Inputs | Outputs | Constraints | Expected effect |
|---|---|---|---|---|
| `_mutate_line_lm` (dispatch) | `ScoredLine`, config, LM budget | `Optional[str]` | LM budget required; picks one LM rewrite strategy by random band. | Single-line LM mutation (structural/metaphor/theme/rhyme/paraphrase/contrast). |
| `_mutate_line_legacy` | `ScoredLine`, config | `Optional[str]` | Requires mutable tokenizable line and corresponding resources. | Symbolic fallback edits (end-word swap, stressed vowel swap, chain-like substitution). |

---

## 11) Proposed instrumentation for operator-level improvement deltas

### 11.1 Goal
Track per-operator **causal delta**: did an operator increase fitness vs the exact pre-operator parent(s), under the current scoring function and run config?

### 11.2 Event schema (JSONL + optional DB table)

Emit one event per operator application attempt:

```json
{
  "run_id": 123,
  "generation": 17,
  "scope": "couplet|verse4|verse16|template|style_genome|prompt_genome|weight_genome|line",
  "operator_kind": "mutation|crossover",
  "operator_name": "lm_rhyme_rewrite",
  "attempt_id": "uuid",
  "parent_ids": [456, 789],
  "child_id": 999,
  "selected": true,
  "succeeded": true,
  "rejected": false,
  "reject_reason": null,
  "fitness_before": 0.612,
  "fitness_after": 0.645,
  "fitness_delta": 0.033,
  "scores_before": {"fluency": 0.71, "semantic": 0.53},
  "scores_after": {"fluency": 0.74, "semantic": 0.57},
  "score_deltas": {"fluency": 0.03, "semantic": 0.04},
  "lm_calls_used": 1,
  "latency_ms": 184,
  "timestamp": "2026-03-24T12:00:00Z",
  "config_fingerprint": "sha256:..."
}
```

### 11.3 Where to instrument

1. **Couplet mutate dispatcher** (`mutation.mutate`):
   - Log `selected`, `succeeded`, fallback usage, LM budget decrement.
   - Evaluate parent fitness/scores before call and child after success.
2. **Couplet crossover** (`crossover.crossover`, `_phrase_slice_crossover`):
   - Log which crossover branch produced child.
3. **Verse operators** (`verse_evolution.verse_crossover`, `verse_mutate`, structural + LM branches):
   - Log branch probabilities and chosen path.
4. **16-bar operators** (`verse_composer`):
   - Log block index/split info and resulting delta.
5. **Template/style/prompt/weight operators**:
   - Log objective delta used in each subsystem (template avg_fitness delta, benchmark meta-fitness delta, etc.).
6. **Line evolution**:
   - Log LM vs legacy strategy and line-score delta.

### 11.4 Minimal implementation plan

1. Add `OperatorEvent` dataclass + `OperatorTracer` with:
   - `record_attempt(...)`
   - `record_result(...)`
   - async-safe JSONL append and optional DB sink.
2. Inject optional `tracer` argument into mutation/crossover entry points.
3. Add helper `evaluate_delta(parent, child, scorer)` that returns fitness + per-score deltas.
4. Start with JSONL under `runs/<run_id>/operator_events.jsonl`; add DB ingestion later.

### 11.5 First analytics to run

- **Per-operator win rate:** `P(fitness_delta > 0 | succeeded)`
- **Expected uplift:** `E[fitness_delta | operator]`
- **Risk profile:** stddev + downside tail (`P(delta < -0.05)`).
- **Constraint rejection profile:** rejection rate by operator.
- **Budget ROI for LM ops:** uplift per LM call.
- **Context segmentation:** performance by theme, generation band, and parent fitness decile.

### 11.6 Practical guardrails

- Always log failed/no-op attempts (selection bias otherwise).
- Store score vectors, not just scalar fitness (operator may trade off dimensions).
- Keep config fingerprint + scorer version so deltas stay comparable across runs.
- Distinguish **operator success** from **survivor selection success** (child entering next gen).

