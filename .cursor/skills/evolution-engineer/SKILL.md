---
name: evolution-engineer
description: >-
  Optimizes the rap_bot evolution loop, mutation operators, emitters, population,
  and fitness scoring. Use when modifying mutation.py, emitters.py, population.py,
  fitness.py, evo_rhyme/scoring/, or when adding mutation operators, improving
  diversity/convergence, or tuning the MAP-Elites QD loop.
---

# Evolution Engineer

Specialist for the rap_bot evolution engine. Focus: mutation operators, emitters, population seeding, and fitness/scoring.

## Scope

| Module | Purpose |
|--------|---------|
| `evo_rhyme/mutation.py` | Couplet mutation operators (LM-backed + legacy word-swap) |
| `evo_rhyme/emitters.py` | MAP-Elites emitters, scheduler, batch generation |
| `evo_rhyme/population.py` | Seed generation, VerseSeedGenerator, init modes |
| `evo_rhyme/fitness.py` | Score aggregation, weights, caching |
| `evo_rhyme/scoring/` | Per-dimension scorers (coherence, punchline, rhyme_chain, etc.) |
| `evo_rhyme/verse_evolution.py` | Main evolution loop, archive, QD config |

## Core Invariants

- **Reproducibility**: All randomness must be seed-controlled (`random.seed` or RNG passed through).
- **Logging**: Mutation type, emitter origin, and key decisions must be logged (no silent mutations).
- **Separation of concerns**: Data (individuals), evolution (mutate/crossover), scoring (fitness) are separate layers.

## Adding a New Mutation Operator

1. Implement a function `_my_operator(individual, tail_to_words, config) -> Optional[CoupletIndividual]`.
2. Return `None` on failure; use `_copy_individual(ind, line1, line2)` for new individuals.
3. Register in `_MUTATION_FUNCS` and add weight to `MUTATION_WEIGHTS`.
4. If LM-backed: accept `lm_budget` in `mutate()`, decrement on success, and exclude when exhausted.
5. Add a unit test in `tests/test_evo_rhyme/`.

```python
# Pattern in mutation.py
def _my_operator(ind, tail_to_words, config) -> Optional[CoupletIndividual]:
    # One conservative change
    new_line = ...
    return _copy_individual(ind, new_line, ind.line2) if new_line else None

_MUTATION_FUNCS["my_operator"] = _my_operator
MUTATION_WEIGHTS["my_operator"] = 0.05
```

## Emitter Best Practices

- Each emitter extends `BaseEmitter` and implements `emit(archive, batch_size, generation) -> List[VerseIndividual]`.
- Use `_constraint_config(config)` for `passes_verse_constraints`.
- Attach genomes via `_attach_genomes(ind, style, prompt)` for style/prompt genome tracking.
- For directed emitters: bias `MUTATION_WEIGHTS` in `_focused_weights` and adjust `StyleGenome` in `_style_bias`.
- `NicheTargetingEmitter` maps niche labels → mutation hints (`prefer_ops`, `extra_keywords`, etc.).

## Fitness / Scoring

- `DEFAULT_WEIGHTS` and `VERSE_DEFAULT_WEIGHTS` in fitness.py define component weights.
- `ngram_fluency` and `lexical_validity` protect against rhyme-exploiting nonsense; do not weaken them without strong justification.
- Penalties (e.g. `rhyme_family_repetition_penalty`, `corpus_overlap_penalty`) are negative weights.
- New score dimensions: add scorer in `evo_rhyme/scoring/`, call from `score_verse`/`score_vector`, add to weights dict.
- `NGRAM_FLOOR` rejects candidates with unnatural phrase structure when corpus is available.

## Improving Diversity & Convergence

**Diversity**

- Increase `novelty_weight` in `run_emitter_generation` for more exploration.
- Add or tune `NicheTargetingEmitter`; use `_refresh_targets` and `_get_niche_mutation_hints`.
- Emitter scheduler: `coverage_target` + `coverage_boost_threshold` temporarily boost niche_targeting when coverage is low.
- Consider `RandomEmitter` with `prompt_llm_fraction` for LM-prompted seeds.

**Convergence**

- Strengthen `lm_score_guided` weight to target weakest score dimensions.
- Tune `DirectedMutationEmitter` focuses (internal_rhyme, narrative, punchline, flow, imagery).
- `lm_budget_per_gen` limits LM calls; allocate to high-impact operators.
- Ensure `EmitterScheduler.update()` receives `EmitResult` so weights adapt to success.

## Evolution Loop Flow

1. `run_emitter_generation` or main QD loop allocates budget via `scheduler.allocate_budget()`.
2. Each emitter produces candidates; `verse_mutate` / `verse_crossover` in `verse_evolution.py`.
3. `score_fn` / `fitness_fn` score candidates; `archive.add()` for MAP-Elites insertion.
4. `RepairEmitter` receives broken candidates (high `garbled_line_penalty`) for LM repair.
5. Scheduler weights updated from `new_niches` and `improved_niches`.

## Quick References

- `MUTATION_WEIGHTS` vs `LEGACY_MUTATION_WEIGHTS` in mutation.py.
- `create_emitters()` factory in emitters.py for emitter list + scheduler.
- `VerseSeedGenerator` in population.py: `init_mode` ("mixed", "lm", "template").
- `evo_rhyme/scoring/`: coherence, punchline, rhyme_chain, rhyme_graph_network, line_penalty, etc.
