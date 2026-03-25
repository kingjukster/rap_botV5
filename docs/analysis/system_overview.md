# RapBotV5 Research System Overview

## Scope and method
This overview is based on static inspection of `evo_rhyme/`, `scripts/`, `webapp/`, and `config/`.
It treats the repository as a hybrid evolutionary-computational-creativity system with two main modes:

- **Couplet evolution** (`evo_rhyme/evolution.py`, `scripts/run_couplet_evolution.py`).
- **Verse QD evolution (MAP-Elites)** (`evo_rhyme/verse_evolution.py`, `evo_rhyme/archive.py`, `scripts/run_verse_qd.py`).

---

## 1) Evolution loop(s)

### 1.1 Couplet EA loop (canonical GA)
Primary loop: `evo_rhyme.evolution.evolve`.

Per generation:
1. Analyze individuals (`analyze_individual`).
2. Score components (`score_couplet`).
3. Aggregate fitness (`compute_fitness`).
4. Sort by fitness.
5. Keep elites (optionally niche-preserving by rhyme family).
6. Create offspring via tournament selection → crossover → mutation.
7. Apply hard constraints (`passes_constraints`) and optional acceptance floors.
8. Inject immigrants.
9. Persist generation/candidate records via DB logger if enabled.

Notable options:
- `multiobjective=True` switches to Pareto ranking + crowding (`evolve_multiobjective`).
- Optional embedding-based semantics (`SiameseRhymeScorer`).
- Optional LM-fluency blend and n-gram floors to block nonsense phrase structures.

### 1.2 Verse loop (non-QD)
`evolve_verse_population` runs 4-line verse evolution similarly, with verse-specific crossover/mutation and verse fitness.

### 1.3 QD loop (MAP-Elites + EA operators)
Primary QD path is `run_verse_qd.py` + `verse_evolution` QD config + `MAPElitesArchive`.

Architecture:
- Behavioral descriptor mapping from a verse to niche coordinates (e.g., rhyme density, intensity, style tone depending on mode).
- Archive stores best individual per niche (`MAPElitesArchive.add`).
- Parent sampling from occupied niches (`sample_parents`) + mutation/crossover and/or emitter strategies (`evo_rhyme/emitters.py`).
- Coverage / occupied niches are tracked and logged per generation.

Archive modes:
- `default_verse_dimensions` (high-dimensional).
- `compact_style` and `ultra_compact` for practical coverage targets.

---

## 2) Fitness function

### 2.1 Couplet fitness
`score_couplet` computes components (rhyme, semantic, fluency, penalties, novelty, coherence, etc.).
`compute_fitness` does weighted sum plus population-level penalties:

- **Positive signals**: end/internal rhyme, rhyme graph, multisyllabic overlap, stress/syllable alignment, semantic relevance, fluency, lexical validity, n-gram fluency, novelty.
- **Penalties**: repetition, template patterns, near duplicates, corpus overlap, theme miss.
- **Population anti-collapse terms**: rhyme-family diversity penalty and repeated-shell penalty.
- **Safety caps/floors**: fitness cap (`FITNESS_CAP=0.95`) and optional n-gram floor (default 0.2 with corpus).

### 2.2 Verse fitness
`score_verse` computes a richer vector:
- rhyme scheme + internal/rhyme chain + graph metrics
- fluency + LM fluency + lexical validity
- semantic, coherence, punchline
- novelty and multiple repetition/template/corpus penalties
- flow alignment/continuity and beat-fit
- style/prompt adherence terms

`compute_verse_fitness` is a weighted linear scalarization (`VERSE_DEFAULT_WEIGHTS`).
Recent comments indicate explicit rebalancing toward coherence/novelty and away from LM-fluency over-dominance.

---

## 3) Mutation / crossover logic

### 3.1 Couplet crossover
`evo_rhyme.crossover.crossover`:
- default line-level crossover (line1 from parent A, line2 from parent B)
- optional phrase-slice crossover under syllable-structure compatibility checks.

### 3.2 Couplet mutation
`evo_rhyme.mutation` includes a mixed operator portfolio:
- legacy symbolic ops (end-word swap, stressed-vowel swap, syllable adjust, etc.)
- LM-heavy rewrite family (`lm_rhyme_rewrite`, `lm_theme_rewrite`, etc.)
- rhyme-graph and embedding walk operators
- chain extension / block or line replacements

Operators are sampled by configurable weights. Constraint gating is applied post-mutation.

### 3.3 Verse mutation/crossover
`verse_crossover` supports half-swap, single-line swap, and phrase-slice.
`verse_mutate` can perform:
- couplet-level mutation within verse,
- whole-verse LM rewrite,
- optional structural mutations (`swap_couplets`, transition-line rewrite).

---

## 4) Policy learning / adaptive controls

The system includes a control-learning pipeline:

- **Control schema/registry**: `evo_rhyme/experiment_controls.py`.
- **Predictive attribution model**: `evo_rhyme/control_model.py` (RandomForest over flattened controls).
- **Policy updater**: `scripts/update_learned_policy.py`:
  - Loads recent run outcomes from DB,
  - ranks configs by fitness with failed-run penalty,
  - writes `artifacts/learned_policy.json`.
- **Continuous learning loop**: `scripts/run_learning_loop.py` + `scripts/run_continuous.py`.

Policy modes in runners:
- `static`, `learned`, `explore_mix` (epsilon exploration).

Interpretation: this is not gradient policy learning; it is **meta-optimization of control settings from run history** (bandit/evolutionary-control flavor).

---

## 5) Database schema (research logging backbone)

Initialized via `scripts/init_db.py`; runtime access in `evo_rhyme/db.py`.

Core tables:
- `runs`: run metadata, config JSON, status, failure reason, experiment links.
- `generations`: per-generation stats (`best_fitness`, `avg_fitness`, diversity, acceptance rate, extra JSON).
- `candidates`: candidate artifacts with lines + fitness + score vector.
- `lineage`: parent-child edges for ancestry graph.
- `archive_cells`: MAP-Elites cell snapshots by run.
- `score_cache`: text-hash keyed cached scores.
- experiment tables: `experiments`, `experiment_arms`.
- seed/artifact tables: `seed_bank`, `artifacts`, plus song catalog tables.

Web UI/API (`webapp/`) exposes this data for runs, generations, archive, lineage, and analysis dashboards.

---

## 6) System-level research assessment

### Strengths
- Explicit QD support with configurable descriptor spaces.
- Rich multi-component fitness with anti-collapse penalties.
- Good observability architecture (runs/generations/candidates/lineage/archive).
- Policy-control feedback loop with failure-aware updates.

### Gaps / risks
- Heavy dependence on external model calls in mutation/proposal paths may add non-deterministic variance.
- Linear scalarization of complex aesthetics may produce gaming between submetrics.
- Some reproducibility metadata appears fragile when run artifacts are missing/not materialized (see convergence analysis notes).

