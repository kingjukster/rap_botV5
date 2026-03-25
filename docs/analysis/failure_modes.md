# Failure Modes Analysis

## Overview
Failure modes were identified from:
- explicit penalties and comments in fitness/evolution code,
- mutation/crossover behavior,
- QD archive mechanics,
- experiment notes in `data/experiments`.

---

## 1) Mode collapse (fitness landscape collapse)

### Symptoms
- Population converges on repeated rhyme families / token shells.
- Best fitness plateaus early while lexical variety decreases.

### Evidence in code
The code explicitly introduces anti-collapse penalties:
- rhyme-family diversity penalty,
- repeated-shell penalty,
- template and repetition penalties,
which indicates prior observed collapse pressure.

### Likely trigger conditions
- high selection pressure + low immigrant diversity,
- weak novelty weighting,
- over-strong rhyme-related positive terms.

---

## 2) Repetitive lyrics / template lock-in

### Symptoms
- Similar line skeletons and repeated end-word patterns.
- “Fluent but formulaic” outputs.

### Evidence
- Dedicated penalties: `template_penalty`, `structural_repetition_penalty`, `cross_verse_repetition_penalty`, `corpus_overlap_penalty`, near-duplicate penalties.
- Plan notes explicitly mention template contamination and mixed-template artifacts.

### Likely trigger conditions
- mutation operators preferring local lexical swaps over semantic restructuring,
- archive niches that separate by style/structure but not sufficiently by semantic novelty.

---

## 3) Low coherence / mixed-topic verses

### Symptoms
- lines individually plausible but globally incoherent,
- abrupt semantic topic jumps inside 4-bar units.

### Evidence
- Plan 2/Plan 5 docs cite contamination examples and weight rebalancing toward coherence.
- Coherence had to be upweighted, suggesting it was underpowered relative to fluency/rhyme at earlier settings.

### Likely trigger conditions
- crossover between semantically distant parents,
- high LM fluency rewards allowing smooth but off-theme rewrites,
- scalarized objective compensation (high rhyme+fluency masking low coherence).

---

## 4) Overfitting to corpus/templates

### Symptoms
- lexical borrowing and phrase reuse from corpus lines,
- reduced originality despite high surface quality.

### Evidence
- explicit `corpus_overlap_penalty`, `cliche_penalty` and anti-template logic.
- score cache and reuse pathways can reinforce known-good text patterns unless novelty pressure stays high.

### Likely trigger conditions
- limited mutation vocabulary diversity,
- strong fitness emphasis on metrics learned/proxied from corpus statistics.

---

## 5) Operational failure modes (evaluation blindness)

### Symptoms
- inability to reliably assess convergence/failure due missing run artifacts.
- experiment comparisons with incomplete metadata.

### Evidence in this checkout
- `score_history.csv` files in local run dirs are LFS pointers (not materialized).
- experiment docs mention missing config snapshots for key runs.
- DB analysis scripts depend on live MySQL + connector.

### Impact
Even when algorithm quality is good, missing telemetry creates research risk: conclusions may rely on anecdotal runs rather than reproducible aggregates.

---

## 6) Failure mode severity ranking

1. **High:** semantic incoherence masked by fluency/rhyme.
2. **High:** mode collapse into repeated shells/rhyme families.
3. **Medium-high:** template/corpus overfitting.
4. **Medium:** niche redundancy across semantically similar outputs.
5. **High (research ops):** missing materialized logs/metadata causing weak reproducibility.

---

## 7) Current built-in mitigations (already present)

- hard constraints and acceptance floors,
- anti-duplicate/template penalties,
- novelty and QD archive pressure,
- niching and multiobjective options,
- adaptive control-policy update with failure penalty,
- stale-run marking and run status management.

These are good foundations; failures are not unaddressed, but mitigation strength appears configuration-sensitive.

