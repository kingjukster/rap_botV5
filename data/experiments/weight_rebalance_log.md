# Weight Rebalance Log — Plan 5 (2026-03-19)

**Goal:** Rebalance fitness weights so high-fitness verses are coherent and novel, not just fluent.  
**Context:** Top verses from `qd_20260318_201102` have strong fluency/style but weaker coherence/novelty; template contamination (e.g., "empire/crown" + "shot in the leg") appears in high-fitness candidates.

---

## 1. Audit: Current Weights (VERSE_DEFAULT_WEIGHTS)

Source: `evo_rhyme/fitness.py` — used by `run_verse_qd.py` for 4-line verse evolution.

### Category Mapping

| Category       | Terms | Weight sum | Notes |
|----------------|-------|------------|-------|
| **Fluency-like** | fluency, lm_fluency, flow_alignment, flow_continuity_score, syllable_balance, lexical_validity | 0.49 | LM perplexity + syllable/flow heuristics |
| **Coherence-like** | coherence, semantic, punchline | 0.38 | Cross-line semantic similarity, theme relevance, punchline |
| **Novelty-like** | novelty | 0.25 | Inverse of repetition |
| **Rhyme-like** | rhyme_scheme_score, internal_rhyme, rhyme_chain_density, global_rhyme_chain_score, internal_chain_score, rhyme_graph_* | 0.57 | Rhyme quality |
| **Penalty-like** | identical_line_penalty, template_penalty, repetition_penalty, near_duplicate_penalty, filler_line_penalty, line_phrase_penalty, corpus_overlap_penalty, garbled_line_penalty, cliche_penalty, structural_repetition_penalty, cross_verse_repetition_penalty | negative | Various penalties |

### Full Positive Weights (before)

```yaml
rhyme_scheme_score: 0.15
internal_rhyme: 0.07
rhyme_chain_density: 0.08
global_rhyme_chain_score: 0.08
internal_chain_score: 0.06
rhyme_graph_density: 0.05
rhyme_graph_cluster_coeff: 0.04
rhyme_graph_chain_length: 0.04
syllable_balance: 0.05
fluency: 0.08
lm_fluency: 0.12
semantic: 0.08
lexical_validity: 0.06
coherence: 0.22
punchline: 0.08
novelty: 0.25
flow_alignment: 0.10
flow_continuity_score: 0.08
style_adherence: 0.06
prompt_adherence: 0.05
beat_fit: 0.05
```

---

## 2. Imbalance Analysis (top_candidates.json)

Effective contribution of each group to **positive** fitness (excl. penalties):

| Group    | Top 5 avg | Bottom 5 avg | Notes |
|----------|-----------|--------------|-------|
| Fluency  | **31%**   | **37%**      | Near-dominant; bot5 up to 40% |
| Coherence| 23%       | 24%          | Moderate; similar across top/bot |
| Novelty  | **9%**    | **3%**       | Very low; many verses novelty ≈ 0 |
| Rhyme    | 28%       | 27%          | Healthy |

**Findings:**
- Fluency dominates positive contribution (31–40%), especially in contaminated verses.
- Novelty is weak (3–9%); contaminated verses often have novelty ≈ 0.
- Coherence is adequate but not sufficient to suppress mixed-topic verses.
- No explicit `cross_line_coherence` term — `score_coherence()` in `evo_rhyme/scoring/coherence.py` already computes cross-line semantic similarity (consecutive similarity, min line–centroid); the weight just needs to be higher.

---

## 3. Proposed New Weights

Per Plan 5: increase coherence and novelty by ~0.02–0.05 each; decrease fluency terms by same total.

| Term | Before | After | Delta |
|------|--------|-------|-------|
| coherence | 0.22 | **0.26** | +0.04 |
| novelty | 0.25 | **0.30** | +0.05 |
| lm_fluency | 0.12 | **0.08** | -0.04 |
| fluency | 0.08 | **0.06** | -0.02 |
| flow_continuity_score | 0.08 | **0.05** | -0.03 |

**Net:** +0.09 coherence/novelty, -0.09 fluency-like. Fluency remains strong enough (fluency + lm_fluency + flow_* = 0.27) to avoid garbled lines; `garbled_line_penalty` (-0.35) still protects.

**Rationale:**
- Boost coherence to favor verses where lines are semantically aligned (reduces template contamination).
- Boost novelty to favor diverse, non-repetitive content.
- Reduce lm_fluency and fluency so that fluent-but-incoherent verses are less favored.
- Reduce flow_continuity_score slightly to avoid over-weighting syllable/flow at the expense of meaning.

---

## 4. Applied Changes (2026-03-19)

- `evo_rhyme/fitness.py`: `VERSE_DEFAULT_WEIGHTS` updated as above.
- `config/evolution.yaml`: `fitness_weights` apply to **couplet** evolution (DEFAULT_WEIGHTS); Plan 5 targets **verse** evolution (VERSE_DEFAULT_WEIGHTS). No evolution.yaml changes for verse. Couplet weights unchanged per Plan 5 scope.

---

## 5. Validation

- Short test: 2 generations of `run_verse_qd.py` with theme "crown,empire" to confirm no immediate regression.
- Full validation: Re-run QD; manually inspect top 20 for improved coherence/novelty and no fluency collapse.
