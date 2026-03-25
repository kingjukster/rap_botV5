# Fitness Function Breakdown

## 1) Couplet fitness components

Couplet scoring (`score_couplet` + `compute_fitness`) combines:

### Positive components
- **Rhyme form:** `end_rhyme`, `internal_rhyme`, `rhyme_graph`, `multisyllabic`.
- **Prosody/flow proxies:** `syllable_balance`, `stress_alignment`.
- **Meaning/style quality:** `semantic`, `fluency`, `lexical_validity`, `ngram_fluency`. (`coherence` and `punchline` are computed in `score_couplet` but **not** in default `DEFAULT_WEIGHTS`; they affect fitness only if you add weights via config.)
- **Exploration:** `novelty`.

### Penalties
- **Degeneracy:** `identical_line_penalty`, `near_duplicate_penalty`, `repetition_penalty`.
- **Template overuse:** `template_penalty`, `rhyme_family_repetition_penalty`.
- **Data leakage/overfit:** `corpus_overlap_penalty`.
- **Theme drift:** `theme_penalty`.

### Population-level anti-collapse (added at aggregation)
- rhyme-family diversity penalty.
- repeated-shell penalty (discourages repeated 4-token skeletons across population).

### Hard safeguards
- n-gram floor (default when corpus present) can force fitness to 0 for unnatural phraseing.
- global cap (`FITNESS_CAP`) limits saturation/exploitation.

---

## 2) Verse fitness components (QD objective scalarization)

Verse scoring (`score_verse` + `compute_verse_fitness`) includes:

### Rhyme/structure
- `rhyme_scheme_score`, `internal_rhyme`, `rhyme_chain_density`, global/internal chain scores,
- rhyme graph metrics (`density`, `cluster_coeff`, `chain_length`).

### Language quality
- `fluency`, `lm_fluency`, `lexical_validity`, `coherence`, `semantic`, `punchline`.

### Rhythm/performance
- `syllable_balance`, `flow_alignment`, `flow_continuity_score`, `beat_fit`.

### Creativity/diversity
- `novelty`, plus style/prompt adherence controls.

### Penalties
- repetition/template/duplicate penalties,
- corpus overlap, cliche, garbled-line penalty,
- structural/cross-verse repetition penalties.

---

## 3) Weighting behavior and design intent

### Couplet weights (default)
Couplet defaults strongly weight rhyme + n-gram fluency + semantics, with substantial negative penalties for repetition/template/corpus overlap/theme miss.

Design intent (explicit in comments): avoid reward hacking where valid words + rhyme tokens yield nonsense.

### Verse weights (default)
Verse defaults were rebalanced (Plan 5) to increase:
- `coherence` and `novelty`,
and reduce:
- `lm_fluency`, `fluency`, `flow_continuity_score`.

Interpretation: system authors observed fluency-heavy solutions that sounded smooth but were semantically weaker / template-contaminated, then shifted objective mass toward global quality.

---

## 4) Potential objective conflicts

### Conflict A: rhyme density vs coherence
- High internal rhyme pressure can push lexical substitutions that are phonetic wins but semantic regressions.
- Mitigation present: coherence + semantic + penalties + minimum acceptance floors.

### Conflict B: novelty vs fluency
- Novel constructions may reduce n-gram/LM fluency; conservative settings risk bland outputs.
- Current system balances through weighted novelty plus strong anti-garbled penalties.

### Conflict C: prompt adherence vs originality
- Strong theme/prompt constraints can induce repeated lexical anchors (e.g., overusing key theme words).
- Existing penalties (`theme_word_repetition_penalty`, repetition/corpus overlap) partially mitigate this.

### Conflict D: local line quality vs global verse arc
- Many components are line/local; fewer explicitly enforce narrative progression across 4 or 16 bars.
- `coherence`, `flow_continuity`, and 16-bar block coherence help, but narrative-level rewards remain comparatively indirect.

---

## 5) Comparison to research norms (rap/creative text)

### Rhyme density / technique
The system is strong here:
- explicit internal/end rhyme,
- multisyllabic overlap,
- rhyme-chain and graph topology metrics,
which is richer than many text-EA baselines.

### Coherence
There is explicit coherence scoring and recent weight reinforcement, but coherence is still scalarized against many style/form signals. In research terms, this risks **metric compensation** (high rhyme/fluency masking moderate coherence).

### Stylistic quality
Strengths:
- style genome and prompt genome,
- style adherence metric,
- archive dimensions including tone/narrativity (in style modes).

Limitations:
- style quality is still proxy-scored, with limited direct human-preference grounding in the optimization loop.

---

## 6) Research critique of current scalarization

Current weighted-sum fitness is practical and tunable, but:
- likely to be sensitive to small weight shifts,
- susceptible to local reward hacking,
- difficult to calibrate across themes/styles.

The repository partially addresses this with:
- Pareto mode (for couplets),
- QD archive coverage pressure,
- acceptance floors/constraints,
- post-hoc policy updates from historical outcomes.

Overall assessment: **well-engineered multi-signal scorer with explicit anti-degeneracy design, but still vulnerable to scalarization trade-off artifacts typical in computational creativity systems**.

