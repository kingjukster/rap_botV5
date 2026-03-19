# Experiments
(See docs/CONTEXT.md)

## Metrics

### Automatic

- acceptance rate
- mean fitness / best fitness
- rhyme density (internal_rhyme)
- diversity (archive coverage)
- duplicate rate
- theme similarity (semantic)
- syllable deviation

### Human

- quality preference
- coherence
- rhyme strength
- originality

---

## Active Improvement Plans (data/experiments/PLANS.md)

| Plan | Focus | Goal |
|------|-------|------|
| 1 | Fitness gap | Reproducibility audit; identify config/seed variance |
| 2 | Template contamination | Reduce incoherent line mixing |
| 3 | Learning validation metrics | Fix avg_fitness = 0.0 bug in DB/manifest |
| 4 | Longer generations | Test 60/80/100/120; recommend default |
| 5 | Weight rebalance | Increase coherence/novelty vs fluency |

---

## Experiment 0: Baseline

- Current system, fixed seeds
- Establish reproducible baseline
- Outputs: metrics, logs, samples

---

## Experiment 1: Rhyme Planner

- Improve rhyme grouping + diversity
- Conditions: current, normalized, confidence-aware, full planner
- Metrics: rhyme diversity, repetition rate, acceptance

---

## Experiment 2: Dedup / Filtering

- Reduce duplicates and repetition
- Conditions: baseline, dedup only, repetition penalty, full filtering
- Metrics: duplicates, lexical diversity

---

## Experiment 3: Corpus

- Better data vs more data
- Conditions: current corpus, expanded, weighted
- Metrics: rhyme quality, coherence, acceptance

---

## Experiment 4: Critic

- Local vs external scoring
- Conditions: external only, local only, calibrated local, hybrid
- Metrics: correlation, ranking agreement, runtime

---

## Experiment 5: LoRA

- Staged adaptation
- Conditions: current model, phase A only, phase B only, A→B
- Metrics: quality, diversity, human preference

---

## Experiment 6: Scoring Ablation

- Importance of each fitness component
- Conditions: full, remove each component individually
- Metrics: quality drop, human preference

---

## Experiment 7: Decoding

- Temperature, top_p, repetition penalty
- Metrics: diversity, acceptance, runtime

---

## Experiment 8: Final Comparison

- Baseline vs improved pipeline vs full optimized system
- Metrics: all metrics, human ranking

---

## Run Order

1. Baseline
2. Rhyme planner
3. Filtering
4. Corpus
5. Critic
6. Ablation
7. LoRA training
8. Decoding
9. Final comparison

---

## Reproducibility Rules

- Fixed seed sets
- Log config per run
- Log git commit hash (planned)
- Separate dev vs eval seeds
- Store outputs per run
