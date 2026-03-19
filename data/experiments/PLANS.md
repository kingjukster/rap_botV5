# Improvement Plans — Five Focus Areas

Based on analysis of recent QD runs (Mar 18–19, 2026) and learning validation experiments.

---

## Plan 1: Investigate Fitness Gap vs. Previous Runs

**Problem:** `qd_20260318_201102` reached best fitness 0.861 vs. 0.903–0.909 in other recent runs.

**Goal:** Identify reproducible causes of performance variance (config, seed, environment).

### Steps

1. **Extract run metadata**
   - Check if each run directory has `config.json`, `params.json`, or `manifest.jsonl` with config snapshot.
   - Compare: population, elites, immigrants, init strategy, policy_mode, seed, generations.

2. **Compare evolution config**
   - Diff `config/evolution.yaml` and `config/settings.py` against git history.
   - Check for env overrides (`.env`, CLI args) that might differ between runs.

3. **Reproducibility audit**
   - Run `run_verse_qd.py` or `run_couplet_evolution.py` with identical config + seed.
   - Run 3–5 replicates; compute mean and std of best_fitness at gen 59.
   - If variance is high, document seed sensitivity.

4. **Document findings**
   - Create `data/experiments/fitness_gap_investigation.md` with:
     - Config diff between high- vs. low-fitness runs.
     - Recommended “golden” config for consistency.
     - Seed or config changes that explain the gap (if any).

### Files to touch

- `scripts/run_verse_qd.py` (or equivalent entrypoint)
- `config/evolution.yaml`, `config/settings.py`
- New: `data/experiments/fitness_gap_investigation.md`

### Success criteria

- Clear explanation of why `qd_20260318_201102` underperformed.
- At least one config/seed that reliably reaches best_fitness ≥ 0.90.

---

## Plan 2: Address Template Contamination

**Problem:** High-fitness verses mix unrelated lines (e.g., “empire/crown” + “shot in the leg”), suggesting crossover or mutation pulling from incompatible templates.

**Goal:** Reduce incoherent line mixing while preserving valid diversity.

### Steps

1. **Characterize contamination**
   - Sample 50–100 verses from archive with fitness > 0.80.
   - Label: (a) coherent, (b) mixed-topic, (c) garbled.
   - Identify recurring “orphan” line patterns and their source (template, corpus, LM).

2. **Strengthen semantic coherence**
   - Add or increase weight for `coherence` in fitness (if not already dominant).
   - Add cross-line theme consistency: penalize verses where lines come from different theme clusters (embeddings or keyword overlap).
   - Optional: require at least N lines to share a theme (e.g., via embedding cosine > threshold).

3. **Constrain crossover**
   - Restrict crossover to lines from same theme/slot.
   - Or: add post-crossover “coherence check”; reject or repair if score drops below threshold.
   - Consider “semantic niching”: only crossover within same behavioral niche.

4. **Constrain mutation sources**
   - Ensure mutation/rewriting pulls from theme-consistent corpus or LM prompts.
   - Add `theme_penalty` or `template_penalty` for lines that don’t match the verse theme.

5. **Validate**
   - Re-run QD; manually inspect top 20 verses for contamination.
   - Measure % of top-N verses that pass a simple coherence check (e.g., human-rated or embedding-based).

### Files to touch

- `evo_rhyme/fitness.py` (coherence weight, new penalties)
- `evo_rhyme/mutation.py` or crossover logic
- `evo_rhyme/verse_evolution.py` (crossover rules)
- `config/evolution.yaml` (new penalty weights)
- New: `scripts/audit_template_contamination.py` (optional analysis script)

### Success criteria

- Top 20 verses have < 10% clear template contamination.
- Coherence subscore improves in top candidates without collapsing diversity.

---

## Plan 3: Fix Learning Validation Experiment Metrics

**Problem:** `learning_validation_*` runs report avg_fitness = 0.0 for all 4 runs; policy comparisons are unusable.

**Goal:** Ensure fitness and related metrics are correctly recorded for learning validation experiments.

### Steps

1. **Trace data flow**
   - Map: `run_learning_validation.py` → experiment runner → DB/log writer.
   - Identify where `fitness` (and `fitness_vector`) should be written.
   - Check schema: which table/columns store run outcomes.

2. **Locate the bug**
   - Add logging at write points (DB insert, manifest append).
   - Run a single learning-validation run; confirm:
     - Fitness is computed (print in evolution loop).
     - Fitness is passed to the experiment recorder.
     - Recorder writes to DB/manifest correctly.
   - Common issues: wrong column name, wrong experiment_id, aggregation over empty list.

3. **Fix recording**
   - Update recorder to persist `fitness`, `best_fitness`, `fitness_vector` (rhyme, flow, semantic, novelty, punchline).
   - Ensure `analyze_control_impact` reads from the correct source (DB vs. manifest).

4. **Backfill if needed**
   - If old runs have logs but not DB rows, write a one-off script to parse logs and insert outcomes.
   - Skip if data is unrecoverable; focus on future runs.

5. **Validate**
   - Run 2 static + 2 learned policy runs.
   - Confirm sanity_report shows non-zero fitness and sensible policy comparison.
   - Re-run `analyze_control_impact` and verify output.

### Files to touch

- `scripts/run_learning_validation.py`
- `evo_rhyme/experiment_controls.py` or equivalent experiment runner
- `evo_rhyme/experiment_metrics.py` (if separate)
- `evo_rhyme/db.py` (schema, insert logic)
- `scripts/analyze_control_impact.py` (input source)

### Success criteria

- Learning validation runs write non-zero fitness to DB/manifest.
- Sanity report shows meaningful avg_fitness and policy deltas.
- At least 5 runs per policy for a low-confidence but usable comparison.

---

## Plan 4: Run Longer Generations

**Problem:** Best-performing run (`qd_20260318_105357`) used 80 generations; default is often 60. Recent run plateaued after ~gen 35.

**Goal:** Systematically test longer runs and tune generation budget.

### Steps

1. **Baseline current config**
   - Record: generations, wall time per gen, total time, best_fitness curve.
   - Compute “fitness per minute” or “fitness per API call” if applicable.

2. **Define experiment**
   - Levels: generations = [60, 80, 100, 120].
   - Fixed: population, elites, init, theme, seed (or small seed set).
   - Metrics: best_fitness, mean_fitness, archive_coverage, wall_time.

3. **Execute runs**
   - Run 2–3 replicates per level (or 1 if compute-limited).
   - Save score_history.csv and top_candidates.json for each.

4. **Analyze**
   - Plot best_fitness vs. generation for each run.
   - Compute marginal gain: (fitness_at_N − fitness_at_60) / (N − 60) for N = 80, 100, 120.
   - Identify diminishing returns; recommend optimal N.

5. **Update config**
   - Set `evolution.generations` in `evolution.yaml` (or experiment config) to chosen value.
   - Document rationale in config comment or `data/experiments/generations_tuning.md`.

### Files to touch

- `config/evolution.yaml` (generations)
- `scripts/run_verse_qd.py` or experiment runner (if generations come from experiment config)
- New: `data/experiments/generations_tuning.md`
- Optional: `scripts/run_generations_sweep.py`

### Success criteria

- Clear recommendation for default generations (e.g., 80 or 100).
- Documentation of marginal gains and compute cost tradeoff.

---

## Plan 5: Rebalance Fitness Weights (LM Fluency vs. Coherence / Novelty)

**Problem:** Top verses have strong fluency and style but weaker coherence and novelty; fitness may overweight local fluency vs. global quality.

**Goal:** Adjust weights so high-fitness verses are coherent and novel, not just fluent.

### Steps

1. **Audit current weights**
   - List all terms in `fitness_weights` (evolution.yaml or fitness.py).
   - Map each to: fluency-like (lm_fluency, ngram_fluency, fluency), coherence-like, novelty-like, rhyme-like, penalty-like.
   - Compute effective contribution of each group to total fitness for 5 top vs. 5 bottom verses.

2. **Identify imbalance**
   - If fluency terms dominate (e.g., > 40% of positive contribution), consider reducing.
   - If coherence/novelty are very low (e.g., < 5%), consider increasing.
   - Check for redundant penalties (e.g., theme_penalty vs. coherence).

3. **Propose new weights**
   - Option A: Increase `coherence` and `novelty` by 0.02–0.05 each; decrease `ngram_fluency` or `lm_fluency` by same total.
   - Option B: Add a `cross_line_coherence` term if missing.
   - Option C: Increase `theme_penalty` for off-theme lines.
   - Document rationale and expected effect.

4. **Validate with ablation**
   - Run 3 configs: (a) current, (b) +coherence/novelty, (c) −fluency.
   - Compare top 20 verses: coherence (manual or proxy), novelty (embedding distance), fluency.
   - Ensure fluency doesn’t collapse (e.g., garbled lines increase).

5. **Iterate**
   - If coherence improves but fluency drops too much, try smaller delta.
   - If novelty improves but quality drops, add a minimum fluency threshold or floor.

6. **Commit**
   - Update `config/evolution.yaml` and/or `evo_rhyme/fitness.py`.
   - Add comment referencing this plan and date.

### Files to touch

- `config/evolution.yaml` (fitness_weights)
- `evo_rhyme/fitness.py` (if weights live there, or to add new terms)
- New: `data/experiments/weight_rebalance_log.md`

### Success criteria

- Top verses show improved coherence and novelty (by audit or metric).
- Fluency remains above an acceptable threshold (no increase in garbled lines).
- Clear before/after weight table and rationale in documentation.

---

## Summary

| Plan | Focus            | Main risk                       | Effort (est.) |
|------|------------------|----------------------------------|---------------|
| 1    | Fitness gap      | No clear config difference      | 2–4 hrs       |
| 2    | Contamination    | Over-constraining diversity     | 4–8 hrs       |
| 3    | Validation metrics | DB schema mismatch             | 2–4 hrs       |
| 4    | Longer runs      | Compute cost                    | 2–6 hrs       |
| 5    | Weight rebalance | Degrading fluency               | 4–6 hrs       |

Plans can be run in parallel except: Plan 3 should precede meaningful policy experiments; Plan 5 may interact with Plan 2 (both affect coherence/theme handling).
