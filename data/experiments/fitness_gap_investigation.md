# Fitness Gap Investigation — Plan 1

**Date:** 2026-03-19  
**Context:** Run `qd_20260318_201102` reached best fitness 0.861 vs. 0.903–0.909 in runs `qd_20260318_163843` and `qd_20260318_105357`.  
**Goal:** Identify causes of performance variance.

---

## 1. Run Metadata Summary

### 1.1 Files Present Per Run

| Run ID | config.json | params.json | manifest.jsonl | archive.json | score_history.csv | top_candidates.json |
|--------|-------------|-------------|----------------|--------------|-------------------|---------------------|
| **qd_20260318_201102** | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **qd_20260318_163843** | ❌ Dir not found | — | — | — | — | — |
| **qd_20260318_105357** | ❌ Dir not found | — | — | — | — | — |

**Conclusion:** No config snapshots found for any run. The underperforming run `qd_20260318_201102` has output artifacts but no `config.json`. The higher-fitness runs `qd_20260318_163843` and `qd_20260318_105357` have no run directories in `data/evo_rhyme/runs/` — they may have been run on another machine, stored only in DB, or removed.

### 1.2 Inferred Params for qd_20260318_201102

From `score_history.csv`:

- **Generations:** 60 (gens 0–59)
- **Best fitness:** 0.861 (reached at gen 37, plateaued)
- **Candidates per gen:** ~90–93 (emitter-driven)
- **Archive coverage (final):** ~31.8%
- **Theme:** Inferred from top verses: `crown,empire`

---

## 2. Config Source Comparison

### 2.1 Current Config Sources

- **evolution.yaml** — Used by `load_settings()` as canonical when `RAPBOT_CONFIG` unset.
  - Contains `evolution` section (population 100, generations 30, init mixed, etc.).
  - Contains **no `qd` section** — so `run_verse_qd` does not read QD-specific overrides from evolution.yaml.
  - `proposer` section (model: gpt-4o-mini) is separate; `run_verse_qd` uses `proposer_model` from `qd` defaults, not `proposer.model`.

- **config/settings.py** — Defines `QD_SECTION_DEFAULTS` (used when `evolution.yaml` has no `qd`):
  - population: 100, generations: 100, init: lm, immigrants: 20
  - proposer_model: gpt-4.1-nano
  - archive_mode: compact_style, fast_mode: True
  - emitter_strategy: multi, novelty_weight: 0.3
  - line_pop: 1500, line_gens: 3, line_seeds: 80, line_lm_budget: 15
  - etc.

- **.env** — Only DB-related vars (RAPBOT_USE_DB, RAPBOT_DB_HOST, etc.). No evolution or QD overrides.

### 2.2 Evolution vs. QD Sections

| Source | Section | Used by |
|--------|---------|---------|
| evolution.yaml `evolution` | population, generations, elites, init, immigrants | `run_couplet_evolution` |
| evolution.yaml `qd` | (none — not defined) | — |
| settings.py `QD_SECTION_DEFAULTS` | All QD params | `run_verse_qd` when no `qd` in YAML |

---

## 3. Param Flow for run_verse_qd

1. **Config loading:** `parse_args()` → `get_qd_defaults(config_path)` → `load_settings()` → `_resolve_section("qd", QD_SECTION_DEFAULTS, cfg_data)`.
2. **evolution.yaml:** Has no `qd` key, so `cfg_data.get("qd")` is empty; merged result = `QD_SECTION_DEFAULTS` unchanged.
3. **CLI overrides:** All `--population`, `--generations`, `--seed`, `--theme`, etc. override the merged defaults.
4. **Policy overrides:** If `policy_mode in ("learned", "explore_mix")`, controls from `learned_policy_path` can override args (except protected keys).
5. **Coverage target:** If `--coverage-target >= 0.5` and archive_mode != ultra_compact, population/generations can be auto-tuned (e.g., 100/150).

**Seed handling:**

- If `--seed` is passed, `seed_everything(seed)` seeds Python/NumPy/Torch.
- Without `--seed`, runs are non-deterministic (LM API, random ops).

---

## 4. Likely Causes of the Fitness Gap

| Factor | Evidence | Impact |
|--------|----------|--------|
| **Generations** | qd_20260318_201102 used 60 gens; PLANS.md states best run (qd_20260318_105357) used 80 gens | Lower fitness ceiling with fewer generations |
| **No seed** | No `config.json`/`seed_info` in run dir; likely no `--seed` used | High run-to-run variance |
| **LM non-determinism** | Proposer uses gpt-4.1-nano/gpt-4o-mini via API; temperature > 0 | Different samples each run |
| **Config not persisted** | `VerseQDRunLogger.write_config()` writes `config.json` only when `config.output_dir` is set; run dir exists but config.json missing | Either older code didn’t write it, or it was removed; can’t confirm exact params |

---

## 5. Recommended Golden Config

To reduce variance and target best_fitness ≥ 0.90:

```yaml
# Add to config/evolution.yaml (new qd section) or use CLI
qd:
  population: 100
  generations: 80   # or 100 — Plan 4 suggests 80 was used by best run
  elites: 5
  immigrants: 20
  init: lm
  lm_budget: 20
  archive_mode: compact_style
  emitter_strategy: multi
  # ... (rest from QD_SECTION_DEFAULTS as needed)
```

**CLI recommendation for reproducibility:**

```bash
python scripts/run_verse_qd.py \
  --theme "crown,empire" \
  --generations 80 \
  --seed 42 \
  --runs-dir
```

**Follow-up:**

1. Run 3–5 replicates with fixed seed (e.g., 42) and identical config; report mean and std of best_fitness at gen 59.
2. Add `qd` section to `evolution.yaml` with explicit generations/population for consistency.
3. Ensure `write_config()` always runs when `--runs-dir` is used, and that `config.json` is retained for future audits.

---

## 6. Config Diff (Cannot Produce)

Because no config snapshots exist for any of the three runs, a config diff between high- vs. low-fitness runs cannot be produced. The investigation relied on:

- Inferred params from `score_history.csv` for qd_20260318_201102.
- PLANS.md statement that qd_20260318_105357 used 80 generations.
- Current config files and param flow in `run_verse_qd.py`.

---

## 7. Action Items

1. **Add config snapshot to every run:** Verify `VerseQDRunLogger.write_config()` is invoked when `output_dir` is set; consider writing `config.json` at run start even if run later fails.
2. **Add `qd` section to evolution.yaml:** Override `generations: 80` (or 100) so default matches best-performing runs.
3. **Reproducibility audit:** Run 3–5 replicates with `--seed 42 --generations 80`; document mean ± std.
4. **DB check:** If runs used `--db`, query MySQL `runs` table for runs matching theme "crown,empire" or run_dir patterns; control_snapshot may contain params for qd_20260318_163843 and qd_20260318_105357.
