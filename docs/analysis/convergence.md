# Convergence Analysis (DB + Logs)

## Data availability and constraints
I attempted to use the DB-first pathway (`scripts/analyze_runs.py`, `evo_rhyme.db`) and local run logs.

Observed constraints in this checkout:
- DB queries are not executable in the current environment because `mysql` connector/runtime is unavailable.
- Local run logs under `data/evo_rhyme/runs/*/score_history.csv` are all Git LFS pointer files, not materialized CSV content.

Because of this, convergence statistics are reconstructed from:
1. **System code paths** (`runs`, `generations` schema + analysis scripts).
2. **Materialized experiment notes** in `data/experiments/*.md`.

---

## 1) Runs table / generations table semantics (what the system logs)

The system is designed to compute these metrics from DB tables:
- `runs.status` for completion/failure rate.
- `generations.best_fitness` and `avg_fitness` for progression and trend.
- optional `diversity`, `acceptance_rate`, and `extra_json` for richer dynamics.

This matches the analysis scripts and web analysis service that derive:
- run-level failure rates,
- best-fitness trend over time,
- stagnation length (`runs since last best`).

---

## 2) Reconstructed performance findings from available logs

### 2.1 Best fitness progression (documented run trajectories)
From `data/experiments/generations_tuning.md`:

- `qd_20260318_201102` (60 gens): best fitness reached **0.861** and plateaued from generation 37 through 59.
- `qd_20260318_105357` (80 gens): best fitness reached **0.909**; at generation 60 it was **0.895**, then improved to **0.909** by generation 72.

Interpretation:
- Within-run evidence shows continued (though diminishing) gain past 60 generations.
- Cross-run evidence suggests higher ceilings are reachable with longer schedules and/or favorable stochastic conditions.

### 2.2 Convergence class (improving / flat / declining)
- `qd_20260318_201102`: **flat/plateauing** after mid-run.
- `qd_20260318_105357`: **improving** beyond gen 60.

Overall class: **improving but variance-sensitive**, with frequent plateau windows.

### 2.3 Stagnation length
Using explicit plateau note in `generations_tuning.md`:
- For 60-gen run: stagnation window = generation 37→59 = **~23 generations**.

### 2.4 Marginal gain estimate
From documented values:
- gain 60→80 in best run: `0.909 - 0.895 = +0.014` over 20 generations
- marginal gain: **0.0007 best_fitness/generation** (diminishing returns regime).

### 2.5 Failure rate
A repository-level failure signal appears in config comments:
- `config/evolution.yaml` notes population 100 correlating with **~89% failure** in prior run analysis and recommends lower population for couplet mode.

Because DB rows are unavailable here, this rate cannot be independently recomputed in this checkout.

---

## 3) Summary metrics (available-evidence estimate)

- **Convergence:** mixed; both improving and plateauing behaviors are present.
- **Best-fitness progression:** up to **0.909** documented; lower runs plateau near **0.861**.
- **Stagnation length:** observed plateau of ~**23 generations** in a 60-gen run.
- **Failure rate:** historically high in some configs (reported ~**89%** for pop=100 in comments), not reproducible locally without DB.

---

## 4) Reliability notes

These findings are reliable for architectural interpretation, but **not a full quantitative replication**, due to:
- missing materialized `score_history.csv` artifacts (LFS pointers only), and
- missing DB connector/runtime in the current environment.

For a full research-grade convergence table, run with:
- a live MySQL backend (`RAPBOT_USE_DB=1`), and
- complete run artifacts (pull LFS objects or store plain CSV snapshots).

