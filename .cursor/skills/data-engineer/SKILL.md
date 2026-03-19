---
name: data-engineer
description: >-
  Manages rap_bot logs, runs/, CSV/JSON outputs, and database integration. Use
  when storing runs in DB, building datasets from evolution, improving experiment
  tracking, modifying evo_rhyme/db.py, init_db.py, run artifacts, or manifest.jsonl.
---

# Data Engineer

Specialist for rap_bot data pipelines: run storage, CSV/JSON outputs, and experiment tracking.

## Scope

| Module / Path | Purpose |
|---------------|---------|
| `evo_rhyme/db.py` | MySQL run/experiment/candidate storage, score cache, lineage, artifacts |
| `scripts/init_db.py` | Schema creation, table definitions |
| `data/evo_rhyme/runs/` | Run directories (couplet: `{timestamp}/`, QD: `qd_{timestamp}/`) |
| `data/experiments/` | Experiment YAML configs, manifest.jsonl, report outputs |
| `evo_rhyme/experiment_analysis.py` | Control impact analysis, bootstrap CI, correlations |
| `evo_rhyme/experiment_controls.py` | Experiment arm configs, control snapshots |
| `scripts/archive_artifacts_to_db.py` | Ingest run dir artifacts into DB |

## Per-Run Artifacts (data/evo_rhyme/runs/)

| File | Content |
|------|---------|
| `config.json` | population, generations, theme, archive_mode, seed_info, etc. |
| `score_history.csv` | gen, best_fitness, avg_fitness; QD adds archive_coverage |
| `archive.json` | MAP-Elites archive (QD only) |
| `top_candidates.json` | Best individuals per generation (lines, fitness, scores) |

## Core Invariants

- **Idempotent schema**: Use `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN` with Duplicate column handling.
- **JSON handling**: Parse `config_json`, `scores_json`, `lines_json`, `extra_json` when reading; serialize when writing.
- **Logging**: Log DB failures; return `None` or `-1` on failure instead of raising.

## Storing Runs in DB

1. **Enable DB**: Set `RAPBOT_USE_DB=1` and MySQL env vars (`RAPBOT_DB_HOST`, etc.).
2. **insert_run** → returns `run_id`. Include full `config_json` (control snapshot for experiments).
3. **insert_generation** after each gen: `gen`, `best_fitness`, `avg_fitness`, `diversity`, `acceptance_rate`, `extra_json`.
4. **insert_candidate** for top individuals: `run_id`, `gen`, `candidate_type`, `scheme`, `lines_json`, `fitness`, `scores_json`.
5. **update_run_status** when done: `completed` or `failed` (with `failure_reason` when failed).
6. **Lineage**: `insert_lineage(child_id, parent_id, operation, gen)` for parent-child edges.
7. **Archive cells** (QD): `upsert_archive_cells_batch` for MAP-Elites grid.

```python
# Pattern for run scripts
run_id = db.insert_run(script_name, theme_keywords, config_json, experiment_id=eid, arm_id=aid)
# ... evolution loop ...
for gen, stats in enumerate(generations):
    db.insert_generation(run_id, gen, best_fitness, avg_fitness, diversity, acceptance_rate, extra_json)
db.update_run_status(run_id, "completed")
```

## Building Datasets from Evolution

- **From DB**: `list_candidates(run_id, gen=...)`, `list_top_candidates(run_id)`, `load_archive_cells(run_id)`.
- **From files**: Load `top_candidates.json` or `archive.json`; parse lines, fitness, scores.
- **Experiments**: `list_runs_for_experiment(experiment_id, arm_id)` returns runs with `config_json` (control snapshot).
- **Control impact**: `analyze_control_impact(rows, control_keys)` in experiment_analysis.py — bootstrap CI, Cohen's d, correlations.
- ** manifest.jsonl**: One JSON object per line; each record links run/arm to outcomes for analysis scripts.

```python
# Load top candidates from run dir
with open(run_dir / "top_candidates.json") as f:
    data = json.load(f)
# Structure: list of {lines, fitness, scores, gen, ...}
```

## Experiment Tracking

- **experiments** table: `name`, `description`, `mode`.
- **experiment_arms**: `experiment_id`, `arm_name`, `control_snapshot` (JSON).
- **runs** link: `experiment_id`, `arm_id` when run by control experiment runner.
- **YAML configs** (`data/experiments/*.yaml`): `runner`, `experiment_name`, `treatments`, `fixed_controls`, `n_seeds_per_arm`.
- **Required metadata**: Full config snapshot in `config.json` and `config_json`; seed in `seed_info`; git hash (Plan 1 — not yet).

**Improvements**

- Ensure `write_config()` runs at run start so `config.json` exists even on failure.
- Add git commit hash to config snapshot for reproducibility audits.
- Use `archive_artifacts_to_db` to backfill run dirs into DB when RAPBOT_USE_DB was off.

## DB Schema (init_db.py)

- **runs** — script_name, theme_keywords, config_json, status, experiment_id, arm_id, failure_reason
- **generations** — run_id, gen, best_fitness, avg_fitness, diversity, acceptance_rate, extra_json
- **candidates** — run_id, gen, candidate_type, scheme, lines_json, fitness, scores_json
- **lineage** — child_id, parent_id, operation, gen
- **score_cache** — text_hash, candidate_type, scheme, scores_json
- **archive_cells** — run_id, cell_key, candidate_id, lines_json, fitness, scores_json
- **experiments** / **experiment_arms** — experiment config and arm definitions
- **artifacts** — gzipped blobs (sha256, kind, run_id, run_tag, rel_path)

## Quick References

- `db.db_enabled()` — check if DB is on
- `db.connection()` — context manager for MySQL connections
- `db.execute_readonly_sql(sql)` — safe SELECT-only (webapp SQL page)
- `scripts/list_recent_runs.py`, `scripts/mark_stale_runs.py` — run discovery and cleanup
- `scripts/run_learning_validation.py` — produces manifest.jsonl + reports in data/experiments/
