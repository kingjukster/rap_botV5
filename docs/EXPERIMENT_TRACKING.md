# Experiment Tracking
(See docs/CONTEXT.md)

## Run Structure

`data/evo_rhyme/runs/` (gitignored)

- **Couplet / verse:** `{timestamp}/` (e.g. `20260319_133526/`)
- **QD:** `qd_{timestamp}/` (e.g. `qd_20260319_133526/`)

### Per-run artifacts

| File | Content |
|------|---------|
| `config.json` | population_size, num_elites, tournament_k, random_immigrants_per_gen, rhyme_scheme, theme_keywords, archive_mode, seed_info (when --seed), etc. |
| `score_history.csv` | gen, best_fitness, avg_fitness; QD adds archive_coverage |
| `archive.json` | MAP-Elites archive (QD only) |
| `top_candidates.json` | Best individuals per generation (lines, fitness, scores) |

---

## Required Metadata

- **config** — Full evolution params (config.json)
- **seed** — In seed_info when --seed passed
- **git commit hash** — Not yet captured (Plan 1 recommendation)
- **timestamp** — In run dir name

---

## DB Schema (RAPBOT_USE_DB=1)

- **runs** — script_name, theme_keywords, config_json, status, created_at
- **generations** — run_id, gen, best_fitness, archive_coverage
- **candidates** — run_id, gen, candidate_type, lines, fitness, scores
- **experiments** — name, description, mode
- **experiment_arms** — experiment_id, arm_name, control_snapshot

---

## Metrics

- **best_fitness** — Max fitness per generation
- **avg_fitness** — Mean fitness per generation
- **archive_coverage** — % of MAP-Elites grid filled (QD)
- **diversity** — Behavioral diversity in archive
- **acceptance rate** — Candidates passing constraints

---

## Naming Convention

`run_<mode>_<timestamp>` or `qd_<timestamp>`

Examples: `qd_20260319_133526`, `20260319_153000`

---

## Experiment Configs (data/experiments/)

YAML files define A/B experiments:

- `runner` — couplet | verse | qd
- `experiment_name` — e.g. ab_static_vs_learned
- `mode` — single_control
- `n_seeds_per_arm` — Replicates per arm
- `fixed_controls` — theme, population, generations, etc.
- `treatments` — arm_name, controls (e.g. policy_mode: static vs learned)

---

## Key Questions

- Did fitness improve?
- Did diversity collapse?
- Which mutations helped?
- Is the run reproducible (same config + seed → same result)?
