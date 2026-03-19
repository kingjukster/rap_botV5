# Coding Standards
(See docs/CONTEXT.md)

## Rules

- **No hardcoded values** — Use config (evolution.yaml, rapbot.yaml) or env vars (RAPBOT_*)
- **Use config for all parameters** — Paths, weights, evolution params
- **All randomness must be seeded** — Pass --seed; use evo_rhyme.repro.seed_everything
- **Log all major operations** — Generation stats, fitness, mutation usage

---

## Naming

- **snake_case** for functions and variables
- **PascalCase** for classes
- **Descriptive names** — e.g. compute_verse_fitness, score_end_rhyme

---

## Structure

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Entry points (run_verse_qd, run_couplet_evolution, etc.) |
| `evo_rhyme/` | Core logic (fitness, mutation, evolution, archive) |
| `config/` | settings.py, evolution.yaml, rapbot.yaml |
| `data/` | Datasets, templates, runs (runs gitignored) |
| `docs/` | Documentation |
| `tests/` | pytest tests |

---

## Logging

Every run must log:

- **generation** — Gen number
- **fitness stats** — best_fitness, avg_fitness
- **mutation usage** — Which operators fired (optional)
- **archive_coverage** — For QD runs

---

## Config Resolution

1. Load from `config/evolution.yaml` (evolution section) and `config/rapbot.yaml`
2. Merge QD defaults from `config/settings.py` when qd section absent
3. Override via CLI (--population, --generations, etc.)
4. Override via RAPBOT_* env vars

---

## Reproducibility

- Always support --seed for reproducibility
- Write config snapshot to run dir
- Document env vars that affect results
