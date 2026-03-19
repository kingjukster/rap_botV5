# SYSTEM CONTEXT (READ FIRST)
(See docs/PROJECT_LOG.md for consolidated source of truth)

## What This Project Is

rap_botV5 is:

> An evolutionary optimization system for structured creative text (rap lyrics)

It treats lyric generation as a **search problem over creative space**, optimizing:

- **Rhyme quality** — end rhyme, internal rhyme, rhyme graph, multisyllabic overlap
- **Fluency** — ngram fluency, lexical validity, LM fluency (optional)
- **Semantic coherence** — theme relevance, semantic similarity
- **Novelty** — distance from corpus
- **Structure** — syllable balance, stress alignment, line length

Evolution modes: **couplet** (2-line), **verse** (4/8/16-line), **Quality-Diversity (QD)** with MAP-Elites over behavioral niches (rhyme density, intensity, syllable tightness).

---

## Key Insight

This system is not:

"a text generator"

It is:

> a search process over structured creative space

---

## Core Rules

- All behavior must be config-driven (`config/evolution.yaml`, `config/rapbot.yaml`)
- No hardcoding — use env vars and config files
- Every run must be reproducible (`--seed` + config snapshot + git commit hash)
- All scoring must be modular (`evo_rhyme/scoring/`, fitness weights in config)
- All experiments must be logged (run dir, DB when `RAPBOT_USE_DB=1`)

---

## System Pipeline

Initialize → Score → Select → Mutate/Crossover → Repeat

(For QD: also maintain MAP-Elites archive; optionally two-tier line evolution + verse assembly.)

---

## Configuration Resolution Order

1. Defaults in code and `config/evolution.yaml`
2. Config file — path from CLI `--config`, or `RAPBOT_CONFIG` env
3. Environment variables — `RAPBOT_*` overrides (e.g. `RAPBOT_ELITE_CORPUS`, `RAPBOT_RHYME_CSV`)

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RAPBOT_CONFIG` | Path to main config (YAML/JSON) |
| `RAPBOT_HOME` | Project root for paths |
| `RAPBOT_ELITE_CORPUS` | Override elite corpus path |
| `RAPBOT_RHYME_CSV` | Override rhyme groups CSV |
| `RAPBOT_SIAMESE_DIR` | Override Siamese model dir |
| `RAPBOT_USE_DB` | Enable MySQL run logging (0/1) |
| `RAPBOT_DB_HOST`, `RAPBOT_DB_*` | DB connection |
| `OPENAI_API_KEY` | Required for LM-backed proposer/rewriter |

---

## Key Paths

| Path | Role |
|------|------|
| `config/evolution.yaml` | Evolution defaults (paths, fitness_weights, mutation_weights, proposer) |
| `config/rapbot.yaml` | Full app config (base model, stage3, critic, stats) |
| `config/settings.py` | `load_settings()`, `get_qd_defaults()` |
| `data/rhymes_grouped.csv` | Rhyme groups for mutation/scoring |
| `data/elite_kaggle_corpus_clean.txt` | Elite corpus lines |
| `data/evo_rhyme/templates.txt` | Line templates |
| `data/evo_rhyme/runs/` | Run outputs (gitignored) |
| `data/experiments/` | Experiment configs, plans, reports |

---

## Reference Docs

- Architecture → docs/ARCHITECTURE.md
- Data Flow → docs/DATA_FLOW.md
- Evolution → docs/EVOLUTION_STRATEGY.md
- Experiments → docs/EXPERIMENTS.md
- Tracking → docs/EXPERIMENT_TRACKING.md
- Standards → docs/CODING_STANDARDS.md
- Research → docs/RESEARCH_LOG.md
- Roadmap → docs/TODO_ROADMAP.md
- Paper → docs/RESEARCH_PAPER_OUTLINE.md

---

## Related Operational Docs

- docs/improvement_plan.md — Rhyme planner, corpus, critic, LoRA
- docs/repo_cleanup_unused_files.md — Repo hygiene
- data/experiments/PLANS.md — Five improvement plans (fitness gap, contamination, validation metrics, generations, weight rebalance)
