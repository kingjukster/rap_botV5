# Rap Bot V5 — Evolutionary Rap Verse Generation

Evolutionary rap verse generation with explicit rhyme, fluency, and penalty scoring. The system evolves **couplets** (2-line) and **4-line verses** via tournament selection, crossover, and mutation, optimizing for end rhyme quality, internal rhyme, fluency, semantic relevance, and novelty while penalizing repetition and nonsense. Optional **Quality-Diversity (MAP-Elites)** evolution explores a grid of behavioral niches (rhyme density, intensity, syllable tightness) to produce diverse, high-quality verses. A **self-hosted web dashboard** provides run monitoring, evolution triggering, SQL exploration, and research-grade analysis.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Web Dashboard](#web-dashboard)
- [Main Entrypoints](#main-entrypoints)
- [Data & Artifacts](#data--artifacts)
- [Architecture](#architecture)
- [Fitness & Scoring](#fitness--scoring)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Scripts Reference](#scripts-reference)
- [License & Citation](#license--citation)

---

## Features

- **Couplet evolution** — Evolve 2-line couplets with theme keywords, optional multi-objective (Pareto) or niching.
- **4-line verse evolution** — AABB / ABAB (and more) rhyme schemes with phrase-level crossover and mutation.
- **Quality-Diversity (QD)** — MAP-Elites archive over behavioral dimensions (rhyme density, intensity, syllable tightness, theme breadth); optional two-tier line + verse evolution and LM-backed proposers.
- **Hierarchical evolution** — Two-stage pipeline: evolve couplets → build 4-bar verses from couplet archive with structural mutations (swap couplets, rewrite transitions).
- **Structural mutations** — Verse-level operators: couplet swap, transition rewrite, block reordering for richer exploration.
- **Fitness components** — End rhyme, internal rhyme, rhyme graph, multisyllabic overlap, syllable balance, stress alignment, semantic/theme, fluency, ngram fluency, novelty; penalties for repetition, weak tails, corpus overlap, cliché.
- **Population init** — Mixed (template + corpus + random), random, template-only, or **LM** (OpenAI/local) for seeding.
- **Weight tuning** — Outer-loop meta-optimization of fitness weights (couplet or verse) over benchmark prompts.
- **Continuous runs** — Config-driven loop over YAML configs with learned policy, elite replay, and parallel execution.
- **Run logging** — Optional per-run directories with config, score history, archive, and top candidates; MySQL-backed when enabled.

---

## Installation

**Requirements:** Python 3.9+ (recommended 3.10+).

```bash
cd rap_botV5
pip install -r requirements.txt
```

### Key Dependencies

| Category | Packages |
|----------|----------|
| **Deep learning** | `torch`, `torchvision`, `torchaudio`, `transformers`, `accelerate`, `peft`, `sentence-transformers` |
| **NLP / phonetics** | `pronouncing`, `g2p-en`, `nltk`, `regex` |
| **Evolution / ML** | `numpy`, `pandas`, `scikit-learn`, `scipy`, `networkx` |
| **Utilities** | `tqdm`, `matplotlib`, `gensim`, `fasttext`, `python-dotenv` |
| **Web** | `fastapi`, `uvicorn[standard]`, `jinja2` |
| **Optional (LM)** | `openai` (for `BarProposer` / `BarRewriter`) |

The project uses **`pyproject.toml`** for metadata and dependencies. Install in development mode with `pip install -e .` (or `pip install -e ".[full]"` for optional ML/plotting deps). Run scripts from the project root with `python scripts/...` so that `config` and `evo_rhyme` are on `PYTHONPATH`.

---

## Configuration

Configuration is resolved in this order:

1. **Defaults** in code and in YAML (see below).
2. **Config file** — path from CLI, or `RAPBOT_CONFIG` env var, or default `config/rapbot.yaml` / evolution defaults in `config/evolution.yaml`.
3. **Environment variables** — `RAPBOT_*` overrides (e.g. `RAPBOT_ELITE_CORPUS`, `RAPBOT_RHYME_CSV`).

### Config Files

- **`config/evolution.yaml`** — Canonical evolution defaults:
  - **paths:** `rhyme_groups_csv`, `siamese_model_dir`, `elite_corpus`
  - **evolution:** `population_size`, `generations`, `num_elites`, `tournament_k`, `random_immigrants_per_gen`, `population_init`
  - **fitness_weights** — end_rhyme, internal_rhyme, rhyme_graph, multisyllabic, syllable_balance, stress_alignment, semantic, fluency, ngram_fluency, novelty, and penalty weights (repetition, theme_penalty, etc.)
  - **mutation_weights** — end_word_swap, internal_rhyme_insert, stressed_vowel_swap, semantic_swap, etc.
  - **output:** `runs_dir`
  - **proposer:** LM bar proposer (backend, model, temperature, etc.)

- **`config/rapbot.yaml`** — Full Rap Bot settings (base model, paths, generation, stage3, critic, stats). Used by `config/settings.py` for `load_settings()`.

Override any path or evolution parameter via CLI flags or env vars.

---

## Web Dashboard

A self-hosted FastAPI web UI to browse runs, trigger evolution, view candidates, explore the MAP-Elites archive, run SQL queries, and analyze evolution trends.

### Quick Start

```bash
# Install web deps (or use full requirements.txt)
pip install fastapi "uvicorn[standard]" jinja2

# With MySQL and RAPBOT_USE_DB=1 for full functionality
uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. After `init_db.py`, run `python scripts/import_song_catalog.py` to populate the song catalog for the **Evolve** page (artist/song selectors and song-seeded evolution). Use `--replace` to reload from CSV after corpus rebuilds.

### Dashboard Pages

| Route | Description |
|-------|-------------|
| **/** | **Dashboard** — List evolution runs with status (running/completed/failed), last best fitness, and generation count. Filter by status, mark stale runs as failed. |
| **/evolve** | **Evolve** — Start a QD verse evolution job from the web. Configure theme keywords, population, generations, rhyme scheme, init mode. Optionally seed from real songs (artist/song selectors). Job runs in background; redirects to run detail. |
| **/runs/{run_id}** | **Run detail** — Summary, config, top candidates, generations table, links to archive, lineage, seeds. |
| **/runs/{run_id}/generations** | **Generations** — Per-generation score chart and table (best_fitness, avg_fitness, diversity, acceptance_rate). |
| **/runs/{run_id}/archive** | **Archive** — MAP-Elites archive explorer for QD runs; cells with behavioral dimensions and best candidates per cell. |
| **/runs/{run_id}/lineage** | **Lineage** — Parent-child edges between candidates; filter by generation. |
| **/runs/{run_id}/candidates/{candidate_id}** | **Candidate detail** — Full candidate (lines, fitness, scores) and lineage. |
| **/runs/{run_id}/seeds** | **Seeds** — Seed bank entries used for generation 0. |
| **/analysis** | **Analysis** — Evolution analytics: run stats (total, running, completed, failed), best fitness, stagnation count, config dominance (top arms by count/avg fitness), fitness trend chart (last 100 runs). Requires DB. |
| **/sql** | **SQL** — Run read-only `SELECT` queries against the rap bot database. Server-side execution; max rows 1–5000. |
| **/score-cache** | **Score cache** — Recent score_cache keys for debugging. |
| **/about** | **About** — Help and project info. |

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/health/db` | DB connectivity (503 if unavailable) |
| GET | `/api/runs` | List runs (limit, offset, status filter) |
| GET | `/api/runs/{run_id}` | Run detail |
| GET | `/api/runs/{run_id}/generations` | Generation history |
| GET | `/api/runs/{run_id}/candidates` | Candidates (optional gen filter) |
| GET | `/api/runs/{run_id}/top-candidates` | Top candidates by fitness |
| GET | `/api/runs/{run_id}/archive` | MAP-Elites archive cells |
| GET | `/api/runs/{run_id}/lineage` | Lineage edges |
| GET | `/api/runs/{run_id}/seeds` | Seed bank |
| GET | `/api/runs/{run_id}/progress` | Run progress (status, current/total generations) |
| GET | `/api/candidates/{candidate_id}` | Candidate detail |
| GET | `/api/candidates/{candidate_id}/lineage` | Candidate lineage |
| POST | `/api/evolution/start` | Start evolution (theme, population, generations, scheme, init_mode, seed_songs) |
| GET | `/api/evolution/artists` | List artists for song seeding |
| GET | `/api/evolution/songs` | List songs for artist |
| GET | `/api/analysis` | Evolution analysis data |
| POST | `/api/sql` | Execute read-only SQL |
| POST | `/api/runs/mark-stale` | Mark stale runs as failed |
| GET | `/api/score-cache/recent` | Recent score cache keys |
| GET | `/api/experiments` | List control experiments |
| GET | `/api/experiments/{id}` | Experiment detail |
| GET | `/api/experiments/{id}/arms` | Experiment arms |
| GET | `/api/experiments/{id}/runs` | Experiment runs |
| GET | `/api/experiments/{id}/impact` | Control impact report |

### Database Setup

The dashboard stores runs, generations, candidates, lineage, score cache, experiments, and song catalog in MySQL.

```bash
# Start MySQL (Docker)
docker compose up -d mysql

# Init schema (creates runs, generations, candidates, lineage, score_cache, experiments, song_catalog, etc.)
python scripts/init_db.py

# Import song catalog for /evolve artist/song selectors
python scripts/import_song_catalog.py
```

Set `RAPBOT_USE_DB=1` and DB credentials in `.env` (see [Environment Variables](#environment-variables)).

### Docker

**Full stack (web + MySQL):** `docker compose up` starts MySQL and the web dashboard. A public tunnel URL is printed in the logs (localtunnel or ngrok). For a **stable URL** that works from any network, add `NGROK_AUTHTOKEN` to `.env` (free at [ngrok.com](https://ngrok.com)); the same URL persists across restarts.

**Evolution from host:** Evolution scripts (`run_verse_qd.py`, `run_verse_evolution.py`, etc.) are typically run on the **host** with the same DB env (`RAPBOT_USE_DB=1`, `RAPBOT_DB_HOST=localhost`, `RAPBOT_DB_PORT=3307`, etc.) so runs appear in the dashboard.

**Evolution via Docker:** The web container can spawn the evolution container when `RAPBOT_DOCKER_NETWORK` is set. Evolution triggered from `/evolve` runs inside the `evolution` container. Build with `docker compose build evolution`.

**Docker-only (no host Python/DB):** When the host cannot reach MySQL or lacks Python deps, run everything in the evolution container. Policy updates and convergence analysis must run in Docker so they can reach the DB:

```bash
# Bootstrap learned policy (one-time)
docker compose --profile evolution run --rm evolution python scripts/update_learned_policy.py --force --limit 300

# Continuous evolution
docker compose --profile evolution run --rm evolution python scripts/run_continuous.py --no-docker --bootstrap-policy --config config/continuous_runs.yaml
```

Use `--no-docker` so evolution runs inline instead of spawning nested Docker.

See [Remote Workflow](docs/REMOTE_WORKFLOW.md) for SSH-based dev.

---

## Main Entrypoints

| Script | Purpose |
|--------|---------|
| `scripts/run_couplet_evolution.py` | Evolve 2-line couplets with theme keywords |
| `scripts/run_verse_evolution.py` | Evolve 4-line verses (AABB/ABAB) |
| `scripts/run_verse_qd.py` | **Quality-Diversity** verse evolution (MAP-Elites), optional LM init and two-tier line+verse |
| `scripts/run_hierarchical.py` | **Hierarchical** verse evolution: couplet evolution → 4-bar from couplet archive + structural mutations |
| `scripts/run_continuous.py` | **Continuous loop** over YAML configs; learned policy, elite replay, parallel runs |
| `scripts/analyze_convergence.py` | **Convergence analysis** — report on convergence, diversity, stagnation from DB and `artifacts/learned_policy.json` |
| `scripts/run_weight_tuner.py` | Meta-optimize **couplet** fitness weights |
| `scripts/run_verse_weight_tuner.py` | Meta-optimize **verse** scoring weights |

### Example Commands

```bash
# Couplet evolution
python scripts/run_couplet_evolution.py --theme "pressure,mask" --population 100 --generations 30

# With run logging
python scripts/run_couplet_evolution.py --theme "pressure,mask" --runs-dir --output results.json

# 4-line verse evolution
python scripts/run_verse_evolution.py --theme "pressure,mask,survival" --population 80 --generations 30 --scheme AABB --runs-dir

# Quality-Diversity verse evolution
python scripts/run_verse_qd.py --theme "pressure,mask,survival" --population 120 --generations 100 --scheme AABB --runs-dir

# QD with LM seeding (requires OPENAI_API_KEY)
python scripts/run_verse_qd.py --theme "crown,empire" --init lm --proposer-model gpt-4o-mini --lm-budget 200

# Hierarchical: couplet → 4-bar
python scripts/run_hierarchical.py --theme "pressure,mask,survival" --couplet-gen 20 --4bar-gen 30 --runs-dir

# Continuous runs (config-driven)
python scripts/run_continuous.py --config config/continuous_runs.yaml --parallel 2

# Convergence analysis
python scripts/analyze_convergence.py --limit 500 --output reports/convergence_report.json

# Weight tuning
python scripts/run_weight_tuner.py --outer-generations 6 --output weights_evolved.json
python scripts/run_verse_weight_tuner.py --outer-generations 6 --output verse_weights_evolved.json
```

---

## Data & Artifacts

### Expected Data Paths (defaults under project root)

| Path | Description |
|------|-------------|
| `data/rhymes_grouped.csv` | Rhyme groups (rhyme family → word list) for mutation and scoring |
| `data/elite_kaggle_corpus_clean.txt` or `data/phaseA_kaggle_verse.txt` | Elite corpus lines for seed generation, ngram fluency, novelty |
| `data/evo_rhyme/templates.txt` | Line templates for template-based population init |
| `data/evo_rhyme/vocab/` | Optional vocab CSVs (adjectives, nouns, verbs, etc.) |
| `data/evo_rhyme/weak_endings.txt` | Weak line endings to penalize or reject |
| `data/elite_songs_lines_clean.csv` | Song catalog for web Evolve page and song-seeded evolution |
| `rhyme_siamese/` | Optional Siamese model directory for embedding-based semantic scoring |
| `artifacts/learned_policy.json` | Learned top configs for continuous runs and convergence analysis |

### Run Outputs (when `--runs-dir` is used)

- **Couplet / verse:** `data/evo_rhyme/runs/{timestamp}/`
- **QD:** `data/evo_rhyme/runs/qd_{timestamp}/`

Typical contents: `config.json`, `score_history.csv`, `archive.json`, `top_candidates.json`.

---

## Architecture

### Package Layout

```
rap_botV5/
├── config/
│   ├── settings.py      # load_settings(), RAPBOT_* env, config file merge
│   ├── evolution.yaml   # Evolution defaults (paths, fitness_weights, mutation_weights)
│   └── rapbot.yaml      # Full app config
├── evo_rhyme/           # Core evolution engine
│   ├── individual.py    # CoupletIndividual, VerseIndividual, LineFeatures, VerseFeatures
│   ├── fitness.py       # score_couplet, score_verse, compute_fitness, compute_verse_fitness
│   ├── constraints.py   # passes_constraints, passes_verse_constraints
│   ├── population.py    # RandomGenerator, TemplateGenerator, VerseSeedGenerator, create_*_population
│   ├── evolution.py     # evolve(), EvolutionConfig (couplet loop)
│   ├── verse_evolution.py  # evolve_verse_population, evolve_verse_qd, QDEvolutionConfig
│   ├── archive.py       # MAPElitesArchive, ArchiveDimension, default_verse_dimensions
│   ├── mutation.py      # mutate(), MUTATION_WEIGHTS
│   ├── crossover.py     # crossover (couplet)
│   ├── selection.py     # elitism, tournament_select, pareto_*, inject_random_immigrants
│   ├── phonetics.py     # count_syllables, extract_rhyme_tail, phonetic_similarity, stress
│   ├── rhyme_graph.py   # rhyme graph scoring
│   ├── block_builder.py # build_4bar_from_couplets, build_4bar_batch (hierarchical)
│   ├── couplet_archive.py # CoupletArchive, ScoredCouplet (hierarchical)
│   ├── structural_mutations.py # couplet_swap, transition_rewrite (verse-level ops)
│   ├── scoring/         # end_rhyme, internal_rhyme, coherence, punchline, novelty, penalties
│   ├── lm_proposer.py   # BarProposer (OpenAI/local)
│   ├── lm_rewriter.py   # BarRewriter
│   ├── emitters.py      # QD emitters
│   ├── line_evolution.py, line_archive.py, verse_builder.py  # Two-tier line+verse QD
│   └── db.py            # MySQL run/candidate/lineage/score_cache/experiments
├── webapp/
│   ├── main.py          # FastAPI app, health, exception handler
│   ├── api/routes.py    # REST API
│   ├── routes/pages.py  # HTML page routes
│   ├── services/        # run_service, evolution_service, analysis_service
│   ├── templates/       # Jinja2 (base, dashboard, evolve, run_detail, archive, etc.)
│   └── static/          # app.css
├── scripts/
│   ├── run_couplet_evolution.py
│   ├── run_verse_evolution.py
│   ├── run_verse_qd.py
│   ├── run_hierarchical.py
│   ├── run_continuous.py
│   ├── analyze_convergence.py
│   ├── update_learned_policy.py
│   ├── init_db.py
│   ├── import_song_catalog.py
│   ├── training/        # build_rhyme_embedding, build_elite_corpus, clean_elite_corpus
│   └── tools/           # validate_rhyme_data, rebuild_rhyme_groups, audit_rhyme_groups
├── data/
└── tests/
```

### Data Flow

1. **Config** → `config/settings.py` or `evolution.yaml` (paths, weights, evolution params).
2. **Corpus + rhyme CSV** → seed population (mixed / random / template / LM).
3. **Evolution loop** → evaluate fitness → selection → crossover → mutation → next generation; for QD, maintain MAP-Elites archive; for hierarchical, couplet archive → 4-bar builder → structural mutations.
4. **Outputs** → JSON, optional `--runs-dir`; when DB enabled, runs/candidates/lineage stored in MySQL and browsable in the web dashboard.

---

## Fitness & Scoring

Fitness is a weighted sum of component scores and penalties. Main components (see `evo_rhyme/fitness.py` and `config/evolution.yaml`):

- **Positive:** end_rhyme, internal_rhyme, rhyme_graph, multisyllabic, syllable_balance, stress_alignment, semantic, fluency, lexical_validity, ngram_fluency, novelty.
- **Penalties (negative weights):** weak_tail_penalty, repetition_penalty, theme_word_repetition_penalty, rhyme_family_repetition_penalty, identical_line_penalty, near_duplicate_penalty, template_penalty, corpus_overlap_penalty, theme_penalty.

Aggregate fitness is capped (e.g. 0.95). **Ngram fluency** is emphasized; a floor (e.g. 0.2) can reject nonsense. For verses, `VERSE_DEFAULT_WEIGHTS` and optional coherence/punchline/rhyme_chain components apply; the verse weight tuner evolves a compact weight genome (Structure / Meaning / Flavor / Penalty) for meta-fitness over benchmark themes.

---

## Testing

Tests live under `tests/test_evo_rhyme/` and `tests/test_webapp/`. Run with pytest from the project root:

```bash
cd rap_botV5
pytest tests/ -v
```

**Coverage:**

```bash
pip install pytest-cov
pytest tests/ -q --cov=evo_rhyme --cov=webapp --cov-report=term
```

Relevant modules: `test_fitness`, `test_constraints`, `test_phonetics`, `test_end_rhyme`, `test_individual`, `test_archive`, `test_mutation`, `test_selection`, `test_crossover`, `test_db`, `test_webapp/*`.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RAPBOT_CONFIG` | Path to main config file (YAML/JSON). |
| `RAPBOT_HOME` | Project root for resolving relative paths. |
| `RAPBOT_ELITE_CORPUS` | Override elite corpus path. |
| `RAPBOT_RHYME_CSV` | Override rhyme groups CSV path. |
| `RAPBOT_SIAMESE_DIR` | Override Siamese model directory. |
| `RAPBOT_USE_DB` | Set to `1` to enable MySQL run logging and web dashboard data. |
| `RAPBOT_DB_HOST` | MySQL host (default `127.0.0.1`; use `mysql` in Docker). |
| `RAPBOT_DB_PORT` | MySQL port (default `3306`; host typically `3307` when MySQL in Docker). |
| `RAPBOT_DB_USER` | MySQL user. |
| `RAPBOT_DB_PASSWORD` | MySQL password. |
| `RAPBOT_DB_NAME` | MySQL database name (default `rapbot`). |
| `RAPBOT_SONG_CATALOG_CSV` | Override song catalog CSV for /evolve. |
| `RAPBOT_DOCKER_NETWORK` | Docker network for web→evolution container spawn. |
| `RAPBOT_EVOLUTION_IMAGE` | Evolution container image name. |
| `RAPBOT_HOST_PROJECT_PATH` | Host path to mount in evolution container. |
| `RAPBOT_DISABLE_LM_REWRITER` | Set to `1` to disable LM rewriter (no OpenAI calls). |
| `OPENAI_API_KEY` | Required for LM-backed BarProposer / BarRewriter. |
| `NGROK_AUTHTOKEN` | Stable public tunnel URL (free at ngrok.com). |

Use `.env` and `python-dotenv` to load these from a file.

---

## Scripts Reference

### Evolution

- **`run_couplet_evolution.py`** — `--theme`, `--population`, `--generations`, `--output`, `--runs-dir`, `--init` (mixed|random|template), `--use-embeddings`, `--multiobjective`, `--use-niching`, etc.
- **`run_verse_evolution.py`** — `--theme`, `--population`, `--generations`, `--scheme` (AABB|ABAB), `--output`, `--runs-dir`.
- **`run_verse_qd.py`** — `--theme` (required), `--population`, `--generations`, `--scheme`, `--num-lines` (4|8|16), `--init` (mixed|random|template|lm), `--lm-budget`, `--runs-dir`, `--proposer-model`, `--db`, `--run-id`, etc.
- **`run_hierarchical.py`** — `--theme`, `--couplet-gen`, `--4bar-gen`, `--couplet-archive`, `--runs-dir`. Two-stage: couplet evolution → 4-bar from archive + structural mutations.
- **`run_continuous.py`** — `--config`, `--bootstrap-policy`, `--bootstrap-if-empty`, `--update-policy-every`, `--elite-replay-fraction`, `--parallel N`, `--no-docker`, `--use-docker`, `--start-at`, `--pause`. Continuous loop over YAML configs.

### Analysis & Policy

- **`analyze_convergence.py`** — `--limit`, `--output`. Analyzes convergence, diversity, stagnation from DB and `artifacts/learned_policy.json`.
- **`update_learned_policy.py`** — `--force`, `--limit`. Bootstrap/refresh `artifacts/learned_policy.json` from DB for continuous runs.

### Weight Tuning

- **`run_weight_tuner.py`** — `--outer-generations`, `--outer-population`, `--inner-population`, `--output`.
- **`run_verse_weight_tuner.py`** — `--outer-generations`, `--weights-file`, `--output`.

### Data & DB

- **`init_db.py`** — Create MySQL schema (runs, generations, candidates, lineage, score_cache, experiments, song_catalog).
- **`import_song_catalog.py`** — Import artist/song catalog from CSV into DB. Use `--replace` to truncate and reload.
- **`scripts/training/build_rhyme_embedding_artifact.py`**, **`build_elite_kaggle_corpus_*`**, **`clean_elite_corpus.py`**
- **`scripts/tools/validate_rhyme_data.py`**, **`rebuild_rhyme_groups.py`**, **`audit_rhyme_groups.py`**

### Inspection

- **`inspect_candidate.py`** — Inspect a candidate from results JSON.

---

## License & Citation

See repository for license. If you use this code in research, please cite the repository.
