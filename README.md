# Rap Bot V5 — Evolutionary Rap Verse Generation

Evolutionary rap verse generation with explicit rhyme, fluency, and penalty scoring. The system evolves **couplets** (2-line) and **4-line verses** via tournament selection, crossover, and mutation, optimizing for end rhyme quality, internal rhyme, fluency, semantic relevance, and novelty while penalizing repetition and nonsense. Optional **Quality-Diversity (MAP-Elites)** evolution explores a grid of behavioral niches (rhyme density, intensity, syllable tightness) to produce diverse, high-quality verses.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Main Entrypoints](#main-entrypoints)
- [Data & Artifacts](#data--artifacts)
- [Architecture](#architecture)
- [Fitness & Scoring](#fitness--scoring)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Scripts Reference](#scripts-reference)

---

## Features

- **Couplet evolution** — Evolve 2-line couplets with theme keywords, optional multi-objective (Pareto) or niching.
- **4-line verse evolution** — AABB / ABAB (and more) rhyme schemes with phrase-level crossover and mutation.
- **Quality-Diversity (QD)** — MAP-Elites archive over behavioral dimensions (rhyme density, intensity, syllable tightness, theme breadth); optional two-tier line + verse evolution and LM-backed proposers.
- **Fitness components** — End rhyme, internal rhyme, rhyme graph, multisyllabic overlap, syllable balance, stress alignment, semantic/theme, fluency, ngram fluency, novelty; penalties for repetition, weak tails, corpus overlap, cliché.
- **Population init** — Mixed (template + corpus + random), random, template-only, or **LM** (OpenAI/local) for seeding.
- **Weight tuning** — Outer-loop meta-optimization of fitness weights (couplet or verse) over benchmark prompts.
- **Run logging** — Optional per-run directories with config, score history, archive, and top candidates.

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
| **Optional (LM)** | `openai` (for `BarProposer` / `BarRewriter`) |

No `setup.py` or `pyproject.toml` is present; run scripts from the project root with `python scripts/...` so that `config` and `evo_rhyme` are on `PYTHONPATH` (scripts add the repo root to `sys.path`).

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

## Main Entrypoints

| Script | Purpose |
|--------|---------|
| `scripts/run_couplet_evolution.py` | Evolve 2-line couplets with theme keywords |
| `scripts/run_verse_evolution.py` | Evolve 4-line verses (AABB/ABAB) |
| `scripts/run_verse_qd.py` | **Quality-Diversity** verse evolution (MAP-Elites), optional LM init and two-tier line+verse |
| `scripts/run_weight_tuner.py` | Meta-optimize **couplet** fitness weights over benchmark prompts |
| `scripts/run_verse_weight_tuner.py` | Meta-optimize **verse** scoring weights (compact weight genome) |

### Example Commands

```bash
# Couplet evolution (default: population 100, generations 30)
python scripts/run_couplet_evolution.py --theme "pressure,mask" --population 100 --generations 30

# With run logging to data/evo_rhyme/runs/{timestamp}/
python scripts/run_couplet_evolution.py --theme "pressure,mask" --runs-dir --output results.json

# 4-line verse evolution
python scripts/run_verse_evolution.py --theme "pressure,mask,survival" --population 80 --generations 30 --scheme AABB --runs-dir

# Quality-Diversity verse evolution (MAP-Elites, theme required)
python scripts/run_verse_qd.py --theme "pressure,mask,survival" --population 120 --generations 100 --scheme AABB --runs-dir

# QD with LM seeding (requires OPENAI_API_KEY)
python scripts/run_verse_qd.py --theme "crown,empire" --init lm --proposer-model gpt-4o-mini --lm-budget 200

# Evolve couplet fitness weights (outer-loop)
python scripts/run_weight_tuner.py --outer-generations 6 --output weights_evolved.json

# Evolve verse scoring weights
python scripts/run_verse_weight_tuner.py --outer-generations 6 --output verse_weights_evolved.json
```

---

## Data & Artifacts

### Expected Data Paths (defaults under project root)

| Path | Description |
|------|--------------|
| `data/rhymes_grouped.csv` | Rhyme groups (e.g. rhyme family → word list) for mutation and scoring |
| `data/elite_kaggle_corpus_clean.txt` or `data/phaseA_kaggle_verse.txt` | Elite corpus lines for seed generation, ngram fluency, and novelty |
| `data/evo_rhyme/templates.txt` | Line templates for template-based population init |
| `data/evo_rhyme/vocab/` | Optional vocab CSVs (adjectives, nouns, verbs, etc.) |
| `data/evo_rhyme/weak_endings.txt` | Weak line endings to penalize or reject |
| `rhyme_siamese/` | Optional Siamese model directory for embedding-based semantic scoring |

If a path is missing, scripts fall back to other defaults (e.g. `elite_kaggle_corpus_clean_plus.jsonl`, `phaseA_kaggle_verse.txt`) where implemented.

### Run Outputs (when `--runs-dir` is used)

- **Couplet / verse:** `data/evo_rhyme/runs/{timestamp}/`
- **QD:** `data/evo_rhyme/runs/qd_{timestamp}/`

Typical contents:

- `config.json` — Full evolution config (population, generations, theme, scheme, archive_mode, etc.)
- `score_history.csv` — Per-generation score history
- `archive.json` — MAP-Elites archive (QD only)
- `top_candidates.json` — Best individuals/candidates

---

## Architecture

### Package layout

```
rap_botV5/
├── config/
│   ├── settings.py      # load_settings(), RAPBOT_* env, config file merge
│   ├── evolution.yaml   # Evolution defaults (paths, fitness_weights, mutation_weights)
│   └── rapbot.yaml      # Full app config (paths, generation, critic, etc.)
├── evo_rhyme/           # Core evolution engine
│   ├── individual.py   # CoupletIndividual, VerseIndividual, LineFeatures, VerseFeatures
│   ├── fitness.py      # score_couplet, score_verse, compute_fitness, compute_verse_fitness, weights
│   ├── constraints.py  # passes_constraints, passes_verse_constraints, ConstraintConfig
│   ├── population.py   # RandomGenerator, TemplateGenerator, VerseSeedGenerator, create_*_population
│   ├── evolution.py    # evolve(), EvolutionConfig (couplet loop)
│   ├── verse_evolution.py  # evolve_verse_population, evolve_verse_qd, VerseEvolutionConfig, QDEvolutionConfig
│   ├── archive.py      # MAPElitesArchive, ArchiveDimension, default_verse_dimensions, compact_style_dimensions
│   ├── mutation.py     # mutate(), MUTATION_WEIGHTS, get_tail_to_words
│   ├── crossover.py    # crossover (couplet)
│   ├── selection.py    # elitism, tournament_select, inject_random_immigrants, pareto_*
│   ├── phonetics.py    # count_syllables, extract_rhyme_tail, phonetic_similarity, stress patterns
│   ├── rhyme_graph.py  # rhyme graph scoring
│   ├── scoring/        # end_rhyme, internal_rhyme, coherence, punchline, novelty, penalties, etc.
│   ├── lm_proposer.py  # BarProposer (OpenAI/local), ProposerConfig
│   ├── lm_rewriter.py  # BarRewriter
│   ├── emitters.py     # QD emitters (e.g. specialized mutation/selection)
│   ├── line_evolution.py, line_archive.py, verse_builder.py  # Two-tier line+verse QD
│   └── ...
├── scripts/
│   ├── run_couplet_evolution.py
│   ├── run_verse_evolution.py
│   ├── run_verse_qd.py
│   ├── run_weight_tuner.py
│   ├── run_verse_weight_tuner.py
│   ├── training/       # build_rhyme_embedding_artifact, build_elite_kaggle_corpus_*, clean_elite_corpus
│   └── tools/          # validate_rhyme_data, rebuild_rhyme_groups, audit_rhyme_groups, enrich_kaggle_corpus
├── data/
│   ├── rhymes_grouped.csv
│   ├── elite_kaggle_corpus_clean.txt (or phaseA_kaggle_verse.txt)
│   └── evo_rhyme/      # templates, vocab, weak_endings, runs/
└── tests/
    └── test_evo_rhyme/ # pytest tests for fitness, constraints, phonetics, archive, etc.
```

### Data flow

1. **Config** → `config/settings.py` or `evolution.yaml` (paths, weights, evolution params).
2. **Corpus + rhyme CSV** → seed population (mixed / random / template / LM).
3. **Evolution loop** → evaluate fitness → selection → crossover → mutation → next generation; for QD, maintain MAP-Elites archive and optionally two-tier line archive + verse assembly.
4. **Outputs** → JSON (top candidates, config), optional `--runs-dir` (config, score_history, archive, top_candidates).

---

## Fitness & Scoring

Fitness is a weighted sum of component scores and penalties. Main components (see `evo_rhyme/fitness.py` and `config/evolution.yaml`):

- **Positive:** end_rhyme, internal_rhyme, rhyme_graph, multisyllabic, syllable_balance, stress_alignment, semantic, fluency, lexical_validity, ngram_fluency, novelty.
- **Penalties (negative weights):** weak_tail_penalty, repetition_penalty, theme_word_repetition_penalty, rhyme_family_repetition_penalty, identical_line_penalty, near_duplicate_penalty, template_penalty, corpus_overlap_penalty, theme_penalty.

Aggregate fitness is capped (e.g. 0.95) to avoid saturation. **Ngram fluency** is emphasized to keep phrase structure natural; a floor (e.g. 0.2) can reject nonsense. For verses, `VERSE_DEFAULT_WEIGHTS` and optional coherence/punchline/rhyme_chain components apply; verse weight tuner evolves a compact weight genome (Structure / Meaning / Flavor / Penalty) for meta-fitness over benchmark themes.

---

## Testing

Tests live under `tests/test_evo_rhyme/`. Run with pytest from the project root:

```bash
cd rap_botV5
pytest tests/ -v
```

Relevant test modules:

- `test_fitness.py` — score_couplet, compute_fitness, score ranges, ngram floor, LM fluency
- `test_constraints.py` — weak endings, word/syllable limits, repetition, theme presence
- `test_phonetics.py` — syllable counts, rhyme similarity, stress
- `test_end_rhyme.py` — end rhyme scoring
- `test_individual.py` — analyze_individual / analyze_verse_individual, features
- `test_style_prompt_genome.py` — style genome mutation/crossover, archive dimensions
- `test_emitters_specialized.py` — QD emitters
- `test_rhyme_embedding.py` — rhyme embedding build and neighbors

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RAPBOT_CONFIG` | Path to main config file (YAML/JSON). |
| `RAPBOT_HOME` | Project root for resolving relative paths. |
| `RAPBOT_ELITE_CORPUS` | Override elite corpus path. |
| `RAPBOT_RHYME_CSV` | Override rhyme groups CSV path. |
| `RAPBOT_SIAMESE_DIR` | Override Siamese model directory. |
| `OPENAI_API_KEY` | Required for LM-backed BarProposer / BarRewriter (e.g. QD with `--init lm`). |

Use `.env` and `python-dotenv` in your own wrapper if you want to load these from a file.

---

## Scripts Reference

### Evolution

- **`run_couplet_evolution.py`** — `--theme`, `--population`, `--generations`, `--output`, `--runs-dir`, `--init` (mixed|random|template), `--use-embeddings`, `--multiobjective`, `--use-niching`, `--min-fluency`, `--min-semantic`, `--min-ngram`, `--style-corpus`, etc.
- **`run_verse_evolution.py`** — `--theme`, `--population`, `--generations`, `--scheme` (AABB|ABAB), `--output`, `--runs-dir`.
- **`run_verse_qd.py`** — `--theme` (required), `--population`, `--generations`, `--scheme`, `--num-lines` (4|8|16), `--init` (mixed|random|template|lm), `--lm-budget`, `--runs-dir`, `--proposer-model`, `--proposer-backend` (openai|local_hf), `--line-pop`, `--line-gens`, `--compose-ratio`, etc.

### Weight tuning

- **`run_weight_tuner.py`** — `--outer-generations`, `--outer-population`, `--inner-population`, `--output`.
- **`run_verse_weight_tuner.py`** — `--outer-generations`, `--weights-file` (to refine), `--output`.

### Training / data

- **`scripts/training/build_rhyme_embedding_artifact.py`** — Build rhyme embedding artifact.
- **`scripts/training/build_elite_kaggle_corpus_multi_stage.py`** — Build elite corpus.
- **`scripts/training/clean_elite_corpus.py`** — Clean elite corpus.
- **`scripts/tools/validate_rhyme_data.py`** — Validate rhyme groups.
- **`scripts/tools/rebuild_rhyme_groups.py`**, **`update_rhyme_groups.py`**, **`audit_rhyme_groups.py`** — Rhyme data maintenance.
- **`scripts/import_vocab_from_csv.py`** — Import vocab from CSV.

### Inspection

- **`scripts/inspect_candidate.py`** — Inspect a candidate (e.g. from results JSON).

---

## License & Citation

See repository for license. If you use this code in research, please cite the repository.
