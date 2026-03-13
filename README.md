# Rap Bot V5 — Evolutionary Rap Verse Generation

Evolutionary rap verse generation with explicit rhyme, fluency, and penalty scoring. The system evolves couplets and 4-line verses via tournament selection, crossover, and mutation, optimizing for end rhyme quality, internal rhyme, fluency, semantic relevance, and novelty while penalizing repetition and nonsense.

## Main Entrypoints

| Script | Purpose |
|--------|---------|
| `run_couplet_evolution.py` | Evolve 2-line couplets with theme keywords |
| `run_verse_evolution.py` | Evolve 4-line verses (AABB/ABAB rhyme schemes) |
| `run_weight_tuner.py` | Meta-optimize fitness weights across benchmark prompts |

### Example Commands

```bash
# Couplet evolution (default: population 100, generations 30)
python scripts/run_couplet_evolution.py --theme "pressure,mask" --population 100 --generations 30

# With run logging to data/evo_rhyme/runs/{timestamp}/
python scripts/run_couplet_evolution.py --theme "pressure,mask" --runs-dir --output results.json

# 4-line verse evolution
python scripts/run_verse_evolution.py --theme "pressure,mask,survival" --population 80 --generations 30 --scheme AABB --runs-dir

# Evolve fitness weights (outer-loop meta-optimization)
python scripts/run_weight_tuner.py --outer-generations 6 --output weights_evolved.json
```

## Architecture

- **evo_rhyme** — Core evolution engine: fitness scoring, mutation, crossover, selection
- **Data flow**: Corpus + rhyme CSV → seed population → evolution loop → outputs (JSON, runs dir)
- **Paths**: `config/evolution.yaml` defines canonical paths (`rhyme_groups_csv`, `siamese_model_dir`, `elite_corpus`) and evolution defaults

## Installation

```bash
pip install -r requirements.txt
```

Dependencies include PyTorch, transformers, pronouncing, g2p-en, networkx, and others. See `requirements.txt`.

## Configuration

Canonical defaults live in `config/evolution.yaml`:

- **paths**: `rhyme_groups_csv`, `siamese_model_dir`, `elite_corpus`
- **evolution**: `population_size`, `generations`, `num_elites`, `tournament_k`
- **fitness_weights** / **mutation_weights**: Scoring and mutation type weights
- **output**: `runs_dir` for run artifacts

Override via CLI flags or environment variables (`RAPBOT_ELITE_CORPUS`, `RAPBOT_CONFIG`, etc.).
