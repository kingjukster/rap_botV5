# Data Flow
(See docs/CONTEXT.md)

## Inputs

| Input | Path/Source | Role |
|-------|-------------|------|
| Theme keywords | CLI `--theme` | Constrain semantic, template filling |
| Rhyme groups | `data/rhymes_grouped.csv` | Mutation swaps, rhyme scoring |
| Elite corpus | `data/elite_kaggle_corpus_clean.txt` | Seed generation, ngram fluency, novelty |
| Templates | `data/evo_rhyme/templates.txt` | Template-based population init |
| Vocab | `data/evo_rhyme/vocab/*.txt` | Template filling (nouns, verbs, adjectives) |
| Weak endings | `data/evo_rhyme/weak_endings.txt` | Penalized line endings |
| Config | `config/evolution.yaml`, `config/rapbot.yaml` | All parameters |
| Siamese model | `rhyme_siamese/` (optional) | Embedding-based semantic scoring |

---

## Pipeline

1. **Load config** — evolution.yaml + rapbot.yaml + env overrides
2. **Seed RNG** — `seed_everything(seed)` when `--seed` passed
3. **Generate initial population** — mixed/random/template/lm
4. **Score candidates** — fitness = weighted sum of component scores
5. **Select parents** — tournament, elitism, random immigrants
6. **Apply crossover** — line-level or phrase-slice
7. **Apply mutation** — lexical, rhyme, LM-backed (if configured)
8. **Update archive** (QD) — MAP-Elites grid
9. **Produce next generation** — repeat from step 4

---

## Outputs

### Run directory (`data/evo_rhyme/runs/`)

- **Couplet / verse:** `{timestamp}/`
- **QD:** `qd_{timestamp}/`

### Per-run artifacts

| File | Content |
|------|---------|
| `config.json` | population, generations, theme, scheme, archive_mode, seed_info (if --seed), etc. |
| `score_history.csv` | gen, best_fitness, avg_fitness (and archive_coverage for QD) |
| `archive.json` | MAP-Elites archive (QD only) |
| `top_candidates.json` | Best individuals by generation |

### DB (when RAPBOT_USE_DB=1)

- **runs** — script_name, theme_keywords, config_json, status
- **generations** — run_id, gen, best_fitness, archive_coverage
- **candidates** — run_id, gen, lines, fitness, scores
- **experiments** — name, description, mode
- **experiment_arms** — arm_name, control_snapshot

---

## Randomness Points

- **Initial generation** — template fill, corpus sample, LM sampling
- **Mutation** — operator choice, word/line selection
- **Selection** — tournament draw, immigrant injection
- **Crossover** — parent swap, phrase-slice decision

**Reproducibility:** Pass `--seed` to seed Python/NumPy/Torch; LM API calls remain non-deterministic.

---

## Reproducibility Requirements

- Fixed seeds (`--seed`)
- Config snapshot in run dir (`config.json`)
- Git commit hash (not yet in config.json — see Plan 1)
- Deterministic RNG usage (`evo_rhyme.repro.seed_everything`)
