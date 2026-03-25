# Architecture
(See docs/CONTEXT.md)

## High-Level Pipeline

Generation → Scoring → Selection → Mutation/Crossover → Next Generation

For Quality-Diversity: same loop plus MAP-Elites archive; optionally two-tier (line evolution + verse assembly).

---

## Package Layout

```
rap_botV5/
├── config/
│   ├── settings.py       # load_settings(), get_qd_defaults(), RAPBOT_* env
│   ├── evolution.yaml    # Evolution defaults (paths, fitness_weights, mutation_weights, proposer)
│   └── rapbot.yaml       # Full app config (base model, stage3, critic, stats)
├── evo_rhyme/            # Core evolution engine
│   ├── individual.py     # CoupletIndividual, VerseIndividual, LineFeatures, VerseFeatures
│   ├── fitness.py        # score_couplet, score_verse, compute_fitness, weights
│   ├── constraints.py    # passes_constraints, passes_verse_constraints
│   ├── population.py     # RandomGenerator, TemplateGenerator, VerseSeedGenerator
│   ├── evolution.py      # evolve(), EvolutionConfig (couplet loop)
│   ├── verse_evolution.py # evolve_verse_population, evolve_verse_qd, QDEvolutionConfig
│   ├── archive.py        # MAPElitesArchive, ArchiveDimension, compact_style_dimensions
│   ├── mutation.py       # mutate(), MUTATION_WEIGHTS, LEGACY_MUTATION_WEIGHTS
│   ├── crossover.py      # crossover (line-level, phrase-slice)
│   ├── selection.py      # elitism, tournament_select, inject_random_immigrants
│   ├── phonetics.py      # syllable_count_line, extract_rhyme_tail, stress patterns
│   ├── rhyme_graph.py    # rhyme graph scoring
│   ├── scoring/          # end_rhyme, internal_rhyme, coherence, punchline, novelty, penalties
│   ├── lm_proposer.py    # BarProposer (OpenAI/local)
│   ├── lm_rewriter.py    # BarRewriter (LM-backed mutation)
│   ├── emitters.py       # QD emitters
│   ├── line_evolution.py # Two-tier line evolution
│   ├── line_archive.py   # Line-level MAP-Elites
│   ├── verse_builder.py  # Assemble verses from lines
│   ├── repro.py          # seed_everything()
│   ├── db.py             # MySQL run/experiment storage
│   └── ...
├── scripts/
│   ├── run_couplet_evolution.py
│   ├── run_verse_evolution.py
│   ├── run_verse_qd.py       # QD (MAP-Elites) verse evolution
│   ├── run_weight_tuner.py   # Couplet weight meta-optimization
│   ├── run_verse_weight_tuner.py
│   ├── run_learning_validation.py
│   ├── run_control_experiment.py
│   ├── archive_artifacts_to_db.py
│   ├── training/             # build_rhyme_embedding, build_elite_kaggle_corpus, clean_elite_corpus
│   └── tools/                # validate_rhyme_data, rebuild_rhyme_groups, audit_rhyme_groups
├── webapp/               # FastAPI dashboard
│   ├── main.py
│   ├── api/routes.py     # REST API
│   ├── routes/pages.py   # HTML pages
│   ├── services/run_service.py
│   └── templates/
├── data/
│   ├── rhymes_grouped.csv
│   ├── elite_kaggle_corpus_clean.txt
│   └── evo_rhyme/        # templates, vocab, weak_endings, runs/
└── tests/
    └── test_evo_rhyme/   # fitness, constraints, phonetics, archive, mutation, etc.
```

---

## Modules

### 1. Generation Layer

- **TemplateGenerator** — Fill templates from vocab (adjectives, nouns, verbs) with theme words
- **RandomGenerator** — Corpus sampling, rhyme-group swaps
- **VerseSeedGenerator** — Mixed init (template + corpus + random)
- **LM (BarProposer)** — OpenAI/local LM for seed lines and mutations (QD init lm)

### 2. Scoring Layer (`evo_rhyme/scoring/`)

- **end_rhyme** — End-word rhyme quality
- **internal_rhyme** — Internal rhyme density
- **coherence** — Cross-line semantic coherence
- **punchline** — Punchline strength
- **novelty** — Distance from corpus
- **penalties** — repetition, theme_penalty, corpus_overlap, etc.
- **rhyme_graph** — Rhyme family graph scoring
- **syllable_balance** — Line length consistency

Plus: ngram_fluency, lm_fluency, semantic, stress_alignment, multisyllabic, lexical_validity.

### 3. Evolution Layer

**Selection:** Tournament, elitism, random immigrants

**Variation:**
- **Legacy:** end_word_swap, internal_rhyme_insert, stressed_vowel_swap, semantic_swap, compression, expansion, etc.
- **LM-backed:** lm_rhyme_rewrite, lm_internal_rhyme, lm_theme_rewrite, lm_paraphrase, lm_structural_rewrite, lm_metaphor_inject, lm_score_guided, etc.

**Crossover:** Line-level (swap full lines); phrase-slice when structure matches

### 4. Archive Layer (QD)

- **MAPElitesArchive** — Grid over behavioral dimensions
- **Dimensions:** rhyme_density, intensity, syllable_tightness, theme_balance, line_length_variance
- **Modes:** compact_style, ultra_compact, default (phonetic graph)

### 5. Logging Layer

- **VerseRunLogger** — Writes config.json, score_history.csv, top_candidates.json
- **DB** — insert_run, insert_generation, insert_candidate, list_runs (when RAPBOT_USE_DB=1)
- **Web dashboard** — Browse runs, candidates, archive, lineage

---

## Design Constraints

- Modular — replaceable scoring, mutation, init
- Config-driven — evolution.yaml, rapbot.yaml
- Reproducible — seed + config snapshot + git hash

For a narrative research-oriented view of the same system (modes, policy loop, assessment), see [docs/analysis/system_overview.md](analysis/system_overview.md).
