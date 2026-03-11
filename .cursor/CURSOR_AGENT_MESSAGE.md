## Message For Cursor Agent

Timestamp: 2026-03-10 (America/Chicago)
From: Codex terminal agent

Please use this as an initial repo handoff.

### Repo Snapshot
- Project: `rap_botV5` (Python rap-generation and rhyme-evolution system)
- Core engine: `evo_rhyme/` (evolution loop, mutation, crossover, fitness, constraints, phonetics, scoring)
- Legacy/helper module: `rapbot/` (rhyme scorer + utilities)
- Config: `config/settings.py` + `config/rapbot.yaml`
- Entrypoint scripts: `scripts/run_verse_evolution.py`, `scripts/run_couplet_evolution.py`, `scripts/run_weight_tuner.py`
- Data/artifacts: `data/` (corpora, rhyme groups, run outputs, scored datasets)
- Tests: `tests/test_evo_rhyme/`

### Notable Defaults
- Base model: `Qwen/Qwen2.5-14B-Instruct`
- Main generation defaults are in `config/rapbot.yaml`
- Shared settings loader resolves paths/env overrides in `config/settings.py`

### Request
If you are active in Cursor chat, acknowledge this note and continue from this repo context.
