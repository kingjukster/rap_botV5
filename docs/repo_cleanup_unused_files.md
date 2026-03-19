# Repo Cleanup: Unused / Reducible Files

This is a best-effort findings report based on:
- Static import analysis of Python modules under `evo_rhyme/`, `webapp/`, `rapbot/`, and `config/`
- Reference scanning for Jinja template names used by `webapp/`

## Unused Python modules (static import graph)

These modules/files appear to be **unused** by any direct Python imports in this repo (including tests):

1. `rapbot/`
2. `rapbot/rhyme_scorer.py`

Notes:
- No `import rapbot` / `from rapbot ...` statements were found in the codebase.
- There are mentions of `rapbot/` in documentation, but not as executable imports.

## Likely-used (false positives avoided)

Some files may look “unused” if only direct imports are checked:

- `evo_rhyme/beat/prosody.py` and `evo_rhyme/beat/scoring.py` are imported indirectly via `evo_rhyme/beat/__init__.py` (relative imports), and are covered by `tests/test_evo_rhyme/test_beat.py`.

## Templates / static assets

Template references used by the FastAPI routes were found for:
- `webapp/templates/base.html`
- `webapp/templates/dashboard.html`
- `webapp/templates/run_detail.html`
- `webapp/templates/run_generations.html`
- `webapp/templates/archive.html`
- `webapp/templates/lineage.html`
- `webapp/templates/candidate_detail.html`
- `webapp/templates/seeds.html`
- `webapp/templates/score_cache.html`
- `webapp/templates/about.html`
- `webapp/templates/error.html`

`webapp/templates/base.html` references the static stylesheet at:
- `/static/app.css`

So there were no orphan HTML templates or CSS references detected.

## Safe reductions / consolidation suggestions

1. **Optional deprecation cleanup**
   - If you no longer need the legacy `rapbot/` helper package, consider removing it from packaging (`pyproject.toml` `include = ["...","rapbot*"]`) or marking it more explicitly as deprecated-only.
   - If you want to keep it for external users, you can leave it as-is but avoid misleading “unused” noise.

2. **Dev artifact hygiene**
   - `.gitignore` has been updated to avoid committing local dev detritus:
     - ignores `.coverage` and `.pytest_cache/`
     - no longer ignores `data/evo_rhyme/runs/` or `checkpoints/` (since you chose to keep both tracked)

## Suggested verification

- Run `pytest` to ensure nothing relied on any legacy import side-effects.
- Run a minimal import smoke test: `python -c "import evo_rhyme; import webapp"`.

