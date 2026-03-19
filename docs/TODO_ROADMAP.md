# Roadmap
(See docs/CONTEXT.md)

## Phase 1 (Current)

- Stabilize evolution pipeline
- Improve scoring (Plan 5: coherence/novelty rebalance)
- Add policy-guided mutation (static | learned | explore_mix)
- Fix learning validation metrics (Plan 3)
- Reproducibility audit (Plan 1)

## Phase 2

- Address template contamination (Plan 2)
- Evolve existing verses (`scripts/improve_verse.py`)
- Preserve style across evolution

## Phase 3

- Run longer generations (Plan 4: 80–100 default)
- Audio → transcription → evolution
- Beat-aware lyrics

## Phase 4

- Full rap generator system
- Web UI + deployment (webapp already exists)
- Docker Compose for MySQL + dashboard

---

## LLM Backend Roadmap

**Context:** ChatGPT/OpenAI calls are used in `lm_proposer.py` (BarProposer) and `lm_rewriter.py` (BarRewriter). Both already support `api_base` / `RAPBOT_REWRITER_API_BASE` for OpenAI-compatible endpoints.

**Constraint:** No good GPU available right now → local vLLM not viable yet. Keep OpenAI as default; swap backend when GPU or cloud inference is available.

### Option A — Keep OpenAI (Current / Baseline)
- Best for: finishing experiments, benchmarking, demos
- Pros: high quality, no infra; Cons: cost and latency scale with population/generations

### Option B — Swap to OpenAI-compatible endpoint (When GPU Available)
- **Highest ROI next step.** Use vLLM, RunPod, LM Studio, together.ai, or Groq.
- Set `api_base` or `RAPBOT_REWRITER_API_BASE`; no code changes needed.
- Gains: cost reduction, faster iterations, more evolutionary runs.
- Do NOT remove OpenAI; keep as fallback / comparison baseline.

### Option C — Fine-tuned local model (Endgame)
- After stabilizing Option B and collecting evolved verses + mutation logs.
- Train rap-specific model from evolutionary data → self-improving loop.
- Defer until infra and dataset are ready.

### Recommended sequence
1. Keep OpenAI as default (no GPU).
2. When GPU or cloud inference is available: add vLLM / compatible endpoint via `api_base`.
3. Stress-test with higher population, generations, LM budget.
4. Log mutation inputs/outputs and score deltas for future fine-tuning dataset.
5. Train own model only after Option B is stable.

---

## LLM Testing Strategy (Progressive Validation)

Before scaling runs, validate each path in isolation:

### Phase 1 — Dry run (no API)
- `RAPBOT_DISABLE_LM_REWRITER=1`
- Run evolution with template/corpus init, no LM mutations
- Check: evolution runs, constraints enforced, scoring works

### Phase 2 — Proposer only
- `--init lm`, `lm_budget=0` (no LM mutations)
- Check: LM generates valid initial population, rhyme correctness, diversity

### Phase 3 — Rewriter only
- Template/corpus init, LM mutations enabled
- Check: LM mutations improve fitness, ~30–60% useful mutation rate

### Phase 4 — Full system (small scale)
- `--population 20 --generations 5`
- Track: API calls, best fitness, diversity

### Phase 5 — Scaling test
- `--population 50 --generations 20`
- Evaluate: convergence, cost vs improvement, archive coverage

### Metrics to log
- `lm_calls_count`, `lm_success_rate`, `lm_mutation_improvement_rate`
- `avg_score_delta_from_lm`, tokens used (if available), time per generation

---

## Immediate Next Steps

1. Run baseline experiment with --seed
2. Improve rhyme planner (docs/improvement_plan.md)
3. Add experiment tracking (git hash in config.json)
4. Fix learning validation avg_fitness bug
