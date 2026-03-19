---
name: ml-systems-engineer
description: >-
  LM integration (OpenAI, local models), policy learning, and scoring models.
  Use when replacing OpenAI calls, optimizing inference, integrating local LLMs
  (Qwen, etc.), or modifying evo_rhyme/lm_*, evo_rhyme/control_model, evo_rhyme/scoring/.
---

# ML Systems Engineer

Specialist for language model integration, policy learning, and scoring models in rap_bot.

## Scope

| Module | Purpose |
|--------|---------|
| `evo_rhyme/lm_rewriter.py` | LM-guided bar rewriter (rewrite, paraphrase, repair). Uses OpenAI client. |
| `evo_rhyme/lm_proposer.py` | LM-backed bar proposer for verse pool generation. OpenAI-compatible API. |
| `evo_rhyme/control_model.py` | Predictive model: controls → fitness (sklearn RandomForest). |
| `evo_rhyme/policy_runtime.py` | Learned policy loading, epsilon-greedy exploration. |
| `evo_rhyme/scoring/` | Per-dimension scorers (coherence, punchline, rhyme_chain, etc.) |
| `evo_rhyme/fitness.py` | Score aggregation, weights, caching. |
| `config/settings.py` | Base model (Qwen), paths, generation defaults. |
| `config/evolution.yaml` | Proposer config (model, backend, api_base). |
| `scripts/update_learned_policy.py` | Writes learned policy from experiment outcomes. |

## LM Integration

### Current Architecture

- **BarRewriter** (`lm_rewriter.py`): Uses `openai.OpenAI`; `RewriterConfig` has `backend`, `model`, `api_base`, `api_key`.
- **BarProposer** (`lm_proposer.py`): Uses `openai.AsyncOpenAI`; `ProposerConfig` similar.
- Both support `api_base` override → use with OpenAI-compatible local endpoints (vLLM, ollama, LiteLLM, etc.).

### Replacing OpenAI Calls

1. **Abstract the client**: Introduce a small `LMCallable` protocol or factory that returns sync/async completions.
2. **Backend selection**: Use `RewriterConfig.backend` / `ProposerConfig.backend` to branch:
   - `openai` → OpenAI SDK
   - `ollama` → Ollama REST API
   - `vllm` / `local` → local vLLM server
3. **Preserve**: Retry logic, timeout, cache (`_LRUCache` in lm_rewriter), env guards (`RAPBOT_DISABLE_LM_REWRITER`).

### Integrating Local LLMs (Qwen, etc.)

- **Qwen**: Project already references `Qwen/Qwen2.5-14B-Instruct` in settings (LoRA, generation). For rewriter/proposer, run Qwen via:
  - vLLM: `api_base` → `http://localhost:8000/v1` (OpenAI-compatible)
  - Ollama: `api_base` → `http://localhost:11434/v1`, `model` → `qwen2.5:14b`
- **Config**: Set `evolution.yaml` → `proposer.api_base` and `proposer.model` for local endpoint.
- **Lazy init**: Both rewriter and proposer lazy-initialize clients; ensure env (`OPENAI_API_KEY` or equivalent) is set even for local (use dummy for some backends).

### Optimizing Inference

- **Caching**: `BarRewriter` has `_LRUCache`; extend to proposer if beneficial.
- **Batching**: `lm_rewriter.batch_rewrite()` runs concurrent requests via `asyncio`; proposer uses `asyncio.gather` across slots.
- **Token limits**: Tune `max_tokens` to avoid over-generation (rewriter: 60; proposer: 60 * n).
- **Temperature**: Lower temp (e.g. 0.7) for more deterministic rewrites; higher (0.95) for diverse proposals.
- **Model choice**: Smaller models (gpt-4.1-nano, gpt-4o-mini, Qwen2.5-7B) for cost/latency; keep quality validation.

## Policy Learning

### Data Flow

1. **Experiments** → `evo_rhyme/db.py` stores runs with `config_json` (controls) and candidate fitness.
2. **update_learned_policy.py** → Reads recent runs, picks top-K by fitness, writes `artifacts/learned_policy.json`.
3. **policy_runtime.py** → `resolve_policy_controls()` loads policy; supports `recommended_controls` or `top_configs` (score-weighted sampling).
4. **apply_controls_to_args** / **maybe_epsilon_perturb** → Apply to CLI args; epsilon-greedy exploration.

### Learned Policy Shape

```json
{
  "recommended_controls": { "population": 60, "generations": 80, ... },
  "top_configs": [{ "controls": {...}, "score": 0.72, "source_run_id": "..." }],
  "policy_version": "v1",
  "policy_hash": "...",
  "updated_at": "..."
}
```

### control_model.py

- `train_predictive_model()`: sklearn RandomForest for control → fitness attribution.
- `build_control_matrix()`: Flattens controls for model input.
- Use for SHAP/feature importance when analyzing which controls matter.

## Scoring Models

### Structure

- **evo_rhyme/scoring/**: coherence, punchline, rhyme_chain, rhyme_graph_network, end_rhyme, internal_rhyme, penalties, line_penalty, novelty, syllable_balance.
- **fitness.py**: `score_verse()`, `score_vector()`, `DEFAULT_WEIGHTS`, `VERSE_DEFAULT_WEIGHTS`.
- **Weights**: Positive components sum to ~1.0; penalties are negative. Do not weaken `ngram_fluency` or `lexical_validity` without strong justification (anti-exploit).

### Adding a New Scorer

1. Implement `score_*(lines, scheme, ...) -> float` in `evo_rhyme/scoring/`.
2. Call from `score_verse` / `score_vector` in fitness.py.
3. Add weight to `DEFAULT_WEIGHTS` or `VERSE_DEFAULT_WEIGHTS`.
4. Add to `evolution.yaml` fitness_weights if canonical.
5. Add unit test in `tests/test_evo_rhyme/`.

### Inference Optimization for Scoring

- **Caching**: `LRUTTLCache` in fitness.py caches verse score vectors by `(scheme, text)`.
- **Batch scoring**: When possible, score multiple candidates in one pass (e.g. batch LM calls for semantic coherence).
- **Lazy loading**: Ngram index, rhyme graphs, etc. are loaded on first use; keep memory bounds.

## Core Invariants

- **Reproducibility**: Seed-controlled randomness in policy sampling (`maybe_epsilon_perturb`).
- **Fallbacks**: LM calls should degrade gracefully (empty response, RAPBOT_DISABLE_LM_REWRITER).
- **Logging**: Log model, backend, cache hits, and failures.
- **Separation**: LM layer (rewriter/proposer) is separate from evolution (mutation, emitters) and scoring (fitness).

## Quick References

- `RewriterConfig` / `ProposerConfig`: backend, model, api_base, temperature, max_tokens.
- `evolution.yaml` → `proposer` section.
- `CONTROL_REGISTRY` in experiment_controls.py defines control domains for epsilon perturbation.
- `artifacts/learned_policy.json` is the canonical learned policy output.
- `config/settings.py` DEFAULTS: `base_model_name: Qwen/Qwen2.5-14B-Instruct`.
