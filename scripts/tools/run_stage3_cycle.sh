#!/bin/bash
set -euo pipefail

# Helper to orchestrate a full Stage-3 cycle:
#   1) Generation passes (in-process caching from config)
#   2) OpenAI critic scoring with score floor
#   3) Local critic retrain + consolidation
#   4) Stats/HTML report refresh
#
# Usage:
#   ./scripts/tools/run_stage3_cycle.sh [artist] [config] [critic_scores_path]
#
# Requires OPENAI_API_KEY for the scoring step.

ARTIST="${1:-mf_doom}"
CONFIG="${2:-config/rapbot.yaml}"
CRITIC_SCORES="${3:-data/critic_scores.jsonl}"

echo "[Stage3] Config      : ${CONFIG}"
echo "[Stage3] Artist      : ${ARTIST}"
echo "[Stage3] Critic file : ${CRITIC_SCORES}"

LOG_PATH=$(python - <<PY
from config.settings import load_settings
cfg = load_settings("${CONFIG}")
print(cfg.generation_log_path)
PY
)

echo "[Stage3][1/4] Generation pass (tmux recommended)..."
python scripts/pipeline/run_stage3_pipeline.py \
  --artist "${ARTIST}" \
  --config "${CONFIG}" \
  --skip_consolidate

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY not set; skip scoring. Populate '${CRITIC_SCORES}' manually before rerunning step 3."
else
  echo "[Stage3][2/4] OpenAI critic scoring (min verse_score = 0.45)..."
  python scripts/tools/score_with_openai.py \
    --config "${CONFIG}" \
    --input "${LOG_PATH}" \
    --output "${CRITIC_SCORES}" \
    --min_verse_score 0.45
fi

echo "[Stage3][3/4] Consolidate + local critic retrain..."
python scripts/pipeline/run_stage3_pipeline.py \
  --artist "${ARTIST}" \
  --config "${CONFIG}" \
  --skip_generation \
  --critic_scores "${CRITIC_SCORES}" \
  --train_local_critic

echo "[Stage3][4/4] Refresh reports..."
python scripts/tools/run_stage3_report.py --config "${CONFIG}"

echo "[DONE] Stage-3 cycle complete."
