#!/bin/bash
set -e

echo "====================================="
echo "   RAP BOT STARTUP & PASS A RUNNER   "
echo "====================================="

# --- 1. Start tmux session ---
SESSION="rapbot"

echo "[INFO] Starting tmux session: $SESSION"
tmux new-session -d -s $SESSION

# --- 2. Inside tmux: create environment and install deps ---
tmux send-keys -t $SESSION "echo '[INFO] Creating Python venv...'" C-m
tmux send-keys -t $SESSION "python3 -m venv qwen_env" C-m

tmux send-keys -t $SESSION "echo '[INFO] Activating venv...'" C-m
tmux send-keys -t $SESSION "source qwen_env/bin/activate" C-m

tmux send-keys -t $SESSION "echo '[INFO] Upgrading pip...'" C-m
tmux send-keys -t $SESSION "pip install --upgrade pip" C-m

tmux send-keys -t $SESSION "echo '[INFO] Installing dependencies...'" C-m
tmux send-keys -t $SESSION "pip install torch transformers datasets peft bitsandbytes accelerate pandas numpy scikit-learn pronouncing unidecode" C-m

echo ""
echo "====================================="
echo " Startup script launched successfully "
echo " Attach with:  tmux attach -t rapbot "
echo "====================================="
