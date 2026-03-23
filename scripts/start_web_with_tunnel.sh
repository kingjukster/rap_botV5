#!/bin/sh
# Start uvicorn, then localtunnel (exposes port 8000 to public URL).
# Public URL is printed to logs: docker compose logs -f web
set -e
echo "[startup] Starting uvicorn on port 8000"
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 &
sleep 3
echo "[startup] Launching localtunnel (public URL below)..."
npx --yes localtunnel --port 8000 &
wait
