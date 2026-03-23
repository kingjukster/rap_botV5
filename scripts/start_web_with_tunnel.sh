#!/bin/sh
# Start uvicorn, then a public tunnel (ngrok or localtunnel).
# For a STABLE URL from any network: set NGROK_AUTHTOKEN in .env (free at ngrok.com).
# Public URL is printed to logs: docker compose logs -f web
set -e
echo "[startup] Starting uvicorn on port 8000"
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 &
sleep 3
if [ -n "${NGROK_AUTHTOKEN}" ]; then
  echo "[startup] Launching ngrok (stable public URL below)..."
  exec ngrok http 8000
else
  echo "[startup] Launching localtunnel (URL changes on restart)..."
  echo "[startup] For a stable URL, add NGROK_AUTHTOKEN to .env (free at ngrok.com)"
  npx --yes localtunnel --port 8000 &
  wait
fi
