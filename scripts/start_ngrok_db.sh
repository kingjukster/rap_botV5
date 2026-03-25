#!/bin/bash
# Start an ngrok TCP tunnel to expose local MySQL (port 3307) to RunPod workers.
#
# Usage:
#   source scripts/start_ngrok_db.sh
#
# This exports RAPBOT_RUNPOD_DB_HOST and RAPBOT_RUNPOD_DB_PORT for use by
# scripts/run_runpod_campaign.py.
#
# Requires: ngrok installed and NGROK_AUTHTOKEN set (in .env or environment).

set -euo pipefail

DB_PORT="${RAPBOT_DB_PORT:-3307}"

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${NGROK_AUTHTOKEN:-}" ]; then
    echo "ERROR: NGROK_AUTHTOKEN not set. Add it to .env or export it."
    exit 1
fi

# Kill any existing ngrok process
pkill -f "ngrok tcp" 2>/dev/null || true
sleep 1

echo "Starting ngrok tunnel to localhost:${DB_PORT}..."
ngrok tcp "${DB_PORT}" --log=stdout > /tmp/ngrok_db.log 2>&1 &
NGROK_PID=$!

# Wait for tunnel to establish
sleep 4

# Parse the public URL from ngrok's local API
TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo "{}")

if echo "$TUNNEL_INFO" | python3 -c "import sys,json; json.load(sys.stdin)['tunnels'][0]" > /dev/null 2>&1; then
    eval $(echo "$TUNNEL_INFO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
url = data['tunnels'][0]['public_url'].replace('tcp://', '')
host, port = url.rsplit(':', 1)
print(f'export RAPBOT_RUNPOD_DB_HOST={host}')
print(f'export RAPBOT_RUNPOD_DB_PORT={port}')
")
    echo ""
    echo "MySQL tunnel ready:"
    echo "  Host: ${RAPBOT_RUNPOD_DB_HOST}"
    echo "  Port: ${RAPBOT_RUNPOD_DB_PORT}"
    echo "  PID:  ${NGROK_PID}"
    echo ""
    echo "Now run: python scripts/run_runpod_campaign.py"
else
    echo "ERROR: Failed to get tunnel info from ngrok API."
    echo "Check /tmp/ngrok_db.log for details."
    kill $NGROK_PID 2>/dev/null || true
    exit 1
fi
