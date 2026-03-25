#!/usr/bin/env sh
# Apply DB schema (init_db.py) and run DB-related unit tests inside Docker.
# Requires: docker compose, services: mysql (healthy). Uses root for CREATE DATABASE.
#
# Usage (from repo root):
#   chmod +x scripts/docker_verify_db.sh
#   ./scripts/docker_verify_db.sh
#
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

docker compose up -d mysql
docker compose run --rm \
  -e RAPBOT_DB_HOST=mysql \
  -e RAPBOT_DB_PORT=3306 \
  -e RAPBOT_DB_USER=root \
  -e RAPBOT_DB_PASSWORD=rootpass \
  -e RAPBOT_DB_NAME=rapbot \
  web python scripts/init_db.py

docker compose --profile evolution run --rm evolution sh -c \
  "pip install -q pytest && pytest tests/test_evo_rhyme/test_db.py tests/test_webapp/test_run_service.py -q --tb=short"

echo "OK: schema applied and tests passed."
