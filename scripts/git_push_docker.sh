#!/bin/bash
# Run git push (with LFS) inside Docker when git-lfs is not installed on host.
# Usage: ./scripts/git_push_docker.sh [git args...]
# Example: ./scripts/git_push_docker.sh
# Example: ./scripts/git_push_docker.sh origin simple_bot
# Run from repo root or any subdirectory.

set -e
# Resolve repo root (must have .git). Prefer current dir if already in repo.
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -z "$ROOT" ]] && ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -d "$ROOT/.git" ]] || { echo "Not in a Git repository."; exit 1; }
cd "$ROOT"

IMAGE="${RAPBOT_GIT_IMAGE:-rapbot-git-lfs}"

# Build image if not present
if ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Building $IMAGE..."
  docker build -t "$IMAGE" -f docker/git.Dockerfile docker/
fi

# Mount repo (use REPO_PATH if set, e.g. for remote Docker host)
MOUNT_SRC="${REPO_PATH:-$ROOT}"
[[ "$DEBUG" ]] && echo "Mounting: $MOUNT_SRC -> /app" >&2
MOUNTS=(-v "$MOUNT_SRC:/app:rw")
[[ -f "$HOME/.gitconfig" ]] && MOUNTS+=(-v "$HOME/.gitconfig:/root/.gitconfig:ro")
[[ -d "$HOME/.ssh" ]] && MOUNTS+=(-v "$HOME/.ssh:/root/.ssh:ro")

# Run git lfs install (local repo only) + push
# safe.directory avoids "dubious ownership" when repo is mounted from host
# GIT_SSH_COMMAND: accept GitHub host key on first connect (known_hosts may be missing in container)
docker run --rm "${MOUNTS[@]}" -w /app -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
  "$IMAGE" sh -c '
  git -c safe.directory=/app lfs install --local
  exec git -c safe.directory=/app push "$@"
' _ "$@"
