# RunPod Serverless evolution worker.
# Build: docker build -f docker/runpod.Dockerfile --build-context ignore=docker/runpod.dockerignore -t YOUR_DOCKERHUB/rapbot-evo-worker:latest .
# Or:    DOCKER_BUILDKIT=1 docker build -f docker/runpod.Dockerfile -t YOUR_DOCKERHUB/rapbot-evo-worker:latest .
# Push:  docker push YOUR_DOCKERHUB/rapbot-evo-worker:latest
#
# NOTE: Use docker/runpod.dockerignore to control what's included.
# Copy it to .dockerignore before building if the default one excludes data/:
#   cp docker/runpod.dockerignore .dockerignore && docker build ... && git checkout .dockerignore
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Ensure data files needed for init are present
RUN mkdir -p /app/data/evo_rhyme /app/artifacts

RUN pip install --no-cache-dir -e ".[full]" runpod

CMD ["python", "docker/runpod_handler.py"]
