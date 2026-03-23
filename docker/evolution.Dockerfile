# Full evolution environment (torch, transformers, ML deps)
# Use for: docker compose run evolution python scripts/run_verse_evolution.py --theme "pressure,mask"
FROM python:3.11-slim

WORKDIR /app

# Build deps for some packages (e.g. fasttext)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project (copy everything; heavy pip layer stays cached until pyproject changes)
COPY . .

# Install full project (torch, transformers, etc.)
RUN pip install --no-cache-dir -e ".[full]"

# Default: show help
CMD ["python", "scripts/run_verse_evolution.py", "--help"]
