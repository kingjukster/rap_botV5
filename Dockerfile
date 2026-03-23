# Lightweight image for web app (no ML deps)
FROM python:3.11-slim

WORKDIR /app

# Node.js for localtunnel (npx fallback) + ngrok for stable public URL
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm curl \
    && curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
       | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
       | tee /etc/apt/sources.list.d/ngrok.list \
    && apt-get update && apt-get install -y --no-install-recommends ngrok \
    && rm -rf /var/lib/apt/lists/*

# Web app dependencies only (fastapi, uvicorn, jinja2, mysql, dotenv, docker for spawning evolution)
RUN pip install --no-cache-dir \
    fastapi>=0.109 \
    "uvicorn[standard]>=0.27" \
    jinja2>=3.1 \
    mysql-connector-python>=8.0 \
    python-dotenv>=1.0 \
    docker>=6.0

COPY . /app
RUN chmod +x /app/scripts/start_web_with_tunnel.sh

EXPOSE 8000

CMD ["/app/scripts/start_web_with_tunnel.sh"]
