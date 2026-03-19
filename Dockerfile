# Lightweight image for web app (no ML deps)
FROM python:3.11-slim

WORKDIR /app

# Web app dependencies only (fastapi, uvicorn, jinja2, mysql, dotenv)
RUN pip install --no-cache-dir \
    fastapi>=0.109 \
    "uvicorn[standard]>=0.27" \
    jinja2>=3.1 \
    mysql-connector-python>=8.0 \
    python-dotenv>=1.0

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
