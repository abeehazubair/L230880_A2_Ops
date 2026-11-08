# VisionGuard web app / model API (Task 6.2)
# Build:  docker build -t visionguard-web:1.0 .
# The model is NOT baked in: mount it at /app/models (see docker-compose.yml).

# Slim Debian base with the same Python version used for training (3.11)
FROM python:3.11-slim

# Python/pip behaviour: no .pyc files, unbuffered logs, no pip download cache in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies FIRST: this layer is rebuilt only when requirements-serve.txt changes,
#    so editing the app code later reuses the cached (large) TensorFlow layer.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) Unprivileged user + a writable folder for the local SQLite fallback
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data /app/models \
 && chown -R appuser:appuser /app/data

# 3) Source code LAST (changes most often). Only what serving needs.
COPY src/data.py src/data.py
COPY app/ app/

# Runtime configuration (secrets such as DB password / API key are passed at run time, never baked in)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/app/models/champion \
    DATABASE_URL=sqlite:////app/data/visionguard.db \
    INFERENCE_MODE=local

USER appuser
EXPOSE 8000

# Container reports "healthy" only when /health answers 200 (model loaded + database reachable)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "app.serve"]
