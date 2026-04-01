# syntax=docker/dockerfile:1
# ---- build ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Torch: install GPU (cu121) version — works on CPU too, just heavier.
# For CPU-only deployments, change to: torch==2.4.1+cpu --extra-index-url .../whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    PYTHONPATH=/install/lib/python3.11/site-packages \
    pip install --prefix=/install -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY app/ ./app/

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=15s --start-period=1800s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--ws-ping-interval", "60", "--ws-ping-timeout", "60"]
