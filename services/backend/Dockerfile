# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/backend/requirements.lock.txt /build/requirements.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels -r /build/requirements.lock.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /home/app --create-home app

COPY --from=builder /wheels /wheels
COPY services/backend/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /tmp/requirements.lock.txt \
    && rm -rf /wheels

COPY services/__init__.py /app/services/__init__.py
COPY services/backend/__init__.py /app/services/backend/__init__.py
COPY services/backend/app /app/services/backend/app

RUN mkdir -p /models /data/jobs /cache/ultralytics /cache/paddlex /cache/paddleocr /cache/paddle \
    && chown -R app:app /app /home/app /data/jobs /cache

ENV PYTHONPATH=/app \
    LENTA_MODEL_PATH=/models/best.pt \
    LENTA_STORAGE_DIR=/tmp/lenta-tech-backend \
    YOLO_CONFIG_DIR=/cache/ultralytics \
    PADDLE_PDX_CACHE_HOME=/cache/paddlex \
    PADDLEOCR_HOME=/cache/paddleocr \
    PADDLE_HOME=/cache/paddle \
    HOME=/home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "services.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
