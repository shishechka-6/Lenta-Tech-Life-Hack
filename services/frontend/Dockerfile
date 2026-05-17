# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/frontend/requirements.lock.txt /build/requirements.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels -r /build/requirements.lock.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /home/app --create-home app

COPY --from=builder /wheels /wheels
COPY services/frontend/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /tmp/requirements.lock.txt \
    && rm -rf /wheels

COPY services/frontend/app.py /app/app.py

RUN chown -R app:app /app /home/app

ENV BACKEND_URL=http://backend:8000 \
    HOME=/home/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app

EXPOSE 8501

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
