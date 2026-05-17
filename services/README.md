# Services

Легкий сервисный слой вокруг ML-пайплайна:

- `backend` — FastAPI-сервис, принимает видео и возвращает строки в формате `data/submission.csv`.
- `frontend` — Streamlit UI для загрузки видео, просмотра таблицы и скачивания CSV.

## Локальный запуск через Docker Compose

```bash
docker compose -f docker-compose.local.yml up --build
```

После старта:

- Frontend: http://localhost:8501
- Backend healthcheck: http://localhost:8000/health
- Backend API: `POST http://localhost:8000/api/v1/process-video`

## Переменные backend

- `LENTA_MODEL_PATH` — путь до `best.pt`.
- `LENTA_DEVICE` — `auto`, `cpu`, `cuda` или `mps`.
- `LENTA_CONF` — confidence threshold YOLO.
- `LENTA_IOU` — IOU threshold.
- `LENTA_IMGSZ` — размер инференса.
- `LENTA_MAX_FRAMES` — опциональный лимит кадров для быстрых smoke-тестов.
- `LENTA_MAX_UPLOAD_MB` — лимит размера загружаемого видео.

Модель не входит в backend image. В `docker-compose.local.yml` локальная директория
`./runs/tags_v1/weights` монтируется в контейнер как `/models:ro`, а backend читает
`/models/best.pt`.

Dockerfile использует multi-stage сборку:

- `builder` собирает wheelhouse из `requirements.lock.txt`;
- `runtime` устанавливает зависимости из локальных wheels и запускается от non-root пользователя `app`.

`requirements.txt` остается человеко-читаемым файлом ограничений для пересборки lock-файла.

## Продовый запуск через Docker Compose

Продовые deploy-артефакты лежат в `deploy/`. Compose не собирает образы на сервере,
а только скачивает готовые images из registry:

```bash
cp deploy/.env.prod.example deploy/.env.prod
vim deploy/.env.prod
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml pull
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml up -d
```

Минимальная подготовка сервера:

```bash
sudo mkdir -p /opt/lenta/models
sudo cp best.pt /opt/lenta/models/best.pt
```

В `.env.prod` нужно указать:

- `BACKEND_IMAGE` и `FRONTEND_IMAGE` — готовые образы из registry;
- `MODEL_PATH` — абсолютный путь до `best.pt` на сервере;
- `FRONTEND_BIND/FRONTEND_PORT` — внешний адрес Streamlit;
- `BACKEND_BIND/BACKEND_PORT` — backend по умолчанию слушает только localhost.

OCR/ML cache вынесен в named volumes (`backend-paddlex-cache`, `backend-paddleocr-cache`,
`backend-paddle-cache`, `backend-ultralytics-cache`), чтобы контейнеры быстрее переживали restart.

Подробная инструкция: `deploy/README.md`.

## Текущее поведение ML-сервиса

Backend использует `runs/tags_v1/weights/best.pt`, детектирует ценники и склеивает их в треки через ByteTrack.
Поля OCR/QR пока заполняются как `нет`, но схема ответа уже совпадает с текущим `data/submission.csv`.
