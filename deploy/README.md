# Deploy

Здесь лежат артефакты, которые нужны для доставки готовых контейнеров на сервер.
Локальная разработка остается в корневом `docker-compose.local.yml`.

## Состав

- `docker-compose.prod.yml` — production compose без `build`, только готовые images из registry.
- `.env.prod.example` — шаблон серверного `.env.prod`.

## Подготовка сервера

```bash
sudo mkdir -p /opt/lenta/app /opt/lenta/models
sudo cp best.pt /opt/lenta/models/best.pt
```

Скопируйте на сервер папку `deploy/` в `/opt/lenta/app/deploy`.

## Настройка

```bash
cd /opt/lenta/app
cp deploy/.env.prod.example deploy/.env.prod
vim deploy/.env.prod
```

Минимально нужно заполнить:

- `BACKEND_IMAGE`
- `FRONTEND_IMAGE`
- `MODEL_PATH`

## Запуск

```bash
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml pull
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml up -d
```

## Обновление

```bash
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml pull
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml up -d
docker image prune -f
```

## Проверка

```bash
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml ps
docker compose --env-file deploy/.env.prod -f deploy/docker-compose.prod.yml logs -f --tail=100
```

## GitHub Actions

Workflow `.github/workflows/deploy.yml` запускается на каждый push в ветку `deploy`:

1. Проверяет Python-синтаксис и оба compose-файла.
2. Собирает backend/frontend images.
3. Пушит images в GHCR.
4. Копирует `deploy/` на сервер.
5. Обновляет `BACKEND_IMAGE` и `FRONTEND_IMAGE` в `deploy/.env.prod`.
6. Выполняет `docker compose pull` и `up -d`.

Нужные repository secrets:

- `DEPLOY_HOST` — адрес сервера.
- `DEPLOY_SSH_KEY` — приватный SSH-ключ для деплоя.
- `DEPLOY_USER` — пользователь на сервере, по умолчанию `deploy`.
- `DEPLOY_PORT` — SSH-порт, по умолчанию `22`.
- `DEPLOY_PATH` — путь приложения, по умолчанию `/opt/lenta/app`.
- `GHCR_TOKEN` — опционально, PAT с `read:packages`; если не задан, используется `GITHUB_TOKEN`.

У пользователя на сервере должны быть права на Docker и запись в `DEPLOY_PATH`.
