# Deployment Guide: Remote Server

Пошаговый гайд для запуска `Identity-Preserved Photo Editing System` на удаленном сервере.

## 1. Минимальные требования

- ОС: Ubuntu 22.04/24.04 LTS
- CPU/RAM: от 8 vCPU / 16 GB RAM
- Диск: от 80 GB SSD
- Для real-model режима: NVIDIA GPU (24 GB VRAM+ рекомендуется)
- DNS: домен, указывающий на сервер (если нужен HTTPS)

## 2. Подготовка сервера

Подключитесь по SSH:

```bash
ssh <user>@<server-ip>
```

Обновите систему:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ca-certificates gnupg lsb-release
```

Откройте фаервол:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## 3. Установка Docker + Compose

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Перелогиньтесь (или `newgrp docker`), затем проверьте:

```bash
docker --version
docker compose version
```

## 4. (Опционально) GPU runtime для Docker

Если запускаете реальные модели на GPU:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 5. Клонирование проекта

```bash
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
git clone <YOUR_REPO_URL> ai-photo-editor
cd ai-photo-editor/deploy
```

## 6. Настройка переменных окружения

```bash
cp env.docker.example env.docker
```

Минимально измените в `env.docker`:

- `MINIO_ROOT_PASSWORD`
- `POSTGRES_PASSWORD`
- `PHOTO_EDITOR_MINIO_SECRET_KEY`
- `PHOTO_EDITOR_JWT_SECRET`
- `PHOTO_EDITOR_API_KEYS` (если включаете auth)

Для продакшена рекомендуется:

- `PHOTO_EDITOR_AUTH_ENABLED=true`
- `PHOTO_EDITOR_RATE_LIMIT_ENABLED=true`
- `PHOTO_EDITOR_ENABLE_NSFW_FILTER=true`

## 7. Запуск сервиса (CPU базовый режим)

```bash
docker compose up -d --build
docker compose ps
```

Проверка:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/metrics | head
```

## 8. Запуск сервиса (GPU + реальные модели)

Используйте основной compose + GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Что делает override:

- ставит `UV_EXTRAS=models` при сборке
- включает `PHOTO_EDITOR_ENABLE_REAL_MODELS=true`
- включает `gpus: all` для `api` и `worker`

Проверка логов:

```bash
docker compose logs -f api
docker compose logs -f worker
```

## 9. Nginx + HTTPS (Let’s Encrypt)

Установите Nginx и Certbot:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

Скопируйте HTTP-конфиг:

```bash
sudo cp /opt/ai-photo-editor/deploy/nginx/photo-editor-http.conf /etc/nginx/sites-available/photo-editor.conf
sudo sed -i 's/example.com/<YOUR_DOMAIN>/g' /etc/nginx/sites-available/photo-editor.conf
sudo ln -sf /etc/nginx/sites-available/photo-editor.conf /etc/nginx/sites-enabled/photo-editor.conf
sudo nginx -t && sudo systemctl reload nginx
```

Выпустите сертификат:

```bash
sudo certbot --nginx -d <YOUR_DOMAIN>
```

Переключитесь на HTTPS-конфиг:

```bash
sudo cp /opt/ai-photo-editor/deploy/nginx/photo-editor-https.conf /etc/nginx/sites-available/photo-editor.conf
sudo sed -i 's/example.com/<YOUR_DOMAIN>/g' /etc/nginx/sites-available/photo-editor.conf
sudo nginx -t && sudo systemctl reload nginx
```

Проверка:

```bash
curl https://<YOUR_DOMAIN>/healthz
```

## 10. Обновление версии

```bash
cd /opt/ai-photo-editor
git pull
cd deploy
docker compose down
docker compose up -d --build
```

Для GPU-режима:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## 11. Диагностика и полезные команды

Статус контейнеров:

```bash
docker compose ps
```

Логи:

```bash
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f redis
docker compose logs -f postgres
docker compose logs -f minio
```

Проверка Celery:

```bash
docker compose exec worker uv run celery -A app.workers.celery_app:celery_app inspect ping
```

Проверка MinIO:

- API: `http://<server-ip>:9000`
- Console: `http://<server-ip>:9001`

## 12. Monitoring / Alerts

В репозитории уже добавлены:

- `deploy/monitoring/alert_rules.yml`
- `deploy/monitoring/alertmanager.yml`
- `deploy/monitoring/grafana-dashboard.json`

Их можно подключить в существующий Prometheus/Grafana стек:

- scrape target: `http://<api-host>:8000/metrics`
- импортируйте dashboard JSON в Grafana
- загрузите alert rules в Prometheus/Alertmanager

## 13. Security checklist для продакшена

- Не публикуйте `9000/9001` наружу без необходимости
- Ограничьте доступ к `/metrics` (уже ограничен в nginx конфиге)
- Включите `PHOTO_EDITOR_AUTH_ENABLED=true`
- Задайте сильные значения для `PHOTO_EDITOR_JWT_SECRET` и API keys
- Храните секреты в vault/secret manager, а не в git
- Регулярно обновляйте образ (`apt`, `docker`, зависимости)
