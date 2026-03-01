# Identity-Preserved Photo Editing System

Production scaffold aligned with `PDR.md`, `PDR-tasks.md`, and `AGENTS.md`.

## Implemented Scope

- Phase 1 baseline service scaffold:
  - `POST /generate`
  - ArcFace-compatible identity extraction interface
  - IP-Adapter and ControlNet integration points
- Phase 2 scaffold:
  - `POST /datasets/prepare`
  - `POST /lora/train`
  - Dynamic LoRA loading in generation pipeline
- Phase 3 scaffold:
  - `POST /masks/auto`
  - Inpainting path via `edit_type=inpaint`
- Phase 4 quality gates:
  - Identity validator
  - Prompt adherence score
  - Artifact detector
  - Rejection reasons + metrics logging
- Phase 5 infra scaffold:
  - `POST /generate/async` (Celery)
  - `POST /generate/async/batch` (Celery batch scheduling)
  - `GET /metrics` (Prometheus)
  - `GET /healthz`
- Production hardening:
  - JWT or API-key auth (configurable)
  - Per-user and per-IP rate limiting
  - Input validation (resolution / file size / face count)
  - Graceful OOM degradation (queue + retry path)
- Quality loop:
  - A/B candidate generation (`ab_variants`)
  - Auto retry (`max_retries`) with seed increment + LoRA strength decay
  - Face lock mask during inpaint
  - NSFW safety filter in quality gates

## Quick Start

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

## Run Tests

```bash
uv run pytest -q
```

## Real Models (Priority 1 stack)

Install optional model dependencies:

```bash
uv sync --extra dev --extra models
```

Enable real backends in `.env`:

```bash
PHOTO_EDITOR_ENABLE_REAL_MODELS=true
PHOTO_EDITOR_ARCFACE_MODEL_NAME=buffalo_l
PHOTO_EDITOR_SDXL_BASE_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
PHOTO_EDITOR_SDXL_REFINER_MODEL_ID=stabilityai/stable-diffusion-xl-refiner-1.0
PHOTO_EDITOR_IP_ADAPTER_REPO=h94/IP-Adapter-FaceID
PHOTO_EDITOR_IP_ADAPTER_WEIGHT_NAME=ip-adapter-faceid-plusv2_sdxl.bin
PHOTO_EDITOR_CLIP_MODEL_ID=openai/clip-vit-large-patch14
```

Notes:

- Services use lazy model init through `ModelRegistry`.
- If heavy dependencies or checkpoints are unavailable, code falls back to deterministic placeholder logic without changing API contracts.
- `PHOTO_EDITOR_ENABLE_CPU_OFFLOAD=true` enables `enable_model_cpu_offload()` where supported.

## Dependency Management

- Source of truth: `pyproject.toml`
- Package manager: `uv`
- `requirements.txt` is not used

## API Contracts

### `POST /generate`

```json
{
  "reference_image_b64": "...",
  "prompt": "cinematic portrait ...",
  "negative_prompt": "distorted face",
  "edit_type": "img2img",
  "control_type": "depth",
  "lora_id": "optional",
  "lora_scale": 0.8
}
```

### `POST /datasets/prepare`

```json
{
  "dataset_id": "person-001",
  "image_paths": ["/abs/path/img1.png", "/abs/path/img2.png"],
  "regularization_paths": ["/abs/path/reg1.png"]
}
```

### `POST /lora/train`

```json
{
  "dataset_id": "person-001",
  "person_token": "<person_token>",
  "rank": 16,
  "learning_rate": 0.0001,
  "epochs": 12
}
```

## Notes

- Current generation/identity/CLIP/artifact models are deterministic placeholders with production-compatible interfaces.
- Replace service internals with real SDXL, ArcFace, CLIP, ControlNet, and detector backends without changing API contracts.
- New code already includes optional real-model backends (ArcFace/SDXL/ControlNet/CLIP/Segmentation/SAM) with fallback mode.

## Auth & Rate Limits

- Set `PHOTO_EDITOR_AUTH_ENABLED=true` to enforce auth.
- Accepts either `Authorization: Bearer <jwt>` or `X-API-Key: <key>`.
- API keys are configured by `PHOTO_EDITOR_API_KEYS` (comma-separated).
- Rate limits are configurable via:
  - `PHOTO_EDITOR_RATE_LIMIT_PER_USER_PER_MINUTE`
  - `PHOTO_EDITOR_RATE_LIMIT_PER_IP_PER_MINUTE`

## Monitoring

- Prometheus metrics endpoint: `GET /metrics`
- Alert rules: `deploy/monitoring/alert_rules.yml`
- Alertmanager template: `deploy/monitoring/alertmanager.yml`
- Grafana dashboard: `deploy/monitoring/grafana-dashboard.json`

## Harness Extensions

- KPI gate harness: `harness/kpi_harness.py`
- A/B regression harness: `harness/ab_regression_harness.py`
- Concurrent load harness: `harness/load_harness.py`

## Dedicated Server Deployment

### Option A: Docker Compose (API + Worker + Redis + MinIO + Postgres)

```bash
cd deploy
cp env.docker.example env.docker
# edit env.docker: MINIO_ROOT_PASSWORD, POSTGRES_PASSWORD
docker compose up -d --build
```

Check:

```bash
curl http://127.0.0.1:8000/healthz
```

Files:

- Compose: `deploy/docker-compose.yml`
- GPU override (real models): `deploy/docker-compose.gpu.yml`
- App image: `deploy/Dockerfile`
- Env template: `deploy/env.docker.example`
- Full remote server runbook: `deploy/REMOTE_SERVER_GUIDE.md`

### Option B: systemd on host

Assume project path: `/opt/ai-photo-editor`

```bash
cd /opt/ai-photo-editor
uv sync --extra dev
cp .env.example .env
```

Install units:

```bash
sudo cp deploy/systemd/photo-editor-api.service /etc/systemd/system/
sudo cp deploy/systemd/photo-editor-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now photo-editor-api
sudo systemctl enable --now photo-editor-worker
```

Check logs:

```bash
sudo journalctl -u photo-editor-api -f
sudo journalctl -u photo-editor-worker -f
```

### Nginx + TLS (Let's Encrypt)

1. HTTP config:

```bash
sudo cp deploy/nginx/photo-editor-http.conf /etc/nginx/sites-available/photo-editor.conf
sudo ln -s /etc/nginx/sites-available/photo-editor.conf /etc/nginx/sites-enabled/photo-editor.conf
sudo nginx -t && sudo systemctl reload nginx
```

2. Issue certificate:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d example.com
```

3. Switch to HTTPS config:

```bash
sudo cp deploy/nginx/photo-editor-https.conf /etc/nginx/sites-available/photo-editor.conf
sudo nginx -t && sudo systemctl reload nginx
```

Remember to replace `example.com` in nginx files before reload.
