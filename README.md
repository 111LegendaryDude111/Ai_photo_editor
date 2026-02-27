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
  - `GET /metrics` (Prometheus)
  - `GET /healthz`

## Quick Start

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

## Run Tests

```bash
uv run pytest -q
```

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
