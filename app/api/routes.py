from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import get_settings
from app.core.rate_limit import enforce_rate_limit
from app.core.security import AuthContext, require_auth
from app.core.container import get_orchestrator, get_segmentation_service, get_training_pipeline
from app.domain.errors import GpuOutOfMemoryError
from app.domain.schemas import (
    DatasetPrepareRequest,
    GenerationRequest,
    GenerationResult,
    LoraTrainRequest,
    Metrics,
    SegmentationRequest,
)
from app.infra.image_io import decode_base64_image, encode_base64_image
from app.services.batch_scheduler import BatchScheduler
from app.workers.celery_app import celery_app

router = APIRouter()


@router.post("/generate", response_model=GenerationResult)
def generate(
    payload: GenerationRequest,
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> GenerationResult:
    orchestrator = get_orchestrator()
    try:
        return orchestrator.process(payload)
    except GpuOutOfMemoryError as exc:
        try:
            task = celery_app.send_task("photo_editor.generate", args=[payload.model_dump()])
            return GenerationResult(
                status="processing",
                rejection_reason=f"gpu_oom_queued_retry: {exc}",
                config={"task_id": task.id},
                metrics=Metrics(
                    identity_similarity=0.0,
                    clip_score=0.0,
                    artifact_score=1.0,
                    latency_ms=0.0,
                    nsfw_score=0.0,
                ),
            )
        except Exception as queue_exc:
            return GenerationResult(
                status="rejected",
                rejection_reason=f"gpu_oom_queue_failed: {queue_exc}",
                metrics=Metrics(
                    identity_similarity=0.0,
                    clip_score=0.0,
                    artifact_score=1.0,
                    latency_ms=0.0,
                    nsfw_score=0.0,
                ),
            )


@router.post("/generate/async")
def generate_async(
    payload: GenerationRequest,
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> dict[str, str]:
    try:
        task = celery_app.send_task("photo_editor.generate", args=[payload.model_dump()])
        return {"task_id": task.id, "status": "queued"}
    except Exception as exc:  # pragma: no cover - requires broker outage simulation
        return {"task_id": "", "status": f"queue_unavailable: {exc}"}


@router.post("/generate/async/batch")
def generate_async_batch(
    payloads: list[GenerationRequest],
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> dict[str, object]:
    settings = get_settings()
    scheduler = BatchScheduler(batch_size=settings.celery_batch_size)

    task_ids: list[str] = []
    try:
        for batch in scheduler.split(payloads):
            batch_payload = [item.model_dump() for item in batch]
            task = celery_app.send_task("photo_editor.generate.batch", args=[batch_payload])
            task_ids.append(task.id)
    except Exception as exc:  # pragma: no cover - requires broker outage simulation
        return {"task_ids": task_ids, "status": f"queue_unavailable: {exc}"}
    return {"task_ids": task_ids, "status": "queued", "batches": len(task_ids)}


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@router.post("/datasets/prepare")
def prepare_dataset(
    payload: DatasetPrepareRequest,
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> dict:
    pipeline = get_training_pipeline()
    return pipeline.prepare_dataset(payload)


@router.post("/lora/train")
def train_lora(
    payload: LoraTrainRequest,
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> dict:
    pipeline = get_training_pipeline()
    return pipeline.train_lora(payload)


@router.post("/masks/auto")
def auto_masks(
    payload: SegmentationRequest,
    _: AuthContext = Depends(require_auth),
    __: None = Depends(enforce_rate_limit),
) -> dict:
    service = get_segmentation_service()
    image = decode_base64_image(payload.reference_image_b64)
    masks = service.generate_masks(image)
    return {name: encode_base64_image(mask, format_name="PNG") for name, mask in masks.items()}
