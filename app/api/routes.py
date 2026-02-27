from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.container import get_orchestrator, get_segmentation_service, get_training_pipeline
from app.domain.schemas import (
    DatasetPrepareRequest,
    GenerationRequest,
    GenerationResult,
    LoraTrainRequest,
    SegmentationRequest,
)
from app.infra.image_io import decode_base64_image, encode_base64_image
from app.workers.celery_app import celery_app

router = APIRouter()


@router.post("/generate", response_model=GenerationResult)
def generate(payload: GenerationRequest) -> GenerationResult:
    orchestrator = get_orchestrator()
    return orchestrator.process(payload)


@router.post("/generate/async")
def generate_async(payload: GenerationRequest) -> dict[str, str]:
    try:
        task = celery_app.send_task("photo_editor.generate", args=[payload.model_dump()])
        return {"task_id": task.id, "status": "queued"}
    except Exception as exc:  # pragma: no cover - requires broker outage simulation
        return {"task_id": "", "status": f"queue_unavailable: {exc}"}


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@router.post("/datasets/prepare")
def prepare_dataset(payload: DatasetPrepareRequest) -> dict:
    pipeline = get_training_pipeline()
    return pipeline.prepare_dataset(payload)


@router.post("/lora/train")
def train_lora(payload: LoraTrainRequest) -> dict:
    pipeline = get_training_pipeline()
    return pipeline.train_lora(payload)


@router.post("/masks/auto")
def auto_masks(payload: SegmentationRequest) -> dict:
    service = get_segmentation_service()
    image = decode_base64_image(payload.reference_image_b64)
    masks = service.generate_masks(image)
    return {name: encode_base64_image(mask, format_name="PNG") for name, mask in masks.items()}
