from app.core.container import get_orchestrator
from app.core.config import get_settings
from app.domain.errors import GpuOutOfMemoryError
from app.domain.schemas import GenerationRequest
from app.services.batch_scheduler import BatchScheduler
from app.workers.celery_app import celery_app


@celery_app.task(
    name="photo_editor.generate",
    bind=True,
    autoretry_for=(GpuOutOfMemoryError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def generate_task(self, payload: dict) -> dict:
    orchestrator = get_orchestrator()
    request = GenerationRequest.model_validate(payload)
    result = orchestrator.process(request)
    return result.model_dump()


@celery_app.task(
    name="photo_editor.generate.batch",
    bind=True,
    autoretry_for=(GpuOutOfMemoryError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def generate_batch_task(self, payloads: list[dict]) -> list[dict]:
    orchestrator = get_orchestrator()
    settings = get_settings()
    scheduler = BatchScheduler(batch_size=settings.celery_batch_size)

    outputs: list[dict] = []
    for batch in scheduler.split(payloads):
        for payload in batch:
            request = GenerationRequest.model_validate(payload)
            outputs.append(orchestrator.process(request).model_dump())
    return outputs
