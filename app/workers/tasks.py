from app.core.container import get_orchestrator
from app.domain.schemas import GenerationRequest
from app.workers.celery_app import celery_app


@celery_app.task(name="photo_editor.generate")
def generate_task(payload: dict) -> dict:
    orchestrator = get_orchestrator()
    request = GenerationRequest.model_validate(payload)
    result = orchestrator.process(request)
    return result.model_dump()
