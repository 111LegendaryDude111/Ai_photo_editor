from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from app.domain.errors import MissingLoraError, NoFaceDetectedError, PipelineError
from app.domain.schemas import GenerationRequest, GenerationResult, Metrics
from app.infra.image_io import decode_base64_image, image_to_bytes
from app.infra.metadata_repo import MetadataRepository
from app.infra.storage import StorageClient
from app.services.generation_agent import GenerationAgent
from app.services.identity import ArcFaceIdentityExtractor
from app.services.metrics import MetricsService
from app.services.validators import QualityGateExecutor


@dataclass
class OrchestratorService:
    generation_agent: GenerationAgent
    identity_extractor: ArcFaceIdentityExtractor
    quality_gates: QualityGateExecutor
    storage: StorageClient
    metadata_repository: MetadataRepository
    metrics: MetricsService

    def process(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        job_id = str(uuid4())
        self.metrics.mark_request()

        try:
            reference_image = decode_base64_image(request.reference_image_b64)
            identity_embedding = self.identity_extractor.extract_identity_embedding(reference_image)

            generated_image, generation_meta = self.generation_agent.generate(
                request=request,
                reference_image=reference_image,
                identity_embedding=identity_embedding,
            )

            decision = self.quality_gates.evaluate(reference_image, generated_image, request.prompt)
            identity_similarity = float(decision.metrics.get("identity_similarity", 0.0))
            clip_score = float(decision.metrics.get("clip_score", 0.0))
            artifact_score = float(decision.metrics.get("artifact_score", 1.0))

            latency_ms = (time.perf_counter() - start) * 1000
            self.metrics.observe_latency(latency_ms / 1000.0)
            self.metrics.observe_identity_similarity(identity_similarity)

            image_url = self.storage.save_image(job_id, image_to_bytes(generated_image))
            payload = {
                "job_id": job_id,
                "status": decision.status,
                "request": request.model_dump(),
                "generation": generation_meta,
                "metrics": {
                    "identity_similarity": identity_similarity,
                    "clip_score": clip_score,
                    "artifact_score": artifact_score,
                    "latency_ms": latency_ms,
                },
                "rejection_reasons": decision.reasons,
            }
            metadata_url = self.metadata_repository.save_metadata(job_id, payload)

            if decision.status == "rejected":
                self.metrics.mark_rejection()

            return GenerationResult(
                job_id=job_id,
                status=decision.status,
                image_url=image_url,
                metadata_url=metadata_url,
                rejection_reason=", ".join(decision.reasons) if decision.reasons else None,
                metrics=Metrics(
                    identity_similarity=identity_similarity,
                    clip_score=clip_score,
                    artifact_score=artifact_score,
                    latency_ms=latency_ms,
                ),
                config={
                    "edit_type": request.edit_type.value,
                    "control_type": request.control_type.value,
                    "lora_id": request.lora_id,
                    "lora_scale": request.lora_scale,
                },
            )
        except (NoFaceDetectedError, MissingLoraError, PipelineError, ValueError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self.metrics.observe_latency(latency_ms / 1000.0)
            self.metrics.mark_rejection()

            payload = {
                "job_id": job_id,
                "status": "rejected",
                "request": request.model_dump(),
                "error": str(exc),
                "metrics": {
                    "identity_similarity": 0.0,
                    "clip_score": 0.0,
                    "artifact_score": 1.0,
                    "latency_ms": latency_ms,
                },
            }
            metadata_url = self.metadata_repository.save_metadata(job_id, payload)
            return GenerationResult(
                job_id=job_id,
                status="rejected",
                metadata_url=metadata_url,
                rejection_reason=str(exc),
                metrics=Metrics(
                    identity_similarity=0.0,
                    clip_score=0.0,
                    artifact_score=1.0,
                    latency_ms=latency_ms,
                ),
                config={
                    "edit_type": request.edit_type.value,
                    "control_type": request.control_type.value,
                    "lora_id": request.lora_id,
                    "lora_scale": request.lora_scale,
                },
            )
