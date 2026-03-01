from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from PIL import Image

from app.domain.errors import (
    GpuOutOfMemoryError,
    InputValidationError,
    MissingLoraError,
    NoFaceDetectedError,
    PipelineError,
)
from app.domain.schemas import GenerationRequest, GenerationResult, Metrics, ValidationDecision
from app.infra.image_io import decode_base64_image, image_to_bytes
from app.infra.metadata_repo import MetadataRepository
from app.infra.storage import StorageClient
from app.services.generation_agent import GenerationAgent
from app.services.identity import ArcFaceIdentityExtractor
from app.services.input_validation import InputValidationService
from app.services.metrics import MetricsService
from app.services.validators import QualityGateExecutor


@dataclass
class _Candidate:
    image: Image.Image
    generation_meta: dict
    decision: ValidationDecision
    attempt: int
    variant: int

    def ranking_score(self) -> float:
        identity = float(self.decision.metrics.get("identity_similarity", 0.0))
        clip = float(self.decision.metrics.get("clip_score", 0.0))
        artifact = float(self.decision.metrics.get("artifact_score", 1.0))
        nsfw = float(self.decision.metrics.get("nsfw_score", 0.0))
        status_bonus = 1.0 if self.decision.status == "completed" else 0.0
        return status_bonus + (identity * 0.5) + (clip * 0.35) + ((1.0 - artifact) * 0.1) + ((1.0 - nsfw) * 0.05)


@dataclass
class OrchestratorService:
    generation_agent: GenerationAgent
    identity_extractor: ArcFaceIdentityExtractor
    quality_gates: QualityGateExecutor
    storage: StorageClient
    metadata_repository: MetadataRepository
    metrics: MetricsService
    input_validator: InputValidationService
    max_quality_retries: int = 3
    retry_seed_increment: int = 1
    retry_lora_scale_decay: float = 0.9
    default_ab_variants: int = 2

    @staticmethod
    def _pick_best(candidates: list[_Candidate]) -> _Candidate:
        return max(candidates, key=lambda item: item.ranking_score())

    def process(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        job_id = str(uuid4())
        self.metrics.mark_request()

        try:
            reference_image = decode_base64_image(request.reference_image_b64)
            self.input_validator.validate(reference_image, request.reference_image_b64)
            identity_embedding = self.identity_extractor.extract_identity_embedding(reference_image)

            if self.identity_extractor.model_registry is not None:
                self.identity_extractor.model_registry.release_stage("arcface")

            retries = request.max_retries if request.max_retries is not None else self.max_quality_retries
            variants = request.ab_variants if request.ab_variants is not None else self.default_ab_variants
            retries = max(0, int(retries))
            variants = max(1, int(variants))

            all_candidates: list[_Candidate] = []
            retry_trace: list[dict[str, object]] = []
            base_scale = self.generation_agent.lora_manager.normalize_scale(
                request.lora_scale,
                self.generation_agent.default_lora_scale,
            )

            for attempt in range(retries + 1):
                attempt_candidates: list[_Candidate] = []
                scaled_lora = max(0.0, min(base_scale * (self.retry_lora_scale_decay**attempt), 1.5))

                for variant in range(variants):
                    effective_seed = request.seed + (attempt * self.retry_seed_increment) + variant
                    generated_image, generation_meta = self.generation_agent.generate(
                        request=request,
                        reference_image=reference_image,
                        identity_embedding=identity_embedding,
                        seed_override=effective_seed,
                        lora_scale_override=scaled_lora,
                    )

                    decision = self.quality_gates.evaluate(
                        reference_image=reference_image,
                        generated_image=generated_image,
                        prompt=request.prompt,
                        reference_embedding=identity_embedding,
                    )

                    candidate = _Candidate(
                        image=generated_image,
                        generation_meta=generation_meta,
                        decision=decision,
                        attempt=attempt,
                        variant=variant,
                    )
                    attempt_candidates.append(candidate)
                    all_candidates.append(candidate)

                best_attempt = self._pick_best(attempt_candidates)
                retry_trace.append(
                    {
                        "attempt": attempt,
                        "scaled_lora": scaled_lora,
                        "best_status": best_attempt.decision.status,
                        "best_score": best_attempt.ranking_score(),
                        "best_metrics": best_attempt.decision.metrics,
                    }
                )
                if best_attempt.decision.status == "completed":
                    break

            best = self._pick_best(all_candidates)
            decision = best.decision
            identity_similarity = float(decision.metrics.get("identity_similarity", 0.0))
            clip_score = float(decision.metrics.get("clip_score", 0.0))
            artifact_score = float(decision.metrics.get("artifact_score", 1.0))
            nsfw_score = float(decision.metrics.get("nsfw_score", 0.0))

            latency_ms = (time.perf_counter() - start) * 1000
            self.metrics.observe_latency(latency_ms / 1000.0)
            self.metrics.observe_identity_similarity(identity_similarity)
            self.metrics.observe_nsfw_score(nsfw_score)

            image_url = self.storage.save_image(job_id, image_to_bytes(best.image))
            payload = {
                "job_id": job_id,
                "status": decision.status,
                "request": request.model_dump(),
                "generation": best.generation_meta,
                "quality_loop": {
                    "attempts": retries + 1,
                    "ab_variants": variants,
                    "trace": retry_trace,
                    "selected_attempt": best.attempt,
                    "selected_variant": best.variant,
                },
                "metrics": {
                    "identity_similarity": identity_similarity,
                    "clip_score": clip_score,
                    "artifact_score": artifact_score,
                    "nsfw_score": nsfw_score,
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
                    nsfw_score=nsfw_score,
                ),
                config={
                    "edit_type": request.edit_type.value,
                    "control_type": request.control_type.value,
                    "lora_id": request.lora_id,
                    "lora_scale": request.lora_scale,
                    "ab_variants": variants,
                    "max_retries": retries,
                },
            )
        except GpuOutOfMemoryError:
            raise
        except (NoFaceDetectedError, MissingLoraError, InputValidationError, PipelineError, ValueError) as exc:
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
                    "nsfw_score": 0.0,
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
                    nsfw_score=0.0,
                ),
                config={
                    "edit_type": request.edit_type.value,
                    "control_type": request.control_type.value,
                    "lora_id": request.lora_id,
                    "lora_scale": request.lora_scale,
                },
            )
