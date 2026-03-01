from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.domain.schemas import ValidationDecision
from app.services.artifact_detection import ArtifactDetectionService
from app.services.identity import ArcFaceIdentityExtractor, cosine_similarity
from app.services.prompt_adherence import PromptAdherenceService
from app.services.safety import ContentSafetyService


@dataclass
class IdentityValidator:
    extractor: ArcFaceIdentityExtractor
    hard_threshold: float = 0.85
    soft_threshold: float = 0.75

    def validate(
        self,
        reference_image: Image.Image,
        generated_image: Image.Image,
        reference_embedding: np.ndarray | None = None,
    ) -> dict[str, float | str]:
        ref = reference_embedding if reference_embedding is not None else self.extractor.extract_identity_embedding(reference_image)
        out = self.extractor.extract_identity_embedding(generated_image)
        similarity = cosine_similarity(ref, out)

        if similarity >= self.hard_threshold:
            status = "pass"
        elif similarity >= self.soft_threshold:
            status = "warning"
        else:
            status = "reject"

        return {
            "identity_similarity": similarity,
            "identity_status": status,
        }


@dataclass
class QualityGateExecutor:
    identity_validator: IdentityValidator
    clip_validator: PromptAdherenceService
    artifact_detector: ArtifactDetectionService
    safety_filter: ContentSafetyService | None = None
    clip_threshold: float = 0.8
    artifact_threshold: float = 0.3
    nsfw_threshold: float = 0.8
    enable_nsfw_filter: bool = True

    def evaluate(
        self,
        reference_image: Image.Image,
        generated_image: Image.Image,
        prompt: str,
        reference_embedding: np.ndarray | None = None,
    ) -> ValidationDecision:
        reasons: list[str] = []

        identity_result = self.identity_validator.validate(
            reference_image=reference_image,
            generated_image=generated_image,
            reference_embedding=reference_embedding,
        )
        identity_similarity = float(identity_result["identity_similarity"])
        identity_status = str(identity_result["identity_status"])

        clip_score = self.clip_validator.score(generated_image, prompt)
        artifacts = self.artifact_detector.evaluate(generated_image)
        artifact_score = float(artifacts["face_artifact_probability"])
        hand_anomaly_detected = float(artifacts.get("hand_anomaly_detected", 0.0)) > 0.5
        nsfw_score = 0.0
        if self.enable_nsfw_filter and self.safety_filter is not None:
            nsfw_score = self.safety_filter.score_nsfw(generated_image)

        status = "completed"
        if identity_status == "reject":
            status = "rejected"
            reasons.append("identity_similarity_below_threshold")
        if clip_score < self.clip_threshold:
            status = "rejected"
            reasons.append("clip_score_below_threshold")
        if artifact_score > self.artifact_threshold:
            status = "rejected"
            reasons.append("artifact_probability_above_threshold")
        if hand_anomaly_detected:
            status = "rejected"
            reasons.append("hand_anomaly_detected")
        if self.enable_nsfw_filter and nsfw_score >= self.nsfw_threshold:
            status = "rejected"
            reasons.append("nsfw_content_detected")

        return ValidationDecision(
            status=status,
            reasons=reasons,
            metrics={
                "identity_similarity": identity_similarity,
                "clip_score": clip_score,
                "artifact_score": artifact_score,
                "nsfw_score": nsfw_score,
            },
        )
