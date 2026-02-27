from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.domain.schemas import ValidationDecision
from app.services.artifact_detection import ArtifactDetectionService
from app.services.identity import ArcFaceIdentityExtractor, cosine_similarity
from app.services.prompt_adherence import PromptAdherenceService


@dataclass
class IdentityValidator:
    extractor: ArcFaceIdentityExtractor
    hard_threshold: float = 0.85
    soft_threshold: float = 0.75

    def validate(self, reference_image: Image.Image, generated_image: Image.Image) -> dict[str, float | str]:
        ref = self.extractor.extract_identity_embedding(reference_image)
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
    clip_threshold: float = 0.8
    artifact_threshold: float = 0.3

    def evaluate(self, reference_image: Image.Image, generated_image: Image.Image, prompt: str) -> ValidationDecision:
        reasons: list[str] = []

        identity_result = self.identity_validator.validate(reference_image, generated_image)
        identity_similarity = float(identity_result["identity_similarity"])
        identity_status = str(identity_result["identity_status"])

        clip_score = self.clip_validator.score(generated_image, prompt)
        artifacts = self.artifact_detector.evaluate(generated_image)
        artifact_score = float(artifacts["face_artifact_probability"])

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

        return ValidationDecision(
            status=status,
            reasons=reasons,
            metrics={
                "identity_similarity": identity_similarity,
                "clip_score": clip_score,
                "artifact_score": artifact_score,
            },
        )
