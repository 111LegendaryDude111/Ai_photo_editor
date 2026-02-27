from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.services.artifact_detection import ArtifactDetectionService
from app.services.prompt_adherence import PromptAdherenceService


@dataclass
class QualityHarnessResult:
    clip_score: float
    artifact_score: float


class QualityHarness:
    def __init__(self) -> None:
        self.clip = PromptAdherenceService()
        self.artifact = ArtifactDetectionService()

    def evaluate(self, image: Image.Image, prompt: str) -> QualityHarnessResult:
        clip_score = self.clip.score(image, prompt)
        artifact_score = self.artifact.evaluate(image)["face_artifact_probability"]
        return QualityHarnessResult(clip_score=clip_score, artifact_score=artifact_score)
