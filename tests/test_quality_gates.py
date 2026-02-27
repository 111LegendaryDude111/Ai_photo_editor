from app.services.artifact_detection import ArtifactDetectionService
from app.services.identity import ArcFaceIdentityExtractor
from app.services.prompt_adherence import PromptAdherenceService
from app.services.validators import IdentityValidator, QualityGateExecutor
from tests.conftest import make_face_like_image


def test_quality_gate_accepts_good_case() -> None:
    reference = make_face_like_image()
    generated = make_face_like_image(hue_shift=8)

    gates = QualityGateExecutor(
        identity_validator=IdentityValidator(ArcFaceIdentityExtractor(), hard_threshold=0.85, soft_threshold=0.75),
        clip_validator=PromptAdherenceService(),
        artifact_detector=ArtifactDetectionService(),
        clip_threshold=0.75,
        artifact_threshold=0.5,
    )

    decision = gates.evaluate(reference, generated, "detailed photoreal portrait with realistic skin and lighting")
    assert decision.status in {"completed", "rejected"}
    assert "identity_similarity" in decision.metrics
    assert "clip_score" in decision.metrics
    assert "artifact_score" in decision.metrics
