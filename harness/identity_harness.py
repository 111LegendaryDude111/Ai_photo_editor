from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.services.identity import ArcFaceIdentityExtractor, cosine_similarity


@dataclass
class IdentityHarnessResult:
    same_person_avg: float
    different_person_avg: float
    drift: float


class IdentityHarness:
    def __init__(self, extractor: ArcFaceIdentityExtractor | None = None) -> None:
        self.extractor = extractor or ArcFaceIdentityExtractor()

    def evaluate(
        self,
        baseline_image: Image.Image,
        same_person_variants: list[Image.Image],
        different_person_images: list[Image.Image],
    ) -> IdentityHarnessResult:
        base_emb = self.extractor.extract_identity_embedding(baseline_image)

        same_scores = []
        for image in same_person_variants:
            emb = self.extractor.extract_identity_embedding(image)
            same_scores.append(cosine_similarity(base_emb, emb))

        diff_scores = []
        for image in different_person_images:
            emb = self.extractor.extract_identity_embedding(image)
            diff_scores.append(cosine_similarity(base_emb, emb))

        same_avg = sum(same_scores) / max(len(same_scores), 1)
        diff_avg = sum(diff_scores) / max(len(diff_scores), 1)
        drift = max(abs(score - same_avg) for score in same_scores) if same_scores else 0.0

        return IdentityHarnessResult(
            same_person_avg=same_avg,
            different_person_avg=diff_avg,
            drift=drift,
        )
