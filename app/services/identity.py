from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.domain.errors import NoFaceDetectedError
from app.domain.schemas import BenchmarkRecord


@dataclass
class ArcFaceIdentityExtractor:
    """ArcFace-compatible interface with lightweight fallback extractor."""

    min_resolution: int = 64
    min_std: float = 7.5

    def extract_identity_embedding(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        if min(rgb.size) < self.min_resolution:
            raise NoFaceDetectedError("Image is too small for reliable face extraction")

        arr = np.asarray(rgb.resize((128, 128)), dtype=np.float32)
        if float(arr.std()) < self.min_std:
            raise NoFaceDetectedError("No face-like signal detected in image")

        # A deterministic compact embedding fallback. Replace with ArcFace model in production.
        center = arr[16:112, 16:112, :].reshape(-1)
        chunks = np.array_split(center, 512)
        emb = np.array([chunk.mean() for chunk in chunks], dtype=np.float32)
        emb -= emb.mean()

        norm = float(np.linalg.norm(emb))
        if norm <= 1e-9:
            raise NoFaceDetectedError("Failed to compute valid embedding")
        return emb / norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def run_identity_benchmark(
    extractor: ArcFaceIdentityExtractor,
    same_person_pairs: list[tuple[Image.Image, Image.Image]],
    different_person_pairs: list[tuple[Image.Image, Image.Image]],
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []

    for idx, (img1, img2) in enumerate(same_person_pairs, start=1):
        emb1 = extractor.extract_identity_embedding(img1)
        emb2 = extractor.extract_identity_embedding(img2)
        records.append(
            BenchmarkRecord(
                pair_id=f"same-{idx}",
                is_same_person=True,
                similarity=cosine_similarity(emb1, emb2),
            )
        )

    for idx, (img1, img2) in enumerate(different_person_pairs, start=1):
        emb1 = extractor.extract_identity_embedding(img1)
        emb2 = extractor.extract_identity_embedding(img2)
        records.append(
            BenchmarkRecord(
                pair_id=f"diff-{idx}",
                is_same_person=False,
                similarity=cosine_similarity(emb1, emb2),
            )
        )

    return records
