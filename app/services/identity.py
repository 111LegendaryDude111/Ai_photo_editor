from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image

from app.domain.errors import NoFaceDetectedError
from app.domain.schemas import BenchmarkRecord
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ArcFaceIdentityExtractor:
    """ArcFace-compatible identity extractor with optional real backend."""

    model_registry: ModelRegistry | None = None
    min_resolution: int = 64
    min_std: float = 7.5

    def _extract_with_arcface(self, image: Image.Image) -> np.ndarray | None:
        if self.model_registry is None:
            return None

        app = self.model_registry.get_arcface()
        if app is None:
            return None

        try:
            bgr = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1]
            faces = app.get(bgr)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("ArcFace inference failed, fallback extractor is used: %s", exc)
            return None

        if not faces:
            raise NoFaceDetectedError("No face detected in image")

        def face_area(face) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                return 0.0
            return float(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

        best = max(faces, key=face_area)
        embedding = getattr(best, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(best, "embedding", None)
        if embedding is None:
            raise NoFaceDetectedError("ArcFace did not return embedding")

        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-9:
            raise NoFaceDetectedError("ArcFace embedding norm is zero")
        return emb / norm

    def detect_faces(self, image: Image.Image) -> list[dict[str, float]]:
        if self.model_registry is not None:
            app = self.model_registry.get_arcface()
            if app is not None:
                try:  # pragma: no cover - runtime env specific
                    bgr = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1]
                    faces = app.get(bgr)
                except Exception:
                    faces = []

                items: list[dict[str, float]] = []
                for face in faces:
                    bbox = getattr(face, "bbox", None)
                    det_score = float(getattr(face, "det_score", 1.0))
                    if bbox is None:
                        continue
                    items.append(
                        {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3]),
                            "score": det_score,
                        }
                    )
                return items

        rgb = image.convert("RGB")
        if min(rgb.size) < self.min_resolution:
            return []
        arr = np.asarray(rgb, dtype=np.float32)
        if float(arr.std()) < self.min_std:
            return []
        w, h = rgb.size
        return [{"x1": w * 0.2, "y1": h * 0.1, "x2": w * 0.8, "y2": h * 0.8, "score": 0.5}]

    def count_faces(self, image: Image.Image) -> int:
        return len(self.detect_faces(image))

    def extract_identity_embedding(self, image: Image.Image) -> np.ndarray:
        real_embedding = self._extract_with_arcface(image)
        if real_embedding is not None:
            return real_embedding

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
