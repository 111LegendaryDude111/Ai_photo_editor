import numpy as np
from PIL import Image
import pytest

from app.domain.errors import NoFaceDetectedError
from app.services.identity import ArcFaceIdentityExtractor, cosine_similarity
from tests.conftest import make_face_like_image


def test_identity_similarity_same_person_high() -> None:
    extractor = ArcFaceIdentityExtractor()
    img = make_face_like_image()

    emb1 = extractor.extract_identity_embedding(img)
    emb2 = extractor.extract_identity_embedding(img.copy())

    assert cosine_similarity(emb1, emb2) >= 0.85


def test_identity_similarity_different_person_low() -> None:
    extractor = ArcFaceIdentityExtractor()
    img1 = make_face_like_image(hue_shift=0)
    rng = np.random.default_rng(seed=7)
    img2 = Image.fromarray(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8), mode="RGB")

    emb1 = extractor.extract_identity_embedding(img1)
    emb2 = extractor.extract_identity_embedding(img2)

    assert cosine_similarity(emb1, emb2) <= 0.6


def test_no_face_detected_for_flat_image() -> None:
    extractor = ArcFaceIdentityExtractor()
    flat = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError):
        extractor.extract_identity_embedding(Image.fromarray(flat, mode="RGB"))
