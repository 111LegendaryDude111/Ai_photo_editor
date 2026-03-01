import pytest

from app.domain.errors import InputValidationError
from app.services.identity import ArcFaceIdentityExtractor
from app.services.input_validation import InputValidationService
from tests.conftest import image_to_b64, make_face_like_image


def test_input_validation_accepts_reasonable_face_image() -> None:
    extractor = ArcFaceIdentityExtractor()
    validator = InputValidationService(
        identity_extractor=extractor,
        max_resolution=1024,
        max_pixels=1024 * 1024,
        max_file_size_mb=8.0,
        min_faces=1,
        max_faces=1,
    )
    image = make_face_like_image(width=256, height=256)
    validator.validate(image, image_to_b64(image))


def test_input_validation_rejects_large_resolution() -> None:
    extractor = ArcFaceIdentityExtractor()
    validator = InputValidationService(
        identity_extractor=extractor,
        max_resolution=256,
        max_pixels=256 * 256,
        max_file_size_mb=8.0,
        min_faces=1,
        max_faces=1,
    )
    image = make_face_like_image(width=512, height=512)
    with pytest.raises(InputValidationError):
        validator.validate(image, image_to_b64(image))


def test_input_validation_rejects_multiple_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArcFaceIdentityExtractor()
    validator = InputValidationService(
        identity_extractor=extractor,
        max_resolution=1024,
        max_pixels=1024 * 1024,
        max_file_size_mb=8.0,
        min_faces=1,
        max_faces=1,
    )
    image = make_face_like_image()

    monkeypatch.setattr(extractor, "count_faces", lambda _: 2)
    with pytest.raises(InputValidationError):
        validator.validate(image, image_to_b64(image))
