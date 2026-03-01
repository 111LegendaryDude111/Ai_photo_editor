from __future__ import annotations

from dataclasses import dataclass
from math import floor

from PIL import Image

from app.domain.errors import InputValidationError
from app.services.identity import ArcFaceIdentityExtractor


@dataclass
class InputValidationService:
    identity_extractor: ArcFaceIdentityExtractor
    max_resolution: int = 2048
    max_pixels: int = 2048 * 2048
    max_file_size_mb: float = 8.0
    min_faces: int = 1
    max_faces: int = 1

    def _validate_image_shape(self, image: Image.Image) -> None:
        width, height = image.size
        if width > self.max_resolution or height > self.max_resolution:
            raise InputValidationError(
                f"Image resolution exceeds limit {self.max_resolution}px: got {width}x{height}"
            )
        if width * height > self.max_pixels:
            raise InputValidationError(
                f"Image pixel count exceeds limit {self.max_pixels}: got {width * height}"
            )

    def _validate_size(self, approximate_bytes: int) -> None:
        max_bytes = int(self.max_file_size_mb * 1024 * 1024)
        if approximate_bytes > max_bytes:
            raise InputValidationError(
                f"Image payload size exceeds limit {self.max_file_size_mb}MB: got {round(approximate_bytes / (1024 * 1024), 2)}MB"
            )

    def _validate_faces(self, image: Image.Image) -> None:
        count = self.identity_extractor.count_faces(image)
        if count < self.min_faces:
            raise InputValidationError(f"Face count is too low: expected >= {self.min_faces}, got {count}")
        if count > self.max_faces:
            raise InputValidationError(f"Face count is too high: expected <= {self.max_faces}, got {count}")

    def validate(self, image: Image.Image, reference_image_b64: str) -> None:
        self._validate_image_shape(image)
        approx_bytes = floor((len(reference_image_b64) * 3) / 4)
        self._validate_size(approx_bytes)
        self._validate_faces(image)
