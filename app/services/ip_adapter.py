from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class IPAdapterFaceConditioner:
    default_strength: float = 0.9

    def build_conditioning(
        self,
        embedding: np.ndarray,
        reference_image: Image.Image | None = None,
        strength: float | None = None,
    ) -> dict[str, float | bool | Image.Image]:
        applied_strength = strength if strength is not None else self.default_strength
        return {
            "ip_adapter_strength": float(applied_strength),
            "embedding_magnitude": float(np.linalg.norm(embedding)),
            "has_reference_image": reference_image is not None,
            "reference_image": reference_image,
        }
