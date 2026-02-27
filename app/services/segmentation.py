from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class AutoSegmentationService:
    """Generates coarse masks for face/hair/clothing/background."""

    def generate_masks(self, image: Image.Image) -> dict[str, Image.Image]:
        w, h = image.size
        yy, xx = np.mgrid[0:h, 0:w]

        face = (((xx - w / 2.0) ** 2) / (w * 0.2) ** 2 + ((yy - h * 0.35) ** 2) / (h * 0.2) ** 2) <= 1.0
        hair = (((xx - w / 2.0) ** 2) / (w * 0.24) ** 2 + ((yy - h * 0.20) ** 2) / (h * 0.14) ** 2) <= 1.0
        clothing = (yy > h * 0.45) & (yy < h * 0.85)
        background = ~(face | hair | clothing)

        return {
            "face": Image.fromarray((face * 255).astype(np.uint8), mode="L"),
            "hair": Image.fromarray((hair * 255).astype(np.uint8), mode="L"),
            "clothing": Image.fromarray((clothing * 255).astype(np.uint8), mode="L"),
            "background": Image.fromarray((background * 255).astype(np.uint8), mode="L"),
        }
