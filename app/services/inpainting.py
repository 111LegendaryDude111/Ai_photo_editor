from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class InpaintingService:
    """Lightweight inpainting approximation for local edits."""

    def apply_masked_edit(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int = 42) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        m = np.asarray(mask.convert("L").resize(image.size), dtype=np.float32) / 255.0
        m = np.expand_dims(m, axis=-1)

        digest = hashlib.sha256(f"{prompt}-{seed}".encode("utf-8")).digest()
        shift = np.array([digest[0], digest[1], digest[2]], dtype=np.float32) - 127.0

        edited = np.clip(rgb + m * 0.35 * shift, 0, 255)
        return Image.fromarray(edited.astype(np.uint8), mode="RGB")
