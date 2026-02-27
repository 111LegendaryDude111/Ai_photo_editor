from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SDXLGeneratorService:
    """Deterministic placeholder for SDXL base+refiner stack."""

    def generate(
        self,
        reference_image: Image.Image,
        prompt: str,
        negative_prompt: str | None,
        cfg: float,
        steps: int,
        seed: int,
        refiner_strength: float,
        conditioning: dict[str, float] | None = None,
    ) -> Image.Image:
        arr = np.asarray(reference_image.convert("RGB"), dtype=np.float32)

        token = f"{prompt}|{negative_prompt or ''}|{seed}|{cfg}|{steps}|{refiner_strength}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()

        alpha = 1.0 + (digest[3] / 255.0 - 0.5) * 0.18
        shift = np.array([digest[0], digest[1], digest[2]], dtype=np.float32) - 127.0

        if conditioning:
            alpha += float(conditioning.get("ip_adapter_strength", 0.0)) * 0.05

        transformed = np.clip(arr * alpha + shift * 0.15, 0, 255)
        return Image.fromarray(transformed.astype(np.uint8), mode="RGB")
