from __future__ import annotations

import numpy as np
from PIL import Image


class PromptAdherenceService:
    """CLIP-like proxy score. Replace with real CLIP encoder in production."""

    def score(self, image: Image.Image, prompt: str) -> float:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        entropy_proxy = min(float(arr.std()) / 96.0, 1.0)
        prompt_complexity = min(len(prompt.split()) / 16.0, 1.0)
        score = 0.55 + 0.3 * entropy_proxy + 0.15 * prompt_complexity
        return float(max(0.0, min(score, 1.0)))
