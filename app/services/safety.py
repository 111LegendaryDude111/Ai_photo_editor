from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.model_registry import ModelRegistry


@dataclass
class ContentSafetyService:
    model_registry: ModelRegistry | None = None

    def score_nsfw(self, image: Image.Image) -> float:
        if self.model_registry is not None:
            classifier = self.model_registry.get_nsfw_classifier()
            if classifier is not None:
                try:  # pragma: no cover - runtime env specific
                    predictions = classifier(image)
                    if isinstance(predictions, list):
                        best = 0.0
                        for item in predictions:
                            label = str(item.get("label", "")).lower()
                            score = float(item.get("score", 0.0))
                            if any(key in label for key in ("nsfw", "porn", "sexual", "explicit", "adult")):
                                best = max(best, score)
                        return max(0.0, min(best, 1.0))
                except Exception:
                    pass

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        skin_like = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r - g) > 15)
        skin_ratio = float(skin_like.mean())
        contrast = float(arr.std() / 255.0)
        heuristic = max(0.0, min((skin_ratio * 1.6) + (0.2 * contrast), 1.0))
        return heuristic
