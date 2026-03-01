from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image

from app.services.model_registry import ModelRegistry
from app.services.torch_utils import get_inference_context, maybe_import_torch


@dataclass
class PromptAdherenceService:
    """CLIP score service with fallback heuristic."""

    model_registry: ModelRegistry | None = None

    def score(self, image: Image.Image, prompt: str) -> float:
        if self.model_registry is not None:
            bundle = self.model_registry.get_clip_bundle()
            if bundle is not None:
                torch = maybe_import_torch()
                if torch is not None:
                    model = bundle["model"]
                    processor = bundle["processor"]
                    with get_inference_context():  # pragma: no branch - tiny branch
                        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                        device = next(model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                        text_features = model.get_text_features(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                        )
                        image_features = torch.nn.functional.normalize(image_features, dim=-1)
                        text_features = torch.nn.functional.normalize(text_features, dim=-1)
                        raw_score = float((image_features * text_features).sum(dim=-1).item())
                        return float(max(0.0, min((raw_score + 1.0) / 2.0, 1.0)))

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        entropy_proxy = min(float(arr.std()) / 96.0, 1.0)
        prompt_complexity = min(len(prompt.split()) / 16.0, 1.0)
        score = 0.55 + 0.3 * entropy_proxy + 0.15 * prompt_complexity
        return float(max(0.0, min(score, 1.0)))
