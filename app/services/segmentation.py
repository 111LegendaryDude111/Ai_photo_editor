from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.model_registry import ModelRegistry


@dataclass
class AutoSegmentationService:
    """Generates coarse masks for face/hair/clothing/background."""

    model_registry: ModelRegistry | None = None

    @staticmethod
    def _fallback_masks(image: Image.Image) -> dict[str, Image.Image]:
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

    def _face_mask_from_parser(self, image: Image.Image) -> Image.Image | None:
        if self.model_registry is None:
            return None
        parser = self.model_registry.get_face_parser()
        if parser is None:
            return None

        try:  # pragma: no cover - runtime env specific
            segments = parser(image)
        except Exception:
            return None

        if not isinstance(segments, list):
            return None

        face_keywords = {"face", "skin", "nose", "mouth", "eye", "brow"}
        mask_acc = np.zeros((image.height, image.width), dtype=np.float32)
        for segment in segments:
            label = str(segment.get("label", "")).lower()
            if not any(key in label for key in face_keywords):
                continue
            segment_mask = segment.get("mask")
            if segment_mask is None:
                continue
            if isinstance(segment_mask, Image.Image):
                arr = np.asarray(segment_mask.resize(image.size).convert("L"), dtype=np.float32) / 255.0
                mask_acc = np.maximum(mask_acc, arr)

        if mask_acc.max() <= 0.01:
            return None
        return Image.fromarray((mask_acc * 255).astype(np.uint8), mode="L")

    def _refine_with_sam(self, image: Image.Image, base_mask: Image.Image) -> Image.Image:
        if self.model_registry is None:
            return base_mask
        predictor = self.model_registry.get_sam_predictor()
        if predictor is None:
            return base_mask

        arr = np.asarray(base_mask.convert("L"), dtype=np.uint8)
        ys, xs = np.where(arr > 0)
        if len(xs) == 0 or len(ys) == 0:
            return base_mask

        try:  # pragma: no cover - runtime env specific
            image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
            predictor.set_image(image_np)
            box = np.array([[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]], dtype=np.float32)
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            if masks is None or len(masks) == 0:
                return base_mask
            refined = (masks[0].astype(np.uint8) * 255).astype(np.uint8)
            return Image.fromarray(refined, mode="L")
        except Exception:
            return base_mask

    def build_face_lock_mask(self, image: Image.Image) -> Image.Image:
        parser_mask = self._face_mask_from_parser(image)
        if parser_mask is None:
            parser_mask = self._fallback_masks(image)["face"]
        return self._refine_with_sam(image, parser_mask)

    def generate_masks(self, image: Image.Image) -> dict[str, Image.Image]:
        masks = self._fallback_masks(image)
        face_lock = self.build_face_lock_mask(image)
        masks["face"] = face_lock
        return masks
