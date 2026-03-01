from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.domain.schemas import ControlType
from app.services.model_registry import ModelRegistry


@dataclass
class ControlNetService:
    """ControlNet dispatcher for depth/pose conditions."""

    model_registry: ModelRegistry | None = None

    def _depth_fallback(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        depth = np.clip(np.sqrt(grad_x**2 + grad_y**2), 0, 255)
        depth = (depth / max(depth.max(), 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(depth, mode="L").convert("RGB")

    def _pose_fallback(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        canvas = np.zeros((h, w), dtype=np.uint8)
        cx = int(w * 0.5)
        cy = int(h * 0.3)
        canvas[max(0, cy - 10) : min(h, cy + 10), max(0, cx - 10) : min(w, cx + 10)] = 255
        canvas[int(h * 0.35) : int(h * 0.75), max(0, cx - 2) : min(w, cx + 2)] = 255
        canvas[int(h * 0.45) : int(h * 0.47), int(w * 0.3) : int(w * 0.7)] = 255
        return Image.fromarray(canvas, mode="L").convert("RGB")

    def build_control_payload(self, control_type: ControlType, reference_image: Image.Image) -> dict[str, object]:
        if control_type == ControlType.NONE:
            return {"control_type": "none", "status": "skipped"}

        if control_type == ControlType.DEPTH:
            control_image = self._depth_fallback(reference_image)
            model_name = "midas-fallback"
            if self.model_registry is not None:
                estimator = self.model_registry.get_depth_estimator()
                if estimator is not None:
                    try:  # pragma: no cover - runtime env specific
                        depth_output = estimator(reference_image)
                        if isinstance(depth_output, dict) and "depth" in depth_output:
                            depth = depth_output["depth"]
                            if isinstance(depth, Image.Image):
                                control_image = depth.convert("RGB")
                                model_name = "midas"
                        elif hasattr(depth_output, "convert"):
                            control_image = depth_output.convert("RGB")
                            model_name = "midas"
                    except Exception:
                        pass
            return {
                "control_type": "depth",
                "status": "applied",
                "model": model_name,
                "control_image": control_image,
            }

        if control_type == ControlType.POSE:
            control_image = self._pose_fallback(reference_image)
            model_name = "openpose-fallback"
            if self.model_registry is not None:
                estimator = self.model_registry.get_pose_estimator()
                if estimator is not None:
                    try:  # pragma: no cover - runtime env specific
                        out = estimator(reference_image)
                        if isinstance(out, Image.Image):
                            control_image = out.convert("RGB")
                            model_name = "dwpose_or_openpose"
                    except Exception:
                        pass
            return {
                "control_type": "pose",
                "status": "applied",
                "model": model_name,
                "control_image": control_image,
            }

        return {"control_type": control_type.value, "status": "unknown"}
