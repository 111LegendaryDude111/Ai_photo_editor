from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.domain.schemas import ControlType


@dataclass
class ControlNetService:
    """ControlNet dispatcher for depth/pose conditions."""

    def build_control_payload(self, control_type: ControlType, reference_image: Image.Image) -> dict[str, str]:
        if control_type == ControlType.NONE:
            return {"control_type": "none", "status": "skipped"}
        if control_type == ControlType.DEPTH:
            return {"control_type": "depth", "status": "applied", "model": "controlnet-depth"}
        if control_type == ControlType.POSE:
            return {"control_type": "pose", "status": "applied", "model": "controlnet-openpose"}
        return {"control_type": control_type.value, "status": "unknown"}
