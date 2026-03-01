from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.domain.errors import InvalidMaskError
from app.domain.schemas import GenerationRequest
from app.infra.image_io import decode_base64_image
from app.services.controlnet import ControlNetService
from app.services.inpainting import InpaintingService
from app.services.ip_adapter import IPAdapterFaceConditioner
from app.services.lora import LoraManager
from app.services.sdxl_generator import SDXLGeneratorService
from app.services.segmentation import AutoSegmentationService


@dataclass
class GenerationAgent:
    sdxl: SDXLGeneratorService
    ip_adapter: IPAdapterFaceConditioner
    controlnet: ControlNetService
    inpainting: InpaintingService
    lora_manager: LoraManager
    segmentation: AutoSegmentationService | None
    default_cfg: float
    default_steps: int
    default_refiner_strength: float
    default_lora_scale: float
    enable_face_lock_mask: bool = True

    def _apply_face_lock(self, edit_mask: Image.Image, reference_image: Image.Image) -> Image.Image:
        if not self.enable_face_lock_mask or self.segmentation is None:
            return edit_mask

        face_mask = self.segmentation.build_face_lock_mask(reference_image).resize(edit_mask.size).convert("L")
        editable = np.asarray(edit_mask.convert("L"), dtype=np.float32) / 255.0
        protected_face = np.asarray(face_mask, dtype=np.float32) / 255.0
        # Keep user-editable regions but automatically protect face pixels.
        safe_mask = np.clip(editable * (1.0 - protected_face), 0.0, 1.0)
        return Image.fromarray((safe_mask * 255).astype(np.uint8), mode="L")

    def generate(
        self,
        request: GenerationRequest,
        reference_image: Image.Image,
        identity_embedding,
        seed_override: int | None = None,
        lora_scale_override: float | None = None,
    ) -> tuple[Image.Image, dict]:
        lora_path = self.lora_manager.resolve(request.lora_id)
        base_lora_scale = lora_scale_override if lora_scale_override is not None else request.lora_scale
        lora_scale = self.lora_manager.normalize_scale(base_lora_scale, self.default_lora_scale)
        effective_seed = int(seed_override if seed_override is not None else request.seed)

        conditioning = self.ip_adapter.build_conditioning(identity_embedding, reference_image=reference_image, strength=lora_scale)
        control_payload = self.controlnet.build_control_payload(request.control_type, reference_image)

        generated = self.sdxl.generate(
            reference_image=reference_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            cfg=self.default_cfg,
            steps=self.default_steps,
            seed=effective_seed,
            refiner_strength=self.default_refiner_strength,
            conditioning=conditioning,
            control_type=request.control_type,
            control_payload=control_payload,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        if request.edit_type.value == "inpaint":
            if not request.edit_mask_b64:
                raise InvalidMaskError("inpaint request requires mask")
            mask = decode_base64_image(request.edit_mask_b64).convert("L")
            if mask.size != generated.size:
                raise InvalidMaskError("mask size must match generated image size")
            safe_mask = self._apply_face_lock(mask, reference_image=reference_image)
            generated = self.inpainting.apply_masked_edit(generated, safe_mask, request.prompt, effective_seed)

        control_meta = {k: v for k, v in control_payload.items() if k != "control_image"}
        conditioning_meta = {k: v for k, v in conditioning.items() if k != "reference_image"}

        metadata = {
            "conditioning": conditioning_meta,
            "controlnet": control_meta,
            "lora_path": str(lora_path) if lora_path else None,
            "lora_scale": lora_scale,
            "seed": effective_seed,
            "edit_type": request.edit_type.value,
            "control_type": request.control_type.value,
            "face_lock_mask": bool(self.enable_face_lock_mask and request.edit_type.value == "inpaint"),
        }
        return generated, metadata
