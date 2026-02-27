from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.domain.errors import InvalidMaskError
from app.domain.schemas import GenerationRequest
from app.infra.image_io import decode_base64_image
from app.services.controlnet import ControlNetService
from app.services.inpainting import InpaintingService
from app.services.ip_adapter import IPAdapterFaceConditioner
from app.services.lora import LoraManager
from app.services.sdxl_generator import SDXLGeneratorService


@dataclass
class GenerationAgent:
    sdxl: SDXLGeneratorService
    ip_adapter: IPAdapterFaceConditioner
    controlnet: ControlNetService
    inpainting: InpaintingService
    lora_manager: LoraManager
    default_cfg: float
    default_steps: int
    default_refiner_strength: float
    default_lora_scale: float

    def generate(self, request: GenerationRequest, reference_image: Image.Image, identity_embedding) -> tuple[Image.Image, dict]:
        lora_path = self.lora_manager.resolve(request.lora_id)
        lora_scale = self.lora_manager.normalize_scale(request.lora_scale, self.default_lora_scale)

        conditioning = self.ip_adapter.build_conditioning(identity_embedding, strength=lora_scale)
        control_payload = self.controlnet.build_control_payload(request.control_type, reference_image)

        generated = self.sdxl.generate(
            reference_image=reference_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            cfg=self.default_cfg,
            steps=self.default_steps,
            seed=request.seed,
            refiner_strength=self.default_refiner_strength,
            conditioning=conditioning,
        )

        if request.edit_type.value == "inpaint":
            if not request.edit_mask_b64:
                raise InvalidMaskError("inpaint request requires mask")
            mask = decode_base64_image(request.edit_mask_b64).convert("L")
            if mask.size != generated.size:
                raise InvalidMaskError("mask size must match generated image size")
            generated = self.inpainting.apply_masked_edit(generated, mask, request.prompt, request.seed)

        metadata = {
            "conditioning": conditioning,
            "controlnet": control_payload,
            "lora_path": str(lora_path) if lora_path else None,
            "lora_scale": lora_scale,
            "edit_type": request.edit_type.value,
            "control_type": request.control_type.value,
        }
        return generated, metadata
