from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.domain.errors import GpuOutOfMemoryError
from app.domain.schemas import ControlType
from app.services.model_registry import ModelRegistry
from app.services.torch_utils import get_inference_context, maybe_import_torch

logger = logging.getLogger(__name__)


@dataclass
class SDXLGeneratorService:
    """SDXL base+refiner service with optional real diffusers backend."""

    model_registry: ModelRegistry | None = None
    default_strength: float = 0.8

    @staticmethod
    def _placeholder_generate(
        reference_image: Image.Image,
        prompt: str,
        negative_prompt: str | None,
        cfg: float,
        steps: int,
        seed: int,
        refiner_strength: float,
        conditioning: dict[str, Any] | None = None,
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

    def _select_base_pipeline(self, control_type: ControlType):
        if self.model_registry is None:
            return None, "none"
        if control_type == ControlType.DEPTH:
            return self.model_registry.get_sdxl_depth(), "sdxl_depth"
        if control_type == ControlType.POSE:
            return self.model_registry.get_sdxl_pose(), "sdxl_pose"
        return self.model_registry.get_sdxl_base(), "sdxl_base"

    @staticmethod
    def _build_generator(seed: int):
        torch = maybe_import_torch()
        if torch is None:
            return None
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(seed)
        return generator

    def _apply_lora(self, pipeline: Any, lora_path: Path | None, lora_scale: float) -> None:
        if lora_path is None or pipeline is None:
            return
        if not hasattr(pipeline, "load_lora_weights"):
            return
        try:  # pragma: no cover - runtime env specific
            pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name="identity")
            if hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters(["identity"], adapter_weights=[lora_scale])
        except Exception as exc:
            logger.warning("LoRA injection failed, continuing without LoRA: %s", exc)

    def _generate_with_pipeline(
        self,
        pipeline: Any,
        prompt: str,
        negative_prompt: str | None,
        image: Image.Image,
        cfg: float,
        steps: int,
        seed: int,
        strength: float,
        control_payload: dict[str, Any] | None,
        conditioning: dict[str, Any] | None,
    ) -> Image.Image:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "strength": strength,
        }
        generator = self._build_generator(seed)
        if generator is not None:
            kwargs["generator"] = generator

        control_image = control_payload.get("control_image") if control_payload else None
        if control_image is not None and "controlnet" in pipeline.__class__.__name__.lower():
            kwargs["control_image"] = control_image

        if conditioning and bool(conditioning.get("has_reference_image", False)):
            ref_image = conditioning.get("reference_image")
            if ref_image is not None and hasattr(pipeline, "load_ip_adapter"):
                cache_key = f"sdxl_pipe_{id(pipeline)}"
                if self.model_registry and self.model_registry.ensure_ip_adapter(pipeline, cache_key=cache_key):
                    kwargs["ip_adapter_image"] = ref_image
                    if hasattr(pipeline, "set_ip_adapter_scale"):
                        pipeline.set_ip_adapter_scale(float(conditioning.get("ip_adapter_strength", 0.9)))

        with get_inference_context():
            result = pipeline(**kwargs)
        return result.images[0]

    def generate(
        self,
        reference_image: Image.Image,
        prompt: str,
        negative_prompt: str | None,
        cfg: float,
        steps: int,
        seed: int,
        refiner_strength: float,
        conditioning: dict[str, Any] | None = None,
        control_type: ControlType = ControlType.NONE,
        control_payload: dict[str, Any] | None = None,
        lora_path: Path | None = None,
        lora_scale: float = 1.0,
    ) -> Image.Image:
        base_pipe, pipe_name = self._select_base_pipeline(control_type)
        if base_pipe is None:
            return self._placeholder_generate(
                reference_image=reference_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg=cfg,
                steps=steps,
                seed=seed,
                refiner_strength=refiner_strength,
                conditioning=conditioning,
            )

        try:
            self._apply_lora(base_pipe, lora_path=lora_path, lora_scale=lora_scale)
            generated = self._generate_with_pipeline(
                pipeline=base_pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=reference_image,
                cfg=cfg,
                steps=steps,
                seed=seed,
                strength=self.default_strength,
                control_payload=control_payload,
                conditioning=conditioning,
            )

            if self.model_registry is not None:
                self.model_registry.release_stage("sdxl")

            refiner_pipe = self.model_registry.get_sdxl_refiner() if self.model_registry else None
            if refiner_pipe is None:
                return generated

            self._apply_lora(refiner_pipe, lora_path=lora_path, lora_scale=lora_scale)
            refined = self._generate_with_pipeline(
                pipeline=refiner_pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=generated,
                cfg=max(3.0, cfg - 1.0),
                steps=max(10, int(steps * 0.6)),
                seed=seed,
                strength=max(0.1, min(refiner_strength, 0.95)),
                control_payload=None,
                conditioning=conditioning,
            )
            if self.model_registry is not None:
                self.model_registry.release_stage("refiner")
            return refined
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda oom" in message:
                if self.model_registry is not None:
                    self.model_registry.release_stage("sdxl")
                    self.model_registry.release_stage("refiner")
                raise GpuOutOfMemoryError("GPU OOM during SDXL inference") from exc
            logger.warning("Real SDXL pipeline failed (%s), fallback renderer is used", pipe_name)
            return self._placeholder_generate(
                reference_image=reference_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg=cfg,
                steps=steps,
                seed=seed,
                refiner_strength=refiner_strength,
                conditioning=conditioning,
            )
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("SDXL pipeline failed, fallback renderer is used: %s", exc)
            return self._placeholder_generate(
                reference_image=reference_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg=cfg,
                steps=steps,
                seed=seed,
                refiner_strength=refiner_strength,
                conditioning=conditioning,
            )
