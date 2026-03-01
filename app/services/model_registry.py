from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from app.core.config import Settings
from app.services.torch_utils import clear_cuda_cache, cuda_available, maybe_import_torch

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    settings: Settings
    _models: dict[str, Any] = field(default_factory=dict, init=False)
    _locks: dict[str, Lock] = field(default_factory=dict, init=False)

    def _get_or_load(self, name: str, loader: Callable[[], Any]) -> Any | None:
        if name in self._models:
            return self._models[name]

        lock = self._locks.setdefault(name, Lock())
        with lock:
            if name in self._models:
                return self._models[name]
            self._models[name] = loader()
            return self._models[name]

    def _torch_dtype(self):
        torch = maybe_import_torch()
        if torch is None:
            return None
        return getattr(torch, self.settings.torch_dtype, torch.float16)

    def _model_device(self) -> str:
        if self.settings.torch_device != "cuda":
            return "cpu"
        return "cuda" if cuda_available() else "cpu"

    def _hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        torch_dtype = self._torch_dtype()
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if self.settings.model_cache_dir:
            kwargs["cache_dir"] = str(self.settings.model_cache_dir)
        return kwargs

    def _apply_offload_or_device(self, pipeline: Any) -> Any:
        if self.settings.enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            try:
                pipeline.enable_model_cpu_offload()
                return pipeline
            except Exception as exc:  # pragma: no cover - runtime env specific
                logger.warning("enable_model_cpu_offload failed: %s", exc)

        if hasattr(pipeline, "to"):
            pipeline.to(self._model_device())
        return pipeline

    def release_model(self, name: str) -> None:
        model = self._models.pop(name, None)
        if model is None:
            return

        try:
            if hasattr(model, "to"):
                model.to("cpu")
        except Exception:  # pragma: no cover - runtime env specific
            pass

        del model
        gc.collect()
        clear_cuda_cache()

    def release_stage(self, stage: str) -> None:
        if not self.settings.release_models_between_stages:
            return

        stage_map = {
            "arcface": ["arcface"],
            "sdxl": ["sdxl_base", "sdxl_depth", "sdxl_pose"],
            "refiner": ["sdxl_refiner"],
        }
        for name in stage_map.get(stage, []):
            self.release_model(name)

    def release_all(self) -> None:
        for name in list(self._models.keys()):
            self.release_model(name)

    def get_arcface(self) -> Any | None:
        return self._get_or_load("arcface", self._load_arcface)

    def _load_arcface(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            import insightface
        except Exception as exc:
            logger.warning("insightface unavailable, using fallback identity extractor: %s", exc)
            return None

        providers = [item.strip() for item in self.settings.arcface_providers.split(",") if item.strip()]
        if not providers:
            providers = ["CPUExecutionProvider"]

        try:
            app = insightface.app.FaceAnalysis(name=self.settings.arcface_model_name, providers=providers)
            ctx_id = 0 if ("CUDAExecutionProvider" in providers and cuda_available()) else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            return app
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("ArcFace initialization failed, using fallback: %s", exc)
            return None

    def get_sdxl_base(self) -> Any | None:
        return self._get_or_load("sdxl_base", self._load_sdxl_base)

    def _load_sdxl_base(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from diffusers import StableDiffusionXLImg2ImgPipeline
        except Exception as exc:
            logger.warning("diffusers unavailable, using placeholder SDXL: %s", exc)
            return None

        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.settings.sdxl_base_model_id,
                **self._hf_kwargs(),
            )
            return self._apply_offload_or_device(pipe)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("SDXL base initialization failed, using placeholder: %s", exc)
            return None

    def _load_sdxl_controlnet(self, control_model_id: str) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
        except Exception as exc:
            logger.warning("ControlNet pipeline unavailable, using placeholder control flow: %s", exc)
            return None

        try:
            controlnet = ControlNetModel.from_pretrained(control_model_id, **self._hf_kwargs())
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.settings.sdxl_base_model_id,
                controlnet=controlnet,
                **self._hf_kwargs(),
            )
            return self._apply_offload_or_device(pipe)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("ControlNet pipeline initialization failed: %s", exc)
            return None

    def get_sdxl_depth(self) -> Any | None:
        return self._get_or_load("sdxl_depth", lambda: self._load_sdxl_controlnet(self.settings.controlnet_depth_model_id))

    def get_sdxl_pose(self) -> Any | None:
        return self._get_or_load("sdxl_pose", lambda: self._load_sdxl_controlnet(self.settings.controlnet_pose_model_id))

    def get_sdxl_refiner(self) -> Any | None:
        return self._get_or_load("sdxl_refiner", self._load_sdxl_refiner)

    def _load_sdxl_refiner(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from diffusers import StableDiffusionXLImg2ImgPipeline
        except Exception as exc:
            logger.warning("diffusers unavailable, skipping SDXL refiner: %s", exc)
            return None

        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.settings.sdxl_refiner_model_id,
                **self._hf_kwargs(),
            )
            return self._apply_offload_or_device(pipe)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("SDXL refiner initialization failed: %s", exc)
            return None

    def ensure_ip_adapter(self, pipeline: Any, cache_key: str) -> bool:
        loaded_marker = f"{cache_key}_ip_adapter_loaded"
        if self._models.get(loaded_marker):
            return True

        if not self.settings.enable_real_models or pipeline is None:
            return False

        if not hasattr(pipeline, "load_ip_adapter"):
            return False

        try:  # pragma: no cover - runtime env specific
            pipeline.load_ip_adapter(
                self.settings.ip_adapter_repo,
                subfolder=self.settings.ip_adapter_subfolder,
                weight_name=self.settings.ip_adapter_weight_name,
            )
            self._models[loaded_marker] = True
            return True
        except Exception as exc:
            logger.warning("IP-Adapter load failed, continuing without it: %s", exc)
            return False

    def get_clip_bundle(self) -> Any | None:
        return self._get_or_load("clip_bundle", self._load_clip_bundle)

    def _load_clip_bundle(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            logger.warning("transformers unavailable, using CLIP fallback scoring: %s", exc)
            return None

        try:
            model = CLIPModel.from_pretrained(self.settings.clip_model_id)
            processor = CLIPProcessor.from_pretrained(self.settings.clip_model_id)
            if hasattr(model, "to"):
                model.to(self._model_device())
            return {"model": model, "processor": processor}
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("CLIP initialization failed, using fallback scoring: %s", exc)
            return None

    def get_depth_estimator(self) -> Any | None:
        return self._get_or_load("depth_estimator", self._load_depth_estimator)

    def _load_depth_estimator(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import pipeline
        except Exception as exc:
            logger.warning("transformers unavailable, depth fallback will be used: %s", exc)
            return None

        try:
            device = 0 if self._model_device() == "cuda" else -1
            return pipeline("depth-estimation", model=self.settings.depth_estimator_model_id, device=device)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("Depth estimator initialization failed, fallback will be used: %s", exc)
            return None

    def get_pose_estimator(self) -> Any | None:
        return self._get_or_load("pose_estimator", self._load_pose_estimator)

    def _load_pose_estimator(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from controlnet_aux import DWposeDetector, OpenposeDetector
        except Exception:
            try:
                from controlnet_aux import OpenposeDetector
            except Exception as exc:
                logger.warning("OpenPose/DWPose unavailable, pose fallback will be used: %s", exc)
                return None
            try:
                return OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            except Exception as exc:  # pragma: no cover - runtime env specific
                logger.warning("OpenPose initialization failed: %s", exc)
                return None

        try:
            return DWposeDetector.from_pretrained("yzd-v/DWPose")
        except Exception:
            try:
                return OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            except Exception as exc:  # pragma: no cover - runtime env specific
                logger.warning("Pose detector initialization failed, fallback will be used: %s", exc)
                return None

    def get_face_parser(self) -> Any | None:
        return self._get_or_load("face_parser", self._load_face_parser)

    def _load_face_parser(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import pipeline
        except Exception as exc:
            logger.warning("transformers unavailable, face parsing fallback will be used: %s", exc)
            return None

        try:
            device = 0 if self._model_device() == "cuda" else -1
            return pipeline("image-segmentation", model=self.settings.face_parsing_model_id, device=device)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("Face parser initialization failed, fallback will be used: %s", exc)
            return None

    def get_sam_predictor(self) -> Any | None:
        return self._get_or_load("sam_predictor", self._load_sam_predictor)

    def _load_sam_predictor(self) -> Any | None:
        if not self.settings.enable_real_models or not self.settings.sam_checkpoint_path:
            return None
        try:  # pragma: no cover - optional dependency
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            logger.warning("segment-anything unavailable, SAM fallback will be used: %s", exc)
            return None

        try:
            model = sam_model_registry[self.settings.sam_model_type](checkpoint=self.settings.sam_checkpoint_path)
            if hasattr(model, "to"):
                model.to(self._model_device())
            return SamPredictor(model)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("SAM initialization failed, fallback will be used: %s", exc)
            return None

    def get_nsfw_classifier(self) -> Any | None:
        return self._get_or_load("nsfw_classifier", self._load_nsfw_classifier)

    def _load_nsfw_classifier(self) -> Any | None:
        if not self.settings.enable_real_models:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import pipeline
        except Exception as exc:
            logger.warning("transformers unavailable, NSFW fallback will be used: %s", exc)
            return None

        try:
            device = 0 if self._model_device() == "cuda" else -1
            return pipeline("image-classification", model=self.settings.nsfw_model_id, device=device)
        except Exception as exc:  # pragma: no cover - runtime env specific
            logger.warning("NSFW classifier initialization failed, fallback will be used: %s", exc)
            return None
