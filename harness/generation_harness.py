from __future__ import annotations

from dataclasses import dataclass

from app.domain.schemas import ControlType, EditType, GenerationRequest
from app.infra.image_io import encode_base64_image
from app.services.orchestrator import OrchestratorService


@dataclass
class GenerationSweepResult:
    cfg: float
    lora_scale: float
    identity_similarity: float
    clip_score: float
    artifact_score: float


class GenerationHarness:
    def __init__(self, orchestrator: OrchestratorService) -> None:
        self.orchestrator = orchestrator

    def sweep(self, reference_image, prompt: str, cfg_values: list[float], lora_scales: list[float]) -> list[GenerationSweepResult]:
        image_b64 = encode_base64_image(reference_image)
        results: list[GenerationSweepResult] = []

        for cfg in cfg_values:
            self.orchestrator.generation_agent.default_cfg = cfg
            for scale in lora_scales:
                request = GenerationRequest(
                    reference_image_b64=image_b64,
                    prompt=prompt,
                    negative_prompt="distorted face",
                    edit_type=EditType.IMG2IMG,
                    control_type=ControlType.DEPTH,
                    lora_scale=scale,
                )
                response = self.orchestrator.process(request)
                if response.metrics is None:
                    continue
                results.append(
                    GenerationSweepResult(
                        cfg=cfg,
                        lora_scale=scale,
                        identity_similarity=response.metrics.identity_similarity,
                        clip_score=response.metrics.clip_score,
                        artifact_score=response.metrics.artifact_score,
                    )
                )
        return results
