from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.domain.errors import MissingLoraError


@dataclass
class LoraManager:
    lora_dir: Path

    def resolve(self, lora_id: str | None) -> Path | None:
        if not lora_id:
            return None
        candidate = self.lora_dir / f"{lora_id}.safetensors"
        if not candidate.exists():
            raise MissingLoraError(f"LoRA checkpoint not found: {lora_id}")
        return candidate

    @staticmethod
    def normalize_scale(scale: float | None, default_scale: float) -> float:
        if scale is None:
            return default_scale
        return max(0.0, min(scale, 1.5))
