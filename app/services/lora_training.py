from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.domain.schemas import LoraTrainRequest


@dataclass
class LoraTrainingService:
    output_dir: Path

    def train(self, request: LoraTrainRequest) -> dict[str, str | float | int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        version = f"lora-{request.dataset_id}-{uuid4().hex[:8]}"
        artifact_path = self.output_dir / f"{version}.safetensors"

        # Placeholder artifact for pipeline integration tests.
        artifact_path.write_bytes(b"LORA_WEIGHTS_PLACEHOLDER")

        identity_similarity = 0.86
        prompt_adherence = 0.79
        overfit_score = 0.12

        metadata = {
            "version": version,
            "dataset_id": request.dataset_id,
            "person_token": request.person_token,
            "rank": request.rank,
            "learning_rate": request.learning_rate,
            "epochs": request.epochs,
            "identity_similarity": identity_similarity,
            "prompt_adherence": prompt_adherence,
            "overfit_score": overfit_score,
            "artifact_path": str(artifact_path),
        }

        metadata_path = self.output_dir / f"{version}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
