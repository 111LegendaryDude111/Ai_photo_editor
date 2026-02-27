from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class EditType(str, Enum):
    IMG2IMG = "img2img"
    INPAINT = "inpaint"


class ControlType(str, Enum):
    NONE = "none"
    DEPTH = "depth"
    POSE = "pose"


class GenerationRequest(BaseModel):
    reference_image_b64: str = Field(..., description="Base64-encoded reference image")
    prompt: str = Field(..., min_length=3)
    negative_prompt: str | None = None
    edit_type: EditType = EditType.IMG2IMG
    control_type: ControlType = ControlType.NONE
    edit_mask_b64: str | None = None
    lora_id: str | None = None
    lora_scale: float | None = Field(default=None, ge=0.0, le=1.5)
    seed: int = 42

    @model_validator(mode="after")
    def validate_mask_for_inpaint(self) -> "GenerationRequest":
        if self.edit_type == EditType.INPAINT and not self.edit_mask_b64:
            raise ValueError("edit_mask_b64 is required when edit_type=inpaint")
        return self


class Metrics(BaseModel):
    identity_similarity: float
    clip_score: float
    artifact_score: float
    latency_ms: float


class GenerationResult(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    status: str
    image_url: str | None = None
    metadata_url: str | None = None
    rejection_reason: str | None = None
    metrics: Metrics | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    pair_id: str
    is_same_person: bool
    similarity: float


class DatasetPrepareRequest(BaseModel):
    image_paths: list[str]
    regularization_paths: list[str] = Field(default_factory=list)
    dataset_id: str


class LoraTrainRequest(BaseModel):
    dataset_id: str
    person_token: str = "<person_token>"
    rank: int = Field(default=16, ge=8, le=32)
    learning_rate: float = Field(default=1e-4, ge=5e-6, le=1e-3)
    epochs: int = Field(default=12, ge=1, le=100)


class ValidationDecision(BaseModel):
    status: str
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class SegmentationRequest(BaseModel):
    reference_image_b64: str
