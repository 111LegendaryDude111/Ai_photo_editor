from __future__ import annotations

from dataclasses import dataclass

from app.domain.schemas import DatasetPrepareRequest, LoraTrainRequest
from app.services.dataset_prep import DatasetPreparationService
from app.services.lora_training import LoraTrainingService


@dataclass
class TrainingPipelineService:
    dataset_prep: DatasetPreparationService
    lora_training: LoraTrainingService

    def prepare_dataset(self, request: DatasetPrepareRequest) -> dict:
        return self.dataset_prep.prepare(request)

    def train_lora(self, request: LoraTrainRequest) -> dict:
        return self.lora_training.train(request)
