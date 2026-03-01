from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.infra.metadata_repo import LocalMetadataRepository, MetadataRepository, PostgresMetadataRepository
from app.infra.storage import LocalStorageClient, MinioStorageClient, StorageClient
from app.services.artifact_detection import ArtifactDetectionService
from app.services.controlnet import ControlNetService
from app.services.generation_agent import GenerationAgent
from app.services.identity import ArcFaceIdentityExtractor
from app.services.inpainting import InpaintingService
from app.services.input_validation import InputValidationService
from app.services.ip_adapter import IPAdapterFaceConditioner
from app.services.lora import LoraManager
from app.services.metrics import MetricsService
from app.services.model_registry import ModelRegistry
from app.services.orchestrator import OrchestratorService
from app.services.prompt_adherence import PromptAdherenceService
from app.services.safety import ContentSafetyService
from app.services.sdxl_generator import SDXLGeneratorService
from app.services.segmentation import AutoSegmentationService
from app.services.training_pipeline import TrainingPipelineService
from app.services.validators import IdentityValidator, QualityGateExecutor
from app.services.dataset_prep import DatasetPreparationService
from app.services.lora_training import LoraTrainingService


@lru_cache(maxsize=1)
def get_storage() -> StorageClient:
    s = get_settings()
    if s.use_minio:
        return MinioStorageClient(
            endpoint=s.minio_endpoint,
            access_key=s.minio_access_key,
            secret_key=s.minio_secret_key,
            bucket=s.minio_bucket,
        )
    return LocalStorageClient(output_dir=s.local_output_dir)


@lru_cache(maxsize=1)
def get_metadata_repo() -> MetadataRepository:
    s = get_settings()
    if s.use_postgres:
        return PostgresMetadataRepository(dsn=s.postgres_dsn)
    return LocalMetadataRepository(metadata_dir=s.local_metadata_dir)


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(settings=get_settings())


@lru_cache(maxsize=1)
def get_orchestrator() -> OrchestratorService:
    settings = get_settings()
    model_registry = get_model_registry()

    identity_extractor = ArcFaceIdentityExtractor(model_registry=model_registry)
    identity_validator = IdentityValidator(
        extractor=identity_extractor,
        hard_threshold=settings.identity_threshold,
        soft_threshold=settings.identity_soft_warning_threshold,
    )

    quality_gates = QualityGateExecutor(
        identity_validator=identity_validator,
        clip_validator=PromptAdherenceService(model_registry=model_registry),
        artifact_detector=ArtifactDetectionService(),
        safety_filter=ContentSafetyService(model_registry=model_registry),
        clip_threshold=settings.clip_threshold,
        artifact_threshold=settings.artifact_threshold,
        nsfw_threshold=settings.nsfw_threshold,
        enable_nsfw_filter=settings.enable_nsfw_filter,
    )

    segmentation = AutoSegmentationService(model_registry=model_registry)
    generation_agent = GenerationAgent(
        sdxl=SDXLGeneratorService(model_registry=model_registry),
        ip_adapter=IPAdapterFaceConditioner(),
        controlnet=ControlNetService(model_registry=model_registry),
        inpainting=InpaintingService(),
        lora_manager=LoraManager(settings.local_lora_dir),
        segmentation=segmentation,
        default_cfg=settings.generation_cfg,
        default_steps=settings.generation_steps,
        default_refiner_strength=settings.refiner_strength,
        default_lora_scale=settings.default_lora_scale,
        enable_face_lock_mask=settings.enable_face_lock_mask,
    )

    return OrchestratorService(
        generation_agent=generation_agent,
        identity_extractor=identity_extractor,
        quality_gates=quality_gates,
        storage=get_storage(),
        metadata_repository=get_metadata_repo(),
        metrics=MetricsService(),
        input_validator=InputValidationService(
            identity_extractor=identity_extractor,
            max_resolution=settings.input_max_resolution,
            max_pixels=settings.input_max_pixels,
            max_file_size_mb=settings.input_max_file_size_mb,
            min_faces=settings.input_min_faces,
            max_faces=settings.input_max_faces,
        ),
        max_quality_retries=settings.max_quality_retries,
        retry_seed_increment=settings.retry_seed_increment,
        retry_lora_scale_decay=settings.retry_lora_scale_decay,
        default_ab_variants=settings.ab_variants,
    )


@lru_cache(maxsize=1)
def get_training_pipeline() -> TrainingPipelineService:
    settings = get_settings()
    return TrainingPipelineService(
        dataset_prep=DatasetPreparationService(output_root=settings.local_metadata_dir.parent / "datasets"),
        lora_training=LoraTrainingService(output_dir=settings.local_lora_dir),
    )


@lru_cache(maxsize=1)
def get_segmentation_service() -> AutoSegmentationService:
    return AutoSegmentationService(model_registry=get_model_registry())
