from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOTO_EDITOR_", env_file=".env", extra="ignore")

    app_name: str = "Identity-Preserved Photo Editing"
    debug: bool = False
    enable_real_models: bool = False
    enable_cpu_offload: bool = True
    release_models_between_stages: bool = True
    torch_device: str = "cuda"
    torch_dtype: str = "float16"
    model_cache_dir: Path | None = None

    generation_cfg: float = 7.0
    generation_steps: int = 30
    refiner_strength: float = 0.3
    default_lora_scale: float = 0.8
    generation_batch_size: int = Field(default=1, ge=1, le=8)

    arcface_model_name: str = "buffalo_l"
    arcface_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"
    sdxl_base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter-FaceID"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter-faceid-plusv2_sdxl.bin"
    controlnet_depth_model_id: str = "diffusers/controlnet-depth-sdxl-1.0"
    controlnet_pose_model_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
    depth_estimator_model_id: str = "Intel/dpt-hybrid-midas"
    clip_model_id: str = "openai/clip-vit-large-patch14"
    face_parsing_model_id: str = "jonathandinu/face-parsing"
    sam_model_type: str = "vit_h"
    sam_checkpoint_path: str | None = None
    nsfw_model_id: str = "Falconsai/nsfw_image_detection"

    identity_threshold: float = 0.85
    identity_soft_warning_threshold: float = 0.75
    clip_threshold: float = 0.8
    nsfw_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    artifact_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_nsfw_filter: bool = True

    max_quality_retries: int = Field(default=3, ge=0, le=5)
    retry_seed_increment: int = Field(default=1, ge=1, le=1000)
    retry_lora_scale_decay: float = Field(default=0.9, ge=0.1, le=1.0)
    ab_variants: int = Field(default=2, ge=1, le=4)
    enable_face_lock_mask: bool = True

    input_max_resolution: int = Field(default=2048, ge=256, le=8192)
    input_max_pixels: int = Field(default=2048 * 2048, ge=256 * 256)
    input_max_file_size_mb: float = Field(default=8.0, ge=0.5, le=50.0)
    input_min_faces: int = Field(default=1, ge=0, le=8)
    input_max_faces: int = Field(default=1, ge=1, le=8)

    auth_enabled: bool = False
    api_keys: str = ""
    jwt_secret: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = None
    rate_limit_enabled: bool = True
    rate_limit_per_user_per_minute: int = Field(default=60, ge=1, le=5000)
    rate_limit_per_ip_per_minute: int = Field(default=120, ge=1, le=10000)

    local_output_dir: Path = Field(default=Path("data/generated"))
    local_metadata_dir: Path = Field(default=Path("data/metadata"))
    local_lora_dir: Path = Field(default=Path("data/lora"))

    use_minio: bool = False
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_bucket: str = "generated-images"

    use_postgres: bool = False
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/photo_editor"

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_batch_size: int = Field(default=4, ge=1, le=64)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
