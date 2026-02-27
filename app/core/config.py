from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOTO_EDITOR_", env_file=".env", extra="ignore")

    app_name: str = "Identity-Preserved Photo Editing"
    debug: bool = False

    generation_cfg: float = 7.0
    generation_steps: int = 30
    refiner_strength: float = 0.3
    default_lora_scale: float = 0.8

    identity_threshold: float = 0.85
    identity_soft_warning_threshold: float = 0.75
    clip_threshold: float = 0.8

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
