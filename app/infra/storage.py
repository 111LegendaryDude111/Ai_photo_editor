from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageClient(ABC):
    @abstractmethod
    def save_image(self, job_id: str, image_bytes: bytes, suffix: str = ".png") -> str:
        raise NotImplementedError


class LocalStorageClient(StorageClient):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, job_id: str, image_bytes: bytes, suffix: str = ".png") -> str:
        target = self.output_dir / f"{job_id}{suffix}"
        target.write_bytes(image_bytes)
        return str(target)


class MinioStorageClient(StorageClient):
    """Minimal MinIO adapter. Requires `minio` package in production deployment."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket

        try:
            from minio import Minio  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install `minio` package to use MinioStorageClient") from exc

        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def save_image(self, job_id: str, image_bytes: bytes, suffix: str = ".png") -> str:
        from io import BytesIO

        object_name = f"{job_id}{suffix}"
        data = BytesIO(image_bytes)
        self._client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=data,
            length=len(image_bytes),
            content_type="image/png",
        )
        return f"minio://{self.bucket}/{object_name}"
