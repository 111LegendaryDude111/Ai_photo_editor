from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MetadataRepository(ABC):
    @abstractmethod
    def save_metadata(self, job_id: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError


class LocalMetadataRepository(MetadataRepository):
    def __init__(self, metadata_dir: Path) -> None:
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, job_id: str, payload: dict[str, Any]) -> str:
        target = self.metadata_dir / f"{job_id}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)


class PostgresMetadataRepository(MetadataRepository):
    """Postgres adapter for metadata persistence."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install `psycopg[binary]` to use PostgresMetadataRepository") from exc

        self._psycopg = psycopg
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS generation_metadata (
                      job_id TEXT PRIMARY KEY,
                      payload JSONB NOT NULL
                    )
                    """
                )
            conn.commit()

    def save_metadata(self, job_id: str, payload: dict[str, Any]) -> str:
        with self._psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generation_metadata (job_id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (job_id)
                    DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (job_id, json.dumps(payload)),
                )
            conn.commit()
        return f"postgres://generation_metadata/{job_id}"
