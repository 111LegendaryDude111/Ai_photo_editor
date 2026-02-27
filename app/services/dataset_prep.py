from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.domain.schemas import DatasetPrepareRequest


@dataclass
class DatasetPreparationService:
    output_root: Path

    def prepare(self, request: DatasetPrepareRequest) -> dict[str, str | int]:
        dataset_dir = self.output_root / request.dataset_id
        train_dir = dataset_dir / "train"
        reg_dir = dataset_dir / "regularization"
        manifests_dir = dataset_dir / "manifests"

        for d in (train_dir, reg_dir, manifests_dir):
            d.mkdir(parents=True, exist_ok=True)

        dedup_hashes: set[str] = set()
        kept_train: list[str] = []

        for path in request.image_paths:
            src = Path(path)
            if not src.exists():
                continue
            image = Image.open(src).convert("RGB")
            aligned = self._align_face_stub(image)
            ahash = self._average_hash(aligned)
            if ahash in dedup_hashes:
                continue
            dedup_hashes.add(ahash)

            target = train_dir / src.name
            aligned.save(target)
            kept_train.append(str(target))

        kept_reg: list[str] = []
        for path in request.regularization_paths:
            src = Path(path)
            if not src.exists():
                continue
            target = reg_dir / src.name
            Image.open(src).convert("RGB").save(target)
            kept_reg.append(str(target))

        captions = {img_path: self._auto_caption(Path(img_path).name) for img_path in kept_train}
        manifest_path = manifests_dir / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_id": request.dataset_id,
                    "train_images": kept_train,
                    "regularization_images": kept_reg,
                    "captions": captions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "dataset_id": request.dataset_id,
            "train_count": len(kept_train),
            "regularization_count": len(kept_reg),
            "manifest_path": str(manifest_path),
        }

    @staticmethod
    def _align_face_stub(image: Image.Image) -> Image.Image:
        return image.resize((768, 1024))

    @staticmethod
    def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
        arr = np.asarray(image.convert("L").resize((hash_size, hash_size)), dtype=np.float32)
        avg = float(arr.mean())
        bits = "".join("1" if px > avg else "0" for px in arr.flatten())
        return hashlib.sha1(bits.encode("utf-8")).hexdigest()

    @staticmethod
    def _auto_caption(filename: str) -> str:
        stem = Path(filename).stem.replace("_", " ")
        return f"portrait photo of <person_token>, {stem}".strip()
