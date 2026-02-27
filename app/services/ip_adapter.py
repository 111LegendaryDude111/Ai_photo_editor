from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IPAdapterFaceConditioner:
    default_strength: float = 0.9

    def build_conditioning(self, embedding: np.ndarray, strength: float | None = None) -> dict[str, float]:
        applied_strength = strength if strength is not None else self.default_strength
        return {
            "ip_adapter_strength": float(applied_strength),
            "embedding_magnitude": float(np.linalg.norm(embedding)),
        }
