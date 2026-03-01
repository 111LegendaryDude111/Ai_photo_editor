from __future__ import annotations

import numpy as np
from PIL import Image


class ArtifactDetectionService:
    """Simple artifact detector with blur/noise heuristics."""

    def evaluate(self, image: Image.Image) -> dict[str, float]:
        gray = np.asarray(image.convert("L"), dtype=np.float32)

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        # Align gradient maps to the same spatial shape: (H-1, W-1).
        gx_aligned = gx[:-1, :]
        gy_aligned = gy[:, :-1]
        edge_strength = float(np.sqrt(gx_aligned**2 + gy_aligned**2).mean())

        noise_level = float(gray.std() / 255.0)
        # Calibrated for the placeholder generator: low edge detail implies blur,
        # but moderate texture/noise should not immediately force a reject.
        blur_score = max(0.0, 1.0 - min(edge_strength / 6.0, 1.0))
        face_artifact_prob = max(0.0, min((blur_score * 0.7) + (noise_level * 0.3), 1.0))
        hand_anomaly_score = max(0.0, min((noise_level * 0.8) + (max(0.0, edge_strength - 8.0) / 20.0), 1.0))
        hand_anomaly_detected = 1.0 if hand_anomaly_score > 0.75 else 0.0

        return {
            "edge_strength": edge_strength,
            "noise_level": noise_level,
            "blur_score": blur_score,
            "face_artifact_probability": face_artifact_prob,
            "hand_anomaly_score": hand_anomaly_score,
            "hand_anomaly_detected": hand_anomaly_detected,
        }
