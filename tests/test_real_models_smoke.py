import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import image_to_b64, make_face_like_image


@pytest.mark.skipif(
    os.getenv("PHOTO_EDITOR_ENABLE_REAL_MODELS", "false").lower() != "true"
    or os.getenv("RUN_REAL_MODEL_SMOKE", "0") != "1",
    reason="Real model smoke tests are disabled in this environment",
)
def test_generate_smoke_with_real_models() -> None:
    client = TestClient(app)
    payload = {
        "reference_image_b64": image_to_b64(make_face_like_image()),
        "prompt": "photorealistic portrait in studio lighting",
        "negative_prompt": "distorted face, low quality, blurry",
        "edit_type": "img2img",
        "control_type": "depth",
        "ab_variants": 1,
        "max_retries": 0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] in {"completed", "rejected", "processing"}
