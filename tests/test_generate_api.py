import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import image_to_b64, make_face_like_image

client = TestClient(app)


@pytest.mark.parametrize(
    "prompt",
    [
        "cinematic portrait in warm golden light, highly detailed skin texture, shallow depth of field",
        "editorial fashion portrait with modern studio setup, crisp details, realistic colors and contrast",
        "outdoor portrait at sunset with soft bokeh background, photo-realistic look and natural facial details",
        "professional headshot with nuanced lighting, realistic hair strands, detailed eyes and sharp composition",
        "lifestyle portrait in urban environment with rich textures, realistic camera optics and balanced exposure",
    ],
)
def test_generate_endpoint_five_successful_generations(prompt: str) -> None:
    image_b64 = image_to_b64(make_face_like_image())
    payload = {
        "reference_image_b64": image_b64,
        "prompt": prompt,
        "negative_prompt": "distorted face, blurry image, extra fingers",
        "edit_type": "img2img",
        "control_type": "depth",
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["image_url"]
    assert body["metadata_url"]
    assert body["metrics"]["identity_similarity"] >= 0.75


def test_inpaint_generate_success() -> None:
    image = make_face_like_image()
    mask = make_face_like_image().convert("L")

    payload = {
        "reference_image_b64": image_to_b64(image),
        "prompt": "portrait with modified hairstyle and preserved identity",
        "negative_prompt": "artifact",
        "edit_type": "inpaint",
        "control_type": "none",
        "edit_mask_b64": image_to_b64(mask.convert("RGB")),
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] in {"completed", "rejected"}


def test_auto_mask_endpoint() -> None:
    payload = {"reference_image_b64": image_to_b64(make_face_like_image())}
    response = client.post("/masks/auto", json=payload)

    assert response.status_code == 200
    masks = response.json()
    assert set(masks.keys()) == {"face", "hair", "clothing", "background"}
