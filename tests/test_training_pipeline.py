from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_face_like_image

client = TestClient(app)


def test_dataset_prepare_and_lora_train(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    reg = tmp_path / "reg.png"

    make_face_like_image().save(img1)
    make_face_like_image(hue_shift=20).save(img2)
    make_face_like_image(hue_shift=180).save(reg)

    prepare_payload = {
        "dataset_id": "unit-test-dataset",
        "image_paths": [str(img1), str(img2)],
        "regularization_paths": [str(reg)],
    }
    prepare_response = client.post("/datasets/prepare", json=prepare_payload)

    assert prepare_response.status_code == 200
    prepared = prepare_response.json()
    assert prepared["train_count"] >= 1
    assert prepared["regularization_count"] == 1

    train_payload = {
        "dataset_id": "unit-test-dataset",
        "person_token": "<person_token>",
        "rank": 16,
        "learning_rate": 0.0001,
        "epochs": 10,
    }
    train_response = client.post("/lora/train", json=train_payload)

    assert train_response.status_code == 200
    trained = train_response.json()
    assert trained["identity_similarity"] >= 0.85
    assert Path(trained["artifact_path"]).exists()
