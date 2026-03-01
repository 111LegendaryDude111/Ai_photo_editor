import app.services.prompt_adherence as prompt_adherence_module
from app.services.prompt_adherence import PromptAdherenceService
from tests.conftest import make_face_like_image


class _BrokenModel:
    def parameters(self):
        return iter(())


class _DummyProcessor:
    def __call__(self, *args, **kwargs):
        return {}


class _DummyRegistry:
    def get_clip_bundle(self):
        return {"model": _BrokenModel(), "processor": _DummyProcessor()}


def test_clip_runtime_failure_uses_fallback(monkeypatch):
    monkeypatch.setattr(prompt_adherence_module, "maybe_import_torch", lambda: object())
    service = PromptAdherenceService(model_registry=_DummyRegistry())

    score = service.score(make_face_like_image(), "cinematic portrait")

    assert 0.0 <= score <= 1.0
