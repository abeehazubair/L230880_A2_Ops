"""VisionGuard web app tests (Task 6.6 CI).

They run WITHOUT TensorFlow or a model file: the app is created in "remote" inference mode and the
predictor is replaced by a fake one. This tests everything around the model - upload validation,
file-name sanitising, the database trail, the API key and the health check.
"""
import io

import pytest
from PIL import Image

from app import web
from app.config import Config


class FakePredictor:
    def predict(self, data, filename):
        return {"label": "AI-generated", "confidence": 0.91, "p_ai": 0.91, "threshold": 0.66,
                "model_version": "VisionGuard test-double", "inference_ms": 1.0}

    def health(self):
        return {"mode": "fake", "model_loaded": True}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RemotePredictor", lambda *args, **kwargs: FakePredictor())

    class TestConfig(Config):
        DATABASE_URL = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
        INFERENCE_MODE = "remote"
        API_KEY = "test-key"
        MAX_UPLOAD_MB = 1

    return web.create_app(TestConfig).test_client()


def png_bytes(size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def upload(client, data, name, url="/predict", headers=None):
    return client.post(url, data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", headers=headers or {})


def test_health_reports_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["database"] == "ok"


def test_rejects_wrong_file_type(client):
    r = upload(client, b"just text", "notes.txt")
    assert r.status_code == 415
    assert b"Unsupported file type" in r.data


def test_rejects_file_over_size_limit(client):
    r = upload(client, b"\0" * (2 * 1024 * 1024), "huge.jpg")       # 2 MB > 1 MB test limit
    assert r.status_code == 413
    assert b"File too large" in r.data


def test_rejects_file_that_is_not_really_an_image(client):
    r = upload(client, b"this only pretends to be a jpeg", "fake.jpg")
    assert r.status_code == 422
    assert b"Not a valid image" in r.data


def test_valid_upload_is_shown_and_saved_in_history(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    history = client.get("/history")
    assert history.status_code == 200
    assert b"cat.png" in history.data
    assert b"VisionGuard test-double" in history.data


def test_file_name_is_sanitised(client):
    upload(client, png_bytes(), "../../etc/evil photo.png")
    history = client.get("/history").data
    assert b"etc_evil_photo.png" in history
    assert b"../" not in history


def test_api_requires_key_and_returns_json(client):
    assert upload(client, png_bytes(), "a.png", url="/api/predict").status_code == 401
    r = upload(client, png_bytes(), "a.png", url="/api/predict", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == "AI-generated" and 0 <= body["confidence"] <= 1


def test_unknown_page_gives_friendly_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert b"Error 404" in r.data
