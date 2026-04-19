import json

import pytest

from deepface_server import create_app
from deepface_server.config import Settings


@pytest.fixture
def secured_settings(settings):
    return Settings(
        port=settings.port,
        log_level=settings.log_level,
        debug=settings.debug,
        max_image_bytes=settings.max_image_bytes,
        max_image_dimension=settings.max_image_dimension,
        deepface_actions=settings.deepface_actions,
        enforce_detection=settings.enforce_detection,
        detector_backend=settings.detector_backend,
        enable_cache=settings.enable_cache,
        cache_max_entries=settings.cache_max_entries,
        api_keys=frozenset({"sekret"}),
        rate_limit_per_minute=0,
        max_batch_size=settings.max_batch_size,
    )


@pytest.fixture
def secured_client(secured_settings):
    return create_app(secured_settings).test_client()


def test_health_remains_public(secured_client):
    assert secured_client.get("/").status_code == 200
    assert secured_client.get("/healthz").status_code == 200


def test_analyze_requires_api_key(secured_client):
    resp = secured_client.post(
        "/analyze",
        data=json.dumps({"image": "abc"}),
        content_type="application/json",
    )
    assert resp.status_code == 401
    assert resp.get_json()["code"] == "unauthorized"


def test_analyze_accepts_x_api_key(secured_client):
    resp = secured_client.post(
        "/analyze",
        data=json.dumps({"image": ""}),
        content_type="application/json",
        headers={"X-API-Key": "sekret"},
    )
    # Auth passes -> validation rejects empty image with 400, not 401.
    assert resp.status_code == 400


def test_analyze_accepts_bearer_token(secured_client):
    resp = secured_client.post(
        "/analyze",
        data=json.dumps({"image": ""}),
        content_type="application/json",
        headers={"Authorization": "Bearer sekret"},
    )
    assert resp.status_code == 400
