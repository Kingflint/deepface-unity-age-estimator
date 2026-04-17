import base64
import io
import json

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _png_b64(size=8, color=(128, 128, 128)) -> str:
    img = Image.new("RGB", (size, size), color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_analyze_success_invokes_deepface_with_configured_actions(client):
    from deepface import DeepFace

    DeepFace.calls.clear()

    encoded = _png_b64()
    resp = client.post(
        "/analyze",
        data=json.dumps({"image": encoded}),
        content_type="application/json",
    )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["cached"] is False
    assert body["result"][0]["age"] == 30

    assert DeepFace.calls, "DeepFace.analyze was not invoked"
    last = DeepFace.calls[-1]
    assert last["actions"] == ["emotion", "age", "gender"]
    assert last["enforce_detection"] is False
    assert last["detector_backend"] == "opencv"


def test_analyze_uses_cache_on_second_identical_request(client):
    encoded = _png_b64(color=(10, 20, 30))
    first = client.post(
        "/analyze",
        data=json.dumps({"image": encoded}),
        content_type="application/json",
    )
    second = client.post(
        "/analyze",
        data=json.dumps({"image": encoded}),
        content_type="application/json",
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["cached"] is False
    assert second.get_json()["cached"] is True
