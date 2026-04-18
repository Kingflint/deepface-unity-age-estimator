import base64
import io
import json

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _png_b64() -> str:
    img = Image.new("RGB", (8, 8), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_batch_rejects_empty_array(client):
    resp = client.post(
        "/analyze/batch",
        data=json.dumps({"images": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "bad_request"


def test_batch_rejects_oversize(client):
    resp = client.post(
        "/analyze/batch",
        data=json.dumps({"images": ["x"] * 99}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "exceeds limit" in resp.get_json()["error"]


def test_batch_returns_per_item_status(client):
    encoded = _png_b64()
    resp = client.post(
        "/analyze/batch",
        data=json.dumps({"images": [encoded, "not!base64", encoded]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total"] == 3
    assert body["succeeded"] == 2
    assert body["results"][1]["ok"] is False
    assert body["results"][1]["code"] == "image_decode_error"
