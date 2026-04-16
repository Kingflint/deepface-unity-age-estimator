import json


def test_analyze_missing_body_returns_400(client):
    resp = client.post("/analyze", data="", content_type="application/json")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["code"] == "bad_request"


def test_analyze_missing_image_field_returns_400(client):
    resp = client.post(
        "/analyze",
        data=json.dumps({"foo": "bar"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No image data provided"


def test_analyze_non_string_image_returns_400(client):
    resp = client.post(
        "/analyze",
        data=json.dumps({"image": 12345}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "bad_request"


def test_analyze_invalid_base64_returns_400(client):
    resp = client.post(
        "/analyze",
        data=json.dumps({"image": "not!base64"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "image_decode_error"
