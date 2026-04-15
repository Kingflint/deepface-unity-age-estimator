def test_index_returns_running_message(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.get_json() == {"message": "DeepFace API is running!"}


def test_index_only_accepts_get(client):
    resp = client.post("/")
    assert resp.status_code == 405
    body = resp.get_json()
    assert body["code"] == "method_not_allowed"


def test_healthz_exposes_version_and_limits(client, app):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["version"] == app.config["VERSION"]
    assert body["limits"]["max_image_bytes"] == 64 * 1024
    assert body["deepface"]["actions"] == ["emotion", "age", "gender"]


def test_unknown_path_returns_404_envelope(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json()["code"] == "not_found"


def test_response_includes_request_id_and_timing(client):
    resp = client.get("/healthz")
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-ms" in resp.headers
