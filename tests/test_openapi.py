from deepface_server.openapi import generate_openapi_spec


def test_openapi_top_level_keys():
    spec = generate_openapi_spec()
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"]
    assert "paths" in spec
    assert "components" in spec


def test_openapi_includes_core_paths():
    spec = generate_openapi_spec()
    paths = spec["paths"]
    for required in ["/", "/healthz", "/analyze", "/analyze/batch", "/jobs", "/history", "/metrics"]:
        assert required in paths


def test_openapi_security_schemes():
    spec = generate_openapi_spec()
    schemes = spec["components"]["securitySchemes"]
    assert "ApiKeyAuth" in schemes
    assert "BearerAuth" in schemes


def test_openapi_request_schemas_have_required_fields():
    spec = generate_openapi_spec()
    schemas = spec["components"]["schemas"]
    assert "image" in schemas["AnalyzeRequest"]["required"]
    assert "images" in schemas["BatchRequest"]["required"]


def test_openapi_blueprint_serves_spec(make_app):
    app = make_app()
    with app.test_client() as client:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["info"]["title"]
