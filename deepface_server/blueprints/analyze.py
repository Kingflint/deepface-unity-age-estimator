"""Single-image analyze endpoint."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from ..schemas import validate_analyze_request

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    encoded = validate_analyze_request(request.get_json(silent=True))
    image_service = current_app.extensions["image_service"]
    deepface_service = current_app.extensions["deepface_service"]
    cache = current_app.extensions.get("cache")

    raw = image_service.decode_b64(encoded)
    fingerprint = image_service.fingerprint(raw)

    if cache is not None:
        cached = cache.get(fingerprint)
        if cached is not None:
            return jsonify({"result": cached, "cached": True})

    img = image_service.to_ndarray(raw)
    result = deepface_service.analyze(img)

    if cache is not None:
        cache.set(fingerprint, result)
    return jsonify({"result": result, "cached": False})
