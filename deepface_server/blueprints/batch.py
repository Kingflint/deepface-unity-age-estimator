"""Batch analyze endpoint."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from ..errors import DeepFaceServerError
from ..schemas import validate_batch_request

batch_bp = Blueprint("batch", __name__)


@batch_bp.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    settings = current_app.config["SETTINGS"]
    images = validate_batch_request(request.get_json(silent=True), max_items=settings.max_batch_size)

    image_service = current_app.extensions["image_service"]
    deepface_service = current_app.extensions["deepface_service"]
    cache = current_app.extensions.get("cache")

    results = []
    for index, encoded in enumerate(images):
        try:
            raw = image_service.decode_b64(encoded)
            fp = image_service.fingerprint(raw)
            if cache is not None and (cached := cache.get(fp)) is not None:
                results.append({"index": index, "ok": True, "cached": True, "result": cached})
                continue
            img = image_service.to_ndarray(raw)
            result = deepface_service.analyze(img)
            if cache is not None:
                cache.set(fp, result)
            results.append({"index": index, "ok": True, "cached": False, "result": result})
        except DeepFaceServerError as err:
            results.append({"index": index, "ok": False, "error": str(err), "code": err.error_code})
    ok_count = sum(1 for r in results if r["ok"])
    return jsonify({"results": results, "succeeded": ok_count, "total": len(results)})
