"""Single-image analyze endpoint."""
from flask import Blueprint, current_app, jsonify, request

from ..schemas import validate_analyze_request

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    encoded = validate_analyze_request(request.get_json(silent=True))
    image_service = current_app.extensions["image_service"]
    deepface_service = current_app.extensions["deepface_service"]

    raw = image_service.decode_b64(encoded)
    return jsonify({"result": deepface_service.analyze(raw)})