"""Health and version endpoints."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/", methods=["GET"])
def index():
    return jsonify({"message": "DeepFace API is running!"})


@health_bp.route("/healthz", methods=["GET"])
def healthz():
    settings = current_app.config["SETTINGS"]
    return jsonify(
        {
            "status": "ok",
            "version": current_app.config["VERSION"],
            "deepface": current_app.extensions["deepface_service"].describe(),
            "limits": {
                "max_image_bytes": settings.max_image_bytes,
                "max_image_dimension": settings.max_image_dimension,
                "max_batch_size": settings.max_batch_size,
            },
        }
    )
