"""Lightweight metrics endpoint (no Prometheus dependency)."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics", methods=["GET"])
def metrics():
    payload: dict = {"requests": current_app.extensions["metrics"].snapshot()}
    cache = current_app.extensions.get("cache")
    if cache is not None:
        payload["cache"] = cache.stats()
    return jsonify(payload)
