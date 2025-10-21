"""Health endpoints."""
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/", methods=["GET"])
def index():
    return jsonify({"message": "DeepFace API is running!"})