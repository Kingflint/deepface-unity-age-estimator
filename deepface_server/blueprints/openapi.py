"""GET /openapi.json blueprint."""
from flask import Blueprint, jsonify

from ..openapi import generate_openapi_spec

openapi_bp = Blueprint("openapi", __name__)


@openapi_bp.route("/openapi.json", methods=["GET"])
def openapi_spec():
    return jsonify(generate_openapi_spec())
