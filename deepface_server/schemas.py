"""Request payload validation."""
from __future__ import annotations

from .errors import BadRequest


def validate_analyze_request(payload):
    if not isinstance(payload, dict):
        raise BadRequest("Request body must be a JSON object")
    image = payload.get("image")
    if not image:
        raise BadRequest("No image data provided")
    if not isinstance(image, str):
        raise BadRequest("'image' must be a base64-encoded string")
    return image