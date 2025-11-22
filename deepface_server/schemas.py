"""Request payload validation.

We use plain Python so we don't pull in marshmallow/pydantic for what
amounts to a handful of simple shape checks.
"""
from __future__ import annotations

from typing import Any, Iterable

from .errors import BadRequest


def _require_dict(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise BadRequest("Request body must be a JSON object")
    return payload


def validate_analyze_request(payload: Any) -> str:
    """Return the base64-encoded image string from an /analyze request."""
    body = _require_dict(payload)
    image = body.get("image")
    if image is None or image == "":
        raise BadRequest("No image data provided")
    if not isinstance(image, str):
        raise BadRequest("'image' must be a base64-encoded string")
    return image


def validate_batch_request(payload: Any, *, max_items: int) -> list:
    """Return the list of base64-encoded images from an /analyze/batch request."""
    body = _require_dict(payload)
    images = body.get("images")
    if not isinstance(images, list) or not images:
        raise BadRequest("'images' must be a non-empty array")
    if len(images) > max_items:
        raise BadRequest(f"Batch size {len(images)} exceeds limit {max_items}")
    out: list[str] = []
    for index, item in enumerate(images):
        if not isinstance(item, str) or not item:
            raise BadRequest(f"images[{index}] must be a base64-encoded string")
        out.append(item)
    return out


def serialize_actions(actions: Iterable[str]) -> list[str]:
    return [str(a) for a in actions]
