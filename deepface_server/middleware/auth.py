"""Optional API-key based authentication.

If the service has zero configured keys (the default), every request is
accepted. When at least one key is configured, requests must present
the ``X-API-Key`` header (or an ``Authorization: Bearer <key>`` header)
matching one of the configured keys.
"""
from __future__ import annotations

from flask import Flask, request

from ..config import Settings
from ..errors import Unauthorized

PUBLIC_PATHS = {"/", "/healthz", "/metrics"}


def _extract_key() -> str | None:
    header = request.headers.get("X-API-Key")
    if header:
        return header.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def register_auth(app: Flask, settings: Settings) -> None:
    if not settings.api_keys:
        return  # auth disabled

    @app.before_request
    def _enforce_api_key():
        if request.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return None
        provided = _extract_key()
        if provided is None or provided not in settings.api_keys:
            raise Unauthorized("Missing or invalid API key")
        return None
