"""Attach a request id to every response (read from header or generated)."""
from __future__ import annotations

import uuid

from flask import Flask, g, request

HEADER = "X-Request-ID"


def register_request_id(app: Flask) -> None:
    @app.before_request
    def _assign():
        g.request_id = request.headers.get(HEADER) or uuid.uuid4().hex

    @app.after_request
    def _emit(response):
        rid = getattr(g, "request_id", None)
        if rid:
            response.headers[HEADER] = rid
        return response
