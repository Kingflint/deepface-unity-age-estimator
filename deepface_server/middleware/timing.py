"""Record per-request timing into the metrics collector."""
from __future__ import annotations

import time

from flask import Flask, g, request


def register_timing(app: Flask) -> None:
    @app.before_request
    def _start():
        g.request_started = time.perf_counter()

    @app.after_request
    def _end(response):
        started = getattr(g, "request_started", None)
        if started is None:
            return response
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
        metrics = app.extensions.get("metrics")
        if metrics is not None:
            metrics.record(request.path, response.status_code, elapsed_ms)
        return response
