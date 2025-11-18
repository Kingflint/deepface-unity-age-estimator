"""Middleware registration."""
from __future__ import annotations

from flask import Flask

from ..config import Settings
from .auth import register_auth
from .rate_limit import register_rate_limit
from .request_id import register_request_id
from .timing import register_timing


def register_middleware(app: Flask, settings: Settings) -> None:
    register_request_id(app)
    register_auth(app, settings)
    register_rate_limit(app, settings)
    register_timing(app)
