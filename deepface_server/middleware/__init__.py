"""Middleware registration."""
from flask import Flask

from ..config import Settings
from .request_id import register_request_id
from .timing import register_timing


def register_middleware(app: Flask, settings: Settings) -> None:
    register_request_id(app)
    register_timing(app)