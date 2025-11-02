"""Middleware registration."""
from flask import Flask

from ..config import Settings


def register_middleware(app: Flask, settings: Settings) -> None:
    pass